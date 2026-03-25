#!/usr/bin/env Rscript

# ============================================================================
# Logistic-regression propensity-score estimation script
# Author: Seungman Kim (refactored for public release)
# License: MIT
# ============================================================================

parse_args <- function(args) {
  defaults <- list(
    data_root = "./psm_simulation_data",
    results_root = "./psm_results",
    bundle_size = 50L,
    use_noised_data = FALSE,
    include_y = FALSE,
    include_squared = TRUE,
    include_interactions = TRUE,
    clip_lower = 0.001,
    clip_upper = 0.999,
    maxit = 50L,
    folder_index = NA_integer_
  )

  if (length(args) == 0L) return(defaults)

  parse_value <- function(x) {
    if (grepl(",", x, fixed = TRUE)) return(strsplit(x, ",", fixed = TRUE)[[1]])
    if (tolower(x) %in% c("true", "false")) return(tolower(x) == "true")
    if (!is.na(suppressWarnings(as.numeric(x)))) return(as.numeric(x))
    x
  }

  out <- defaults
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) stop("Unexpected argument: ", key)
    key <- sub("^--", "", key)
    if (i == length(args) || startsWith(args[[i + 1L]], "--")) {
      out[[key]] <- TRUE
      i <- i + 1L
    } else {
      out[[key]] <- parse_value(args[[i + 1L]])
      i <- i + 2L
    }
  }

  out$bundle_size <- as.integer(out$bundle_size)
  out$maxit <- as.integer(out$maxit)
  out$folder_index <- as.integer(out$folder_index)
  out
}

parse_basic_info <- function(path) {
  lines <- readLines(path, warn = FALSE)
  get_nums <- function(prefix) {
    line <- lines[grepl(paste0("^", prefix), lines)]
    if (length(line) == 0L) return(numeric(0))
    nums <- regmatches(line, gregexpr("-?[0-9.]+", line))[[1]]
    as.numeric(nums)
  }

  list(
    n_sample = as.integer(get_nums("The number of samples")[[1]]),
    n_rep = as.integer(get_nums("The number of iterations")[[1]]),
    num_x = as.integer(get_nums("The number of covariates")[[1]]),
    r_xxs = get_nums("r_XXs"),
    r_tys = get_nums("r_TYs"),
    group_sizes = get_nums("Group size")
  )
}

build_folder_names <- function(basic) {
  names_xx <- sprintf("XX%02d_", round(100 * basic$r_xxs))
  names_ty <- sprintf("TY%02d_", round(100 * basic$r_tys))
  names_ts <- sprintf("TS%02d", round(100 * basic$group_sizes))
  as.vector(outer(outer(names_xx, names_ty, paste0), names_ts, paste0))
}

engineer_features <- function(df, num_x, include_y = FALSE, include_squared = TRUE, include_interactions = TRUE) {
  x_df <- df[, sprintf("X%d", seq_len(num_x)), drop = FALSE]
  parts <- list(x_df)

  if (isTRUE(include_y) && "Y" %in% names(df)) {
    parts <- c(parts, list(df["Y"]))
  }
  if (isTRUE(include_squared)) {
    sq <- x_df^2
    names(sq) <- paste0(names(x_df), "SQ")
    parts <- c(parts, list(sq))
  }
  if (isTRUE(include_interactions) && num_x >= 2L) {
    combs <- utils::combn(names(x_df), 2, simplify = FALSE)
    inter_list <- lapply(combs, function(pair) x_df[[pair[1]]] * x_df[[pair[2]]])
    inter_df <- as.data.frame(inter_list)
    names(inter_df) <- vapply(combs, function(pair) paste0(pair[1], "_", pair[2]), character(1))
    parts <- c(parts, list(inter_df))
  }

  out <- do.call(cbind, parts)
  data.frame(out, check.names = FALSE)
}

bundle_csv_files <- function(folder, bundle_size) {
  csv_files <- list.files(folder, pattern = "^XX.*\\.csv$", full.names = TRUE)
  csv_files <- sort(csv_files)
  if (length(csv_files) == 0L) stop("No simulation CSV files found in ", folder)
  if (length(csv_files) %% bundle_size != 0L) {
    stop("The number of CSV files in ", folder, " is not divisible by bundle_size = ", bundle_size)
  }
  split(csv_files, ceiling(seq_along(csv_files) / bundle_size))
}

fit_logistic_ps <- function(df, num_x, include_y, include_squared, include_interactions, maxit) {
  x <- engineer_features(df, num_x, include_y, include_squared, include_interactions)
  fit_df <- cbind(T = df$T, x)
  formula_txt <- paste("T ~", paste(names(x), collapse = " + "))
  model <- stats::glm(stats::as.formula(formula_txt), data = fit_df, family = stats::binomial(), control = list(maxit = maxit))
  stats::predict(model, type = "response")
}

save_json <- function(obj, path) {
  json <- paste(capture.output(jsonlite::toJSON(obj, pretty = TRUE, auto_unbox = TRUE)), collapse = "\n")
  writeLines(json, con = path)
}

main <- function() {
  suppressPackageStartupMessages({
    if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Package 'jsonlite' is required but not installed.")
  })

  cfg <- parse_args(commandArgs(trailingOnly = TRUE))
  basic <- parse_basic_info(file.path(cfg$data_root, "basic_info.txt"))
  condition_dirs <- build_folder_names(basic)

  source_subdir <- if (isTRUE(cfg$use_noised_data)) "datasets_noised" else "datasets"
  source_dir <- file.path(cfg$data_root, source_subdir)
  selected_indices <- if (is.na(cfg$folder_index)) seq_along(condition_dirs) else cfg$folder_index + 1L

  run_name <- paste0(
    "lr_features-",
    paste(c(
      if (cfg$include_y) "Y" else NULL,
      if (cfg$include_squared) "SQ" else NULL,
      if (cfg$include_interactions) "INT" else NULL
    ), collapse = "-"),
    "_clip-", cfg$clip_lower, "-", cfg$clip_upper,
    "_bundle-", cfg$bundle_size
  )
  run_name <- sub("features-_", "features-main_", run_name)

  for (idx in selected_indices) {
    folder_name <- condition_dirs[[idx]]
    bundle_files <- bundle_csv_files(file.path(source_dir, folder_name), cfg$bundle_size)
    out_dir <- file.path(cfg$results_root, "LR", folder_name, run_name)
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

    metadata <- list(
      folder_name = folder_name,
      source_dir = normalizePath(file.path(source_dir, folder_name), winslash = "/", mustWork = FALSE),
      bundle_size = cfg$bundle_size,
      feature_options = list(
        include_y = cfg$include_y,
        include_squared = cfg$include_squared,
        include_interactions = cfg$include_interactions
      ),
      model_options = list(
        maxit = cfg$maxit,
        clip_lower = cfg$clip_lower,
        clip_upper = cfg$clip_upper
      )
    )
    save_json(metadata, file.path(out_dir, "run_metadata.json"))

    for (bundle_idx in seq_along(bundle_files)) {
      files <- bundle_files[[bundle_idx]]
      parts <- lapply(files, utils::read.csv)
      bundle_df <- do.call(rbind, parts)
      ps <- fit_logistic_ps(
        df = bundle_df,
        num_x = basic$num_x,
        include_y = cfg$include_y,
        include_squared = cfg$include_squared,
        include_interactions = cfg$include_interactions,
        maxit = cfg$maxit
      )
      ps <- pmin(pmax(ps, cfg$clip_lower), cfg$clip_upper)
      utils::write.csv(data.frame(ps = ps), file.path(out_dir, sprintf("bundle_%03d_predictions.csv", bundle_idx - 1L)), row.names = FALSE)
      message("Saved LR predictions for ", folder_name, " bundle ", bundle_idx, " / ", length(bundle_files))
    }
  }

  message("Logistic-regression propensity-score estimation completed successfully.")
}

main()
