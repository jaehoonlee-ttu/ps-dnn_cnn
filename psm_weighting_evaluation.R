#!/usr/bin/env Rscript

# ============================================================================
# Weighting / covariate-balance evaluation script
# Author: Seungman Kim (refactored for public release)
# License: MIT
# ============================================================================
# This script reads generated datasets plus one or more prediction directories
# (LR, DNN, CNN, or any custom propensity-score model), computes unweighted and
# weighted treatment-vs-control coefficient differences, and saves tidy outputs.
# ============================================================================

parse_args <- function(args) {
  defaults <- list(
    data_root = "./psm_simulation_data",
    output_dir = "./psm_results/weighting_evaluation",
    use_noised_data = FALSE,
    bundle_size = 50L,
    include_y = TRUE,
    clip_lower = 0.001,
    clip_upper = 0.999,
    methods = "",
    att_odds = TRUE,
    stabilized_iptw = TRUE,
    folder_index = NA_integer_
  )

  if (length(args) == 0L) return(defaults)

  parse_value <- function(x) {
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

parse_method_spec <- function(spec) {
  if (!nzchar(spec)) stop("You must provide --methods, e.g. 'LR=./results/LR,CNN=./results/CNN,DNN=./results/DNN'.")
  pairs <- strsplit(spec, ",", fixed = TRUE)[[1]]
  out <- list()
  for (pair in pairs) {
    kv <- strsplit(pair, "=", fixed = TRUE)[[1]]
    if (length(kv) != 2L) stop("Invalid method specification: ", pair)
    out[[trimws(kv[[1]])]] <- normalizePath(trimws(kv[[2]]), winslash = "/", mustWork = FALSE)
  }
  out
}

resolve_bundle_prediction <- function(method_root, condition_name, bundle_idx) {
  pattern <- sprintf("bundle_%03d_predictions\\.csv$", bundle_idx)
  hits <- list.files(method_root, pattern = pattern, recursive = TRUE, full.names = TRUE)
  hits <- hits[grepl(paste0("/", condition_name, "/"), gsub("\\\\", "/", hits))]
  if (length(hits) == 0L) {
    stop("No prediction file found for condition ", condition_name, ", bundle ", bundle_idx, " in ", method_root)
  }
  if (length(hits) > 1L) {
    stop(
      "Multiple matching prediction files found for condition ", condition_name,
      ", bundle ", bundle_idx, " in ", method_root,
      ". Provide a more specific method root containing only one run."
    )
  }
  hits[[1]]
}

split_bundle_predictions <- function(pred_file, bundle_files, clip_lower, clip_upper) {
  pred <- utils::read.csv(pred_file)
  if (!"ps" %in% names(pred)) stop("Prediction file must contain a column named 'ps': ", pred_file)
  row_counts <- vapply(bundle_files, function(path) nrow(utils::read.csv(path)), integer(1))
  if (sum(row_counts) != nrow(pred)) {
    stop("Prediction file row count does not match bundled dataset sizes: ", pred_file)
  }
  idx_end <- cumsum(row_counts)
  idx_start <- c(1L, head(idx_end, -1L) + 1L)
  lapply(seq_along(bundle_files), function(i) {
    pmin(pmax(pred$ps[idx_start[[i]]:idx_end[[i]]], clip_lower), clip_upper)
  })
}

calc_weights <- function(treat, ps, att_odds = TRUE, stabilized_iptw = TRUE) {
  out <- list()
  if (isTRUE(att_odds)) {
    out$odds <- ifelse(treat == 1, 1, ps / (1 - ps))
  }
  if (isTRUE(stabilized_iptw)) {
    out$iptw <- ifelse(treat == 1, mean(ps) / ps, mean(1 - ps) / (1 - ps))
  }
  out
}

estimate_effect <- function(y, treat, weights = NULL) {
  fit <- if (is.null(weights)) {
    stats::lm(y ~ treat)
  } else {
    stats::lm(y ~ treat, weights = weights)
  }
  unname(stats::coef(fit)[[2]])
}

parse_condition_meta <- function(condition_name) {
  m <- regexec("XX([0-9]{2})_TY([0-9]{2})_TS([0-9]{2})", condition_name)
  parts <- regmatches(condition_name, m)[[1]]
  if (length(parts) == 4L) {
    return(list(r_xx = as.numeric(parts[[2]]) / 100, r_ty = as.numeric(parts[[3]]) / 100, group_size = as.numeric(parts[[4]]) / 100))
  }
  list(r_xx = NA_real_, r_ty = NA_real_, group_size = NA_real_)
}

main <- function() {
  cfg <- parse_args(commandArgs(trailingOnly = TRUE))
  basic <- parse_basic_info(file.path(cfg$data_root, "basic_info.txt"))
  condition_dirs <- build_folder_names(basic)
  method_roots <- parse_method_spec(cfg$methods)
  source_subdir <- if (isTRUE(cfg$use_noised_data)) "datasets_noised" else "datasets"
  source_dir <- file.path(cfg$data_root, source_subdir)
  dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)

  selected_indices <- if (is.na(cfg$folder_index)) seq_along(condition_dirs) else cfg$folder_index + 1L
  all_rows <- list()
  row_id <- 1L

  for (idx in selected_indices) {
    condition_name <- condition_dirs[[idx]]
    meta <- parse_condition_meta(condition_name)
    condition_path <- file.path(source_dir, condition_name)
    dataset_files <- sort(list.files(condition_path, pattern = "^XX.*\\.csv$", full.names = TRUE))
    if (length(dataset_files) == 0L) next
    bundle_groups <- split(dataset_files, ceiling(seq_along(dataset_files) / cfg$bundle_size))

    method_bundle_cache <- lapply(names(method_roots), function(method_name) {
      method_root <- method_roots[[method_name]]
      bundle_predictions <- lapply(seq_along(bundle_groups), function(bundle_idx) {
        pred_file <- resolve_bundle_prediction(method_root, condition_name, bundle_idx - 1L)
        split_bundle_predictions(pred_file, bundle_groups[[bundle_idx]], cfg$clip_lower, cfg$clip_upper)
      })
      names(bundle_predictions) <- paste0("bundle_", seq_along(bundle_groups) - 1L)
      bundle_predictions
    })
    names(method_bundle_cache) <- names(method_roots)

    for (bundle_idx in seq_along(bundle_groups)) {
      for (local_idx in seq_along(bundle_groups[[bundle_idx]])) {
        dataset_path <- bundle_groups[[bundle_idx]][[local_idx]]
        df <- utils::read.csv(dataset_path)
        variables <- c(sprintf("X%d", seq_len(basic$num_x)), if (isTRUE(cfg$include_y) && "Y" %in% names(df)) "Y")
        treat <- df$T
        replication <- as.integer(sub(".*_([0-9]{4})\\.csv$", "\\1", basename(dataset_path)))

        for (variable in variables) {
          all_rows[[row_id]] <- data.frame(
            condition = condition_name,
            replication = replication,
            variable = variable,
            method = "Unweighted",
            weighting = "none",
            estimate = estimate_effect(df[[variable]], treat),
            r_xx = meta$r_xx,
            r_ty = meta$r_ty,
            group_size = meta$group_size,
            stringsAsFactors = FALSE
          )
          row_id <- row_id + 1L
        }

        for (method_name in names(method_bundle_cache)) {
          ps <- method_bundle_cache[[method_name]][[bundle_idx]][[local_idx]]
          weights <- calc_weights(treat, ps, att_odds = cfg$att_odds, stabilized_iptw = cfg$stabilized_iptw)
          for (weight_name in names(weights)) {
            for (variable in variables) {
              all_rows[[row_id]] <- data.frame(
                condition = condition_name,
                replication = replication,
                variable = variable,
                method = method_name,
                weighting = weight_name,
                estimate = estimate_effect(df[[variable]], treat, weights = weights[[weight_name]]),
                r_xx = meta$r_xx,
                r_ty = meta$r_ty,
                group_size = meta$group_size,
                stringsAsFactors = FALSE
              )
              row_id <- row_id + 1L
            }
          }
        }
      }
      message("Processed condition ", condition_name, ", bundle ", bundle_idx, " / ", length(bundle_groups))
    }
  }

  long_df <- do.call(rbind, all_rows)
  utils::write.csv(long_df, file.path(cfg$output_dir, "covariate_effects_long.csv"), row.names = FALSE)

  summary_df <- stats::aggregate(
    estimate ~ condition + variable + method + weighting + r_xx + r_ty + group_size,
    data = long_df,
    FUN = function(x) c(mean = mean(x), sd = stats::sd(x), median = stats::median(x))
  )

  summary_out <- data.frame(
    condition = summary_df$condition,
    variable = summary_df$variable,
    method = summary_df$method,
    weighting = summary_df$weighting,
    r_xx = summary_df$r_xx,
    r_ty = summary_df$r_ty,
    group_size = summary_df$group_size,
    mean_estimate = summary_df$estimate[, "mean"],
    sd_estimate = summary_df$estimate[, "sd"],
    median_estimate = summary_df$estimate[, "median"],
    stringsAsFactors = FALSE
  )
  utils::write.csv(summary_out, file.path(cfg$output_dir, "covariate_effects_summary.csv"), row.names = FALSE)

  message("Weighting evaluation completed successfully.")
}

main()
