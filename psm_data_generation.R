#!/usr/bin/env Rscript

# ============================================================================
# Data-generation script for propensity-score simulation studies
# Author: Seungman Kim (refactored for public release)
# License: MIT
# ============================================================================
# This script generates simulation datasets under user-specified correlation,
# treatment-prevalence, and replication settings. It is designed as a portable,
# stand-alone replacement for the original project-specific data-generation code.
# ============================================================================

suppressPackageStartupMessages({
  if (!requireNamespace("MASS", quietly = TRUE)) {
    stop("Package 'MASS' is required but not installed.")
  }
})

parse_args <- function(args) {
  defaults <- list(
    root_path = "./psm_simulation_data",
    n_rep = 200L,
    n_sample = 10000L,
    num_x = 18L,
    r_xxs = c(0.1, 0.3, 0.5),
    r_xys = c(0.1, 0.3, 0.5),
    r_xts = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    group_sizes = c(0.1, 0.25, 0.4),
    r_tys = c(0.0, 0.1, 0.3, 0.5),
    seed = 2026L,
    save_corr_mats = TRUE,
    perfect_sep_check = TRUE
  )

  if (length(args) == 0L) return(defaults)

  parse_value <- function(x) {
    if (grepl(",", x, fixed = TRUE)) {
      return(as.numeric(strsplit(x, ",", fixed = TRUE)[[1]]))
    }
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

  out$n_rep <- as.integer(out$n_rep)
  out$n_sample <- as.integer(out$n_sample)
  out$num_x <- as.integer(out$num_x)
  out$seed <- as.integer(out$seed)
  out
}

format_level <- function(prefix, value, suffix = "") {
  sprintf("%s%02d%s", prefix, round(100 * value), suffix)
}

cor_to_cov <- function(cor_mat, sds) {
  cor_mat * tcrossprod(sds)
}

validate_settings <- function(cfg) {
  if (cfg$num_x <= 0L) stop("num_x must be positive.")
  if (cfg$n_rep <= 0L) stop("n_rep must be positive.")
  if (cfg$n_sample <= 1L) stop("n_sample must be greater than 1.")
  if (cfg$num_x %% length(cfg$r_xys) != 0L) {
    stop("num_x must be divisible by length(r_xys).")
  }
  if (cfg$num_x %% length(cfg$r_xts) != 0L) {
    stop("num_x must be divisible by length(r_xts).")
  }
  if (any(cfg$group_sizes <= 0 | cfg$group_sizes >= 1)) {
    stop("group_sizes must contain proportions strictly between 0 and 1.")
  }
}

write_basic_info <- function(cfg, output_file) {
  lines <- c(
    format(Sys.time(), tz = "UTC", usetz = TRUE),
    "",
    paste("The number of samples:", cfg$n_sample),
    paste("The number of iterations:", cfg$n_rep),
    paste("The number of covariates:", cfg$num_x),
    paste("r_XXs:", paste(cfg$r_xxs, collapse = " ")),
    paste("r_XYs:", paste(cfg$r_xys, collapse = " ")),
    paste("r_XTs:", paste(cfg$r_xts, collapse = " ")),
    paste("r_TYs:", paste(cfg$r_tys, collapse = " ")),
    paste("Group size:", paste(cfg$group_sizes, collapse = " "))
  )
  writeLines(lines, con = output_file)
}

build_dataset <- function(num_x, n_sample, r_xx, r_xy_vec, r_xt_vec, r_ty, group_size) {
  temp_cor <- matrix(r_xx, nrow = num_x + 2L, ncol = num_x + 2L)
  diag(temp_cor) <- 1

  loc_xy <- rep(seq_along(r_xy_vec), each = num_x / length(r_xy_vec))
  loc_xt <- rep(seq_along(r_xt_vec), each = num_x / length(r_xt_vec))

  for (ix in seq_len(num_x)) {
    temp_cor[num_x + 1L, ix] <- r_xy_vec[loc_xy[ix]]
    temp_cor[ix, num_x + 1L] <- r_xy_vec[loc_xy[ix]]
    temp_cor[num_x + 2L, ix] <- r_xt_vec[loc_xt[ix]]
    temp_cor[ix, num_x + 2L] <- r_xt_vec[loc_xt[ix]]
  }

  temp_cor[num_x + 2L, num_x + 1L] <- r_ty
  temp_cor[num_x + 1L, num_x + 2L] <- r_ty

  cov_mat <- cor_to_cov(temp_cor, rep(1, num_x + 2L))
  latent <- MASS::mvrnorm(n = n_sample, mu = rep(0, num_x + 2L), Sigma = cov_mat)

  observed_t <- ifelse(rank(latent[, num_x + 2L], ties.method = "first") <= n_sample * group_size, 1, 0)
  df <- data.frame(latent[, seq_len(num_x + 1L)], T = observed_t)
  colnames(df) <- c(sprintf("X%d", seq_len(num_x)), "Y", "T")

  list(
    data = df,
    latent_cor = stats::cor(latent),
    glm_warning = NULL,
    glm_error = NULL
  )
}

check_perfect_separation <- function(df, num_x) {
  predictors <- sprintf("X%d", seq_len(num_x))
  formula_txt <- paste("T ~", paste(predictors, collapse = " + "))
  warning_msg <- NULL
  error_msg <- NULL

  fit <- withCallingHandlers(
    tryCatch(
      stats::glm(stats::as.formula(formula_txt), data = df, family = stats::binomial()),
      error = function(e) {
        error_msg <<- conditionMessage(e)
        NULL
      }
    ),
    warning = function(w) {
      warning_msg <<- conditionMessage(w)
      invokeRestart("muffleWarning")
    }
  )

  list(model = fit, warning = warning_msg, error = error_msg)
}

main <- function() {
  cfg <- parse_args(commandArgs(trailingOnly = TRUE))
  validate_settings(cfg)
  set.seed(cfg$seed)

  root_path <- normalizePath(cfg$root_path, winslash = "/", mustWork = FALSE)
  dir.create(root_path, recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(root_path, "datasets"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(root_path, "corr_mats"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(root_path, "perfect_sep"), recursive = TRUE, showWarnings = FALSE)

  write_basic_info(cfg, file.path(root_path, "basic_info.txt"))

  names_xx <- vapply(cfg$r_xxs, function(x) format_level("XX", x, "_"), character(1))
  names_ts <- vapply(cfg$group_sizes, function(x) format_level("TS", x), character(1))
  names_ty <- vapply(cfg$r_tys, function(x) format_level("TY", x, "_"), character(1))
  col_names <- c(sprintf("X%d", seq_len(cfg$num_x)), "Y", "T")

  perfect_messages <- character(0)
  perfect_count <- 0L
  converge_count <- 0L

  for (j in seq_along(cfg$r_xxs)) {
    for (k in seq_along(cfg$group_sizes)) {
      for (l in seq_along(cfg$r_tys)) {
        folder_name <- paste0(names_xx[[j]], names_ty[[l]], names_ts[[k]])
        folder_path <- file.path(root_path, "datasets", folder_name)
        dir.create(folder_path, recursive = TRUE, showWarnings = FALSE)
        message("Generating condition: ", folder_name)

        corr_list <- vector("list", cfg$n_rep)

        for (iter_idx in seq_len(cfg$n_rep)) {
          result <- build_dataset(
            num_x = cfg$num_x,
            n_sample = cfg$n_sample,
            r_xx = cfg$r_xxs[[j]],
            r_xy_vec = cfg$r_xys,
            r_xt_vec = cfg$r_xts,
            r_ty = cfg$r_tys[[l]],
            group_size = cfg$group_sizes[[k]]
          )

          df <- result$data
          corr_list[[iter_idx]] <- result$latent_cor
          dataset_name <- sprintf("%s_%04d.csv", folder_name, iter_idx)
          utils::write.csv(df, file.path(folder_path, dataset_name), row.names = FALSE)

          if (isTRUE(cfg$perfect_sep_check)) {
            check <- check_perfect_separation(df, cfg$num_x)
            if (!is.null(check$warning)) {
              if (grepl("fitted probabilities", check$warning, ignore.case = TRUE)) {
                perfect_count <- perfect_count + 1L
                perfect_messages <- c(perfect_messages, file.path(folder_path, dataset_name))
              }
              if (grepl("did not converge", check$warning, ignore.case = TRUE)) {
                converge_count <- converge_count + 1L
              }
            }
            if (!is.null(check$error)) {
              perfect_messages <- c(
                perfect_messages,
                paste0(dataset_name, ": ", check$error)
              )
            }
          }

          if (iter_idx %% max(1L, floor(cfg$n_rep / 10L)) == 0L || iter_idx == cfg$n_rep) {
            message("  Completed ", iter_idx, " / ", cfg$n_rep, " replications")
          }
        }

        if (isTRUE(cfg$save_corr_mats)) {
          arr <- array(unlist(corr_list), dim = c(cfg$num_x + 2L, cfg$num_x + 2L, cfg$n_rep))
          ave_cor <- round(apply(arr, c(1, 2), mean, na.rm = TRUE), 3)
          rownames(ave_cor) <- col_names
          colnames(ave_cor) <- col_names
          utils::write.csv(ave_cor, file.path(root_path, "corr_mats", paste0("corr_mat_", folder_name, ".csv")))
        }
      }
    }
  }

  if (length(perfect_messages) == 0L) perfect_messages <- "No perfect-separation or convergence issues were recorded."
  writeLines(perfect_messages, con = file.path(root_path, "perfect_sep", "Result_PerfectSeparation.csv"))
  writeLines(
    c(
      format(Sys.time(), tz = "UTC", usetz = TRUE),
      "",
      paste("Total number of perfect separation:", perfect_count),
      paste("Total number of convergence warnings:", converge_count)
    ),
    con = file.path(root_path, "perfect_sep", "Summary.txt")
  )

  message("Data generation completed successfully.")
}

main()
