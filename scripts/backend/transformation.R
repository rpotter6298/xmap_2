## Basic-ish Maths
geometric_mean <- function(numbers) {
  gm <- prod(numbers)^(1 / length(numbers))
  return(gm)
}
## Diff Analysis
limma_funct <- function(data) {
  t_set <- t(data[-1:-2])
  design <- model.matrix(~ 0 + group, data = data)
  colnames(design) <- c("case", "control")
  contrasts <- makeContrasts(Diff = control - case, levels = design)
  fit <- lmFit(t_set, design, method = "robust", maxit = 1000)
  contrast_fit <- contrasts.fit(fit, contrasts)
  ebay_fit <- eBayes(contrast_fit)
  DE_results <- topTable(ebay_fit, n = ncol(data), adjust.method = "fdr", confint = TRUE)
  # print(summary(decideTests(ebay_fit)))
  return(DE_results)
}

## Wrappers
differential_reports <- function(dflist) {
  output <- list()
  for (i in seq_along(dflist)) {
    # print(head(dflist[[i]]))
    df_name <- paste0("Set_", i)
    limma_output <- limma_funct(dflist[[i]])
    limma_name <- paste0(df_name, "_limma")
    output[[limma_name]] <- limma_output
    # comp_output <- comparative_statistics(dflist[[i]])
    # comp_name <- paste0(df_name, "_comparative_stats")
    # output[[comp_name]] <- comp_output
  }
  return(output)
}
limma_subset <- function(df, mode = "default", n = 15, P = 0.05) {
  lim <- limma_funct(df)

  if (mode == "default") {
    lim_sigs <- row.names(lim[lim$adj.P.Val < P, ])
    lim_data <- cbind(df[1:2], df[lim_sigs])
  } else if (mode == "raw") {
    lim_sigs <- row.names(lim[lim$P.Val < P, ])
    lim_data <- cbind(df[1:2], df[lim_sigs])
  } else if (mode == "top") {
    lim_sigs <- lim %>%
      arrange(adj.P.Val) %>%
      head(n) %>%
      row.names()
    lim_data <- cbind(df[1:2], df[lim_sigs])
  }
  return(lim_data)
}

clustering_dflist_wrapper <- function(dflist, clust_function) {
  dflist_name <- deparse(substitute(dflist))
  plot_name <- deparse(substitute(clust_function))
  dir_path <- file.path(getwd(), "plots", dflist_name, plot_name)
  dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
  for (setid in seq_along(dflist)) {
    set_name <- paste("Set", setid, sep = "_")
    name <- (file.path(dir_path, set_name))
    clust_function(dflist[[setid]], name)
  }
}
## Transformers
# BoxCox
transformer_boxcox <- function(df, weighted = TRUE) {
  t_df <- df
  for (colname in colnames(df)[-1:-2]) {
    # print(colname)
    lambda <- determine_lambda(colname, df)
    if (weighted == TRUE) {
      t_df[[colname]] <- bc_weighted_transform(df[[colname]], lambda)
    } else {
      t_df[[colname]] <- bc_transform(df[[colname]], lambda)
    }
  }
  rm(dataf, column, envir = globalenv())
  return(t_df)
}
## Boxcox Modules
determine_lambda <- function(colname, df) {
  dataf <<- as.data.frame(df)
  column <<- df[, colname]
  # column <<- column
  model <- lm(column ~ group, data = dataf)
  bc <- boxcox(model, lambda = seq(-5, 5))
  lambda <- bc$x[which(bc$y == max(bc$y))]
  return(lambda)
}
bc_transform <- function(y, lambda = 0) {
  if (lambda == 0L) {
    log(y)
  } else {
    (y^lambda - 1) / lambda
  }
}
bc_weighted_transform <- function(y, lambda = 0) {
  geom <- geometric_mean(y)
  if (lambda == 0L) {
    log(y)
  } else {
    (y^lambda - 1) / lambda * geom^(lambda - 1)
  }
}
# RSN
transformer_rsn <- function(df) {
  subset <- as.matrix(df[-1:-2])
  sink(nullfile <- tempfile())
  rsn_transform <- lumiN(subset, method = "rsn")
  sink(NULL)
  na_cols <- which(colSums(is.na(rsn_transform)) > 0)
  if (length(na_cols) > 0) {
    cat("Columns with NAs:", colnames(rsn_transform)[na_cols], "\n")
    rsn_transform <- rsn_transform[, -na_cols]
  }
  return(cbind(df[1:2], rsn_transform))
}



## Automatic Processing
stage_3 <- function(dflist, name) {
  pdf(file = NULL)
  trans_list <- lapply(dflist, transformer_boxcox)
  trans_list_rsn <- lapply(trans_list, transformer_rsn)
  dev.off()
  bc_report_name <- paste0("reports_", name, "_bc")
  bcrsn_report_name <- paste0("reports_", name, "_bcrsn")
  assign(bc_report_name, differential_reports(trans_list), envir = .GlobalEnv)
  assign(bcrsn_report_name, differential_reports(trans_list_rsn), envir = .GlobalEnv)
  assign(name, trans_list_rsn, envir = .GlobalEnv)
}
