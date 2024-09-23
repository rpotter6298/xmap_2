#fill out useful info
fill_analyte_info <- function(df, antigens=I00_Antigens){
  lookup_df <- bind_rows(antigens)[-1]
  # Remove duplicates from lookup_df based on Antigen.name
  lookup_df <- lookup_df %>%
    distinct(Antigen.name, .keep_all = TRUE)
  # Create a new column with the row names in main_df
  main_df <- df %>%
    rownames_to_column(var = "RowName")
  # Check if most RowName values are not found in Antigen.name
  rowname_matches <- main_df$RowName %in% lookup_df$Antigen.name
  proportion_matches <- sum(rowname_matches) / length(main_df$RowName)
  
  if (proportion_matches < 0.5) {
    warning("Most RowName values are not found in Antigen.name")
  } else {
    message("Most RowName values are found in Antigen.name")
  }
  # Merge the main dataframe with the lookup dataframe based on the Antigen.name column
  result_df <- main_df %>%
    left_join(lookup_df, by = c("RowName" = "Antigen.name"))
  return(result_df)
}

# Function to subset the dataframe based on the lowest (n) values
subset_top_n <- function(df, n) {
  if ("Q.Value" %in% colnames(df)) {
    subset_df <- df %>% top_n(-n, Q.Value)
  } else if ("adj.P.Val" %in% colnames(df)) {
    subset_df <- df %>% top_n(-n, adj.P.Val)
  } else {
    return(df)
  }
  #subset_df <- rownames_to_column(subset_df, var = "RowName")
  return(subset_df)
}

excel_subset_export <- function(dflist, n=15){
  # Apply the function to the dflist
  subsetted_dflist <- lapply(dflist, subset_top_n, n)
  subsetted_dflist <- lapply(subsetted_dflist,fill_analyte_info)
  # Get the name of the dflist
  dflist_name <- deparse(substitute(dflist))
  # Write the list of subsetted dataframes to an Excel file
  excel_file_name <- paste0(dflist_name, "_Top_", n, ".xlsx")
  # Create a new workbook
  wb <- createWorkbook()
  # Add worksheets for each subsetted dataframe and write data
  for (i in seq_along(subsetted_dflist)) {
    sheet_name <- names(dflist)[i]
    addWorksheet(wb, sheet_name)
    writeData(wb, sheet_name, subsetted_dflist[[i]])
  }
  
  # Save the workbook
  saveWorkbook(wb, file = file.path("stats", excel_file_name), overwrite = TRUE)
}

#Final Modifications
full_analyte_info <- function(df, antigens=I00_Antigens){
  lookup_df <- bind_rows(antigens)[-1]
  # Remove duplicates from lookup_df based on Antigen.name
  lookup_df <- lookup_df %>%
    distinct(Antigen.name, .keep_all = TRUE)
  # Create a new column with the row names in main_df
  main_df <- df %>%
    rownames_to_column(var = "RowName")
  # Check if most RowName values are not found in Antigen.name
  rowname_matches <- main_df$RowName %in% lookup_df$Antigen.name
  proportion_matches <- sum(rowname_matches) / length(main_df$RowName)
  
  if (proportion_matches < 0.5) {
    warning("Most RowName values are not found in Antigen.name")
  }
  # else {
  #   message("Most RowName values are found in Antigen.name")
  # }
  # Merge the main dataframe with the lookup dataframe based on the Antigen.name column
  result_df <- main_df %>%
    left_join(lookup_df, by = c("RowName" = "Antigen.name"))
  return(result_df)
}

excel_export <- function(dflist, name = deparse(substitute(dflist))) {
  file_path <- file.path(getwd(), "stats", paste0(name, ".xlsx"))
  
  # Create a new workbook
  wb <- createWorkbook()
  
  # Iterate through the list of dataframes and add worksheets
  for (setid in seq_along(dflist)) {
    sheet_name <- paste0("Dataset_", setid)
    addWorksheet(wb, sheet_name)
    writeData(wb, sheet_name, dflist[[setid]])
  }
  
  # Save the workbook
  saveWorkbook(wb, file = file_path, overwrite = TRUE)
}

replace_antigen_names<- function(vector, antigens=I00_Antigens, replacement_col = "Gene.name"){
  lookup_df <- bind_rows(antigens)[-1]
  # Remove duplicates from lookup_df based on Antigen.name
  lookup_df <- lookup_df %>%
    distinct(Antigen.name, .keep_all = TRUE)
  
  # Create a lookup dictionary with Antigen.name as the key and the specified column as the value
  lookup_dict <- setNames(lookup_df[[replacement_col]], lookup_df$Antigen.name)
  replaced_vector <- lookup_dict[vector]
  return(replaced_vector)
}

untransform_subset <- function(restriction_df, df, subset_method = limma_subset, ...){
  subset <- do.call(subset_method, c(list(restriction_df), list(...)))
  df_subset <- df[,colnames(df) %in% colnames(subset)]
  colnames(df_subset)[-1:-2]=replace_antigen_names(colnames(df_subset[-1:-2]))
  return(df_subset)
}