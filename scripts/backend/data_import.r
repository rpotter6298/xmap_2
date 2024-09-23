#Takes a list of package names and automates the process of checking, installing, and loading them. 
#Handles both CRAN and Bioconductor packages, ensuring that the necessary packages are available in the user's R environment.
load_dependencies <- function(pkg_list) {
  # Load or install packages from list
  for (pkg in pkg_list) {
    if (substr(pkg, 1, 11) == "BiocManager") {
      pkg = substr(pkg, 14, nchar(pkg))
      if (!requireNamespace("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager")
      }
      if (!require(pkg, character.only = TRUE)) {
        BiocManager::install(pkg)
      }
    }
    else{
      if (!require(pkg, character.only = TRUE)) {
        install.packages(pkg)
      }
    }
    library(pkg, character.only = TRUE)
  }
}
# This function checks the integrity of a set of dataset files in a specified directory, filtering them by keyword, 
# and ensures that each Data Intensity file has a corresponding Antigen List file. 
# If any files are found to be missing an Antigen List, the function returns a list of these files, otherwise it returns TRUE.
  dataset_file_integrity_check <- function(keyword) {
    # Get the list of files in the "sample_data" subdirectory
    file_list <- list.files("data", pattern = "\\.xlsx", full.names = TRUE)
    
    # Filter the file list to include only files containing the specified keyword
    keyword_files <- grep(keyword, file_list, value = TRUE, ignore.case = TRUE)
    
    # Replace any space or hyphen in each of the names in keyword files with an underscore
    keyword_files <- gsub(" |-", "_", keyword_files)
    
    # Remove characters in the file names up to one after the dataset keyword
    keyword_files <- gsub(paste0(".*", keyword, "[ _-](.*)\\.xlsx$"), "\\1.xlsx", keyword_files)
    
    # Find the non-dictionary tags present in the names of several files
    non_dict_tags <- unique(gsub("(_Antigen_list.xlsx|_Data_Intensity.xlsx)", "", keyword_files))
    
    # Identify Data Intensity files that don't have a corresponding Antigen List file
    missing_antigen_list_files <- character()
    for (tag in non_dict_tags) {
      antigen_file <- paste0(tag, "_Antigen_list.xlsx")
      intensity_file <- paste0(tag, "_Data_Intensity.xlsx")
      if (intensity_file %in% keyword_files && !(antigen_file %in% keyword_files)) {
        missing_antigen_list_files <- c(missing_antigen_list_files, intensity_file)
      }
    }
    
    # Check if the list of data files with no antigen file is empty
    if (length(missing_antigen_list_files) == 0) {
      return(TRUE)
    } else {
      #print(paste("Some files are missing antigen lists:", missing_antigen_list_files))
      return(missing_antigen_list_files)
    }
  }
# Function to filter out data frames with a given keyword in their name
# keyword: the keyword to search for in the data frame names
# e: the environment to search in (default is parent frame)
  filterclean <- function(keyword, e = parent.frame()) {
  
  # Get list of data frames in the specified environment
  dflist = Filter(function(x) is (x, "data.frame"),
                  mget(ls(e),envir= e))
  # print(dflist)
  # Filter out data frames without the specified keyword
  dflist = dflist[grepl(keyword,ls(dflist))]
  # print(dflist)
  # Return filtered list of data frames
  return (dflist)
}
## This import function first brings all excel documents in the sample_data directory which include the dataset string and imports them as dataframes
#  The dataframes are then grouped according to _Data and _Antigen keywords in the file naming convention (filterclean function)
  import <- function(dataset){
    ## Import all libraries needed for downstream
    # pkg_list = c("rlang",
    #              "gridExtra", 
    #              "ggplot2", 
    #              "ggfortify", 
    #              "MASS", 
    #              "BiocManager::lumi",
    #              "BiocManager::limma", 
    #              "readxl", 
    #              "dplyr",
    #              "broom",
    #              "BiocManager::qvalue",
    #              "openxlsx",
    #              "tibble",
    #              "pheatmap",
    #              "pROC",
    #              "tidyverse",
    #              "msigdbr",
    #              "BiocManager::clusterProfiler")
    # load_dependencies(pkg_list)
    # set path to sample data directory
    sample_data=paste(getwd(),"/data/", sep="")
    # get names of all files with .xlsx extension in sample_data directory
    names = list.files(path=sample_data, pattern = ".xlsx", recursive=TRUE)
    
    # check if any required files are missing
    no_missing_files <- dataset_file_integrity_check(dataset)
    
    # if files are missing
    if (no_missing_files != TRUE) {
      # if files are missing
      # ask user if they want to halt the function or continue without missing files
      stop_message <- paste0("Some files are missing antigen lists: ", no_missing_files, "\n")
      cat(stop_message)
      choice <- readline("Do you want to halt the function? (y/n) ")
      # if user chooses to halt the function, stop and print error message
      if (choice == "y") {
        stop(stop_message)
      }}
    # read all files in sample_data directory and assign to variables with shortened names
    for (file in names){
      shortindex = gregexpr(pattern=dataset,file)[[1]][1]
      shortname = substring(file,shortindex+6)
      shortname = gsub(" |-", "_", shortname)
      assign(paste0(shortname), read_xlsx(paste(sample_data,file,sep="")))
    }
    # assign cleaned data and antigen data to variables with dataset name and appropriate suffixes
    assign(paste(dataset, "_A00", sep=""),filterclean("_Data"))
    assign(paste(dataset, "_Antigens", sep=""), filterclean("_Antigen"))
    # clean and format data
    A01_Input = lapply(get(paste(dataset, "_A00", sep="")),function(df){
      df = data.frame(df)
      names(df)[2] = "group"
      # assign group value of 1 if it contains "GC", 0 if it contains "HD", and "NA" if neither
      df$group =ifelse(grepl("GC",df$group), 1,
                       ifelse(grepl("HD",df$group), 0, "NA"))
      df
    })
    # format antigen data
    AA_Antigens = lapply(get(paste(dataset, "_Antigens", sep="")),function(df){
      df = data.frame(df)
      names(df)[1] = "analyte"
      df
    })
    # assign cleaned and formatted data to global environment variables
    assign("I01_Import", A01_Input, env=globalenv())
    assign("I00_Antigens", AA_Antigens, env=globalenv())
  }
#   
  

