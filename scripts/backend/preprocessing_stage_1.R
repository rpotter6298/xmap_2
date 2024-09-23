
detect_background_noise <- function(set, empty_id = NULL){
  # Get list of unique Internal.LIMS.ID values containing the word "empty"
  empty_ids <- unique(grep("empty", set$Internal.LIMS.ID, value = TRUE, ignore.case = TRUE))
  
  # If empty_id is not specified by the user, prompt the user to choose from the available options
  if (is.null(empty_id)){
    if (length(empty_ids) == 0){
      stop("No empty samples found in dataset")
    } else if (length(empty_ids) == 1){
      empty_id <- empty_ids
      message(paste0("Using empty ID: ", empty_id))
    } else {
      message("Multiple empty IDs found in dataset:")
      for (i in seq_along(empty_ids)){
        message(paste0(i, ": ", empty_ids[i]))
      }
      empty_id <- readline(prompt = "Enter the number corresponding to the desired empty ID: ")
      if (!as.numeric(empty_id) %in% seq_along(empty_ids)){
        stop("Invalid input. Aborting.")
      } else {
        empty_id <- empty_ids[as.numeric(empty_id)]
      }
    }
  }
  # Filter the data frame to include only the chosen empty ID
  emptyset <- set[set$Internal.LIMS.ID == empty_id, ]
  
  # Calculate the background noise
  inset = emptyset[-1:-2][colMeans(emptyset[-1:-2])<(median(colMeans(emptyset[-1:-2]))+(1*sd(colMeans(emptyset[-1:-2]))))]
  if (length(inset) < 0.95*length(emptyset)){
    calset = emptyset[-1:-2]
  }else{
    calset = inset
  }
  return(max(colMeans(calset))+sd(colMeans(calset)))
}

purge_background <- function(set, cutoff){
  bgmap = set[-1:-2]> cutoff
  above_cutoff = set[-1:-2][,colSums(bgmap)>0]
  below_cutoff = set[-1:-2][,colSums(bgmap)==0]
  keep_set = cbind(set[1:2],above_cutoff)
  remove_set = cbind(set[1:2],below_cutoff)
  return(list(keep_set, remove_set))
}

# The handle_background function takes a list of dataframes dflist,
# applies the purge_background function to each dataframe using detect_background_noise to determine the cutoff, 
# and replaces the original dataframe with the keep_set. 
# The remove_set is added to a new dflist in the global environment called X01_bg_purge. 
# The function returns the new dflist.
handle_background <- function(dflist) {
  new_dflist <- list()
  scrap_dflist <-list()
  for (i in seq_along(dflist)) {
    df <- dflist[[i]]
    cutoff <- detect_background_noise(df)
    keep_set <- purge_background(df, cutoff)[[1]]
    remove_set <- purge_background(df, cutoff)[[2]]
    new_dflist[[i]] <- keep_set
    scrap_dflist[[i]] <- remove_set
  }
  names(new_dflist) <- names(dflist)
  names(scrap_dflist) <- names(dflist)
  assign("X01_bg_purge", scrap_dflist, envir = .GlobalEnv)
  return(new_dflist)
}

# This function takes a dataframe "set" and adjusts it by subtracting the average value of an "EMPTY-0001" sample from all other samples. 
# It then sets any values less than 0 to 0, and adds 1 to all values to avoid breaking the log() function. The adjusted dataframe is returned.
emptyadjust <- function(set){
  emptyset = set[set$Internal.LIMS.ID=="EMPTY-0001",]
  emptyvector = colMeans(emptyset[-1:-2])
  set = set[set$Internal.LIMS.ID!="EMPTY-0001" & set$Internal.LIMS.ID!="MIX_2-0029",]
  labels = set[1:2]
  set = cbind(set[1:2],sweep(set[-1:-2],2,FUN="-",emptyvector))
  ### This sets anything with less read than the empty to 0, then adds one to everything, so that it doesn't break the log()
  set[-1:-2][set[-1:-2]<0] <- 0
  set[-1:-2] = set[-1:-2]+1
  return(set)
}

# The compress_duplicates function takes a dataframe "set" and an excel file "layout" as input
# and combines rows in the dataframe based on a shared identifier in the "layout" file. 
# It identifies matching rows, averages their values, and keeps the first row while removing the rest. 
# If any technical replicates have a deviation greater than either item, it prints notifications.
compress_duplicates <- function(set, layout){
  outlist = c()
  ### Reads the layout from excel file
  slayout <- read_excel(layout)
  ## Looks through the column named Tube Label for any items containing a hyphen, then uses any name preceding a hyphen as the for item
  for (n in strsplit(slayout$`Tube label`[grepl("-",slayout$`Tube label`)],"-")){
    # Filters the layout to only hold items either ending in or containing the for item immediately before the hyphen
    mergerows = ((slayout %>% filter(grepl(paste0(n[1],"$|",n[1],"-"),`Tube label`)))$`Sample id_LIMS`)
    # Returns the sample id for those samples, which is present in the main data set
    mergeset = set[grepl(paste(mergerows, collapse = "|"), set$Internal.LIMS.ID),]
    # Checks that both technical replicates are close enough together that their deviation is not greater than either item (like if one was 1000 and one was 10, this would print notifications)
    outlierflag = colSums(sweep(mergeset[-1:-2],2,apply(mergeset[-1:-2], 2, sd ), '-')<0)
    if (sum(outlierflag)>0){
      #print(mergerows)
      #print(outlierflag[outlierflag>0])
      #tupsum = c(mergerows, colnames(outlierflag[outlierflag>0]))
      #print(tupsum)
    }
    # Reassigns the first row in the set of matches so that it is equal to the mean of all matching sets, for each variable
    set[grepl(paste(mergerows, collapse = "|"), set$Internal.LIMS.ID),][1,][-1:-2] = colMeans(set[grepl(paste(mergerows, collapse = "|"), set$Internal.LIMS.ID),][-1:-2])
    # Eliminates all matching samples except the first (which is now the average of all matching) from the dataset
    for (i in 2:length(mergerows)){
      set = set[row.names(set) != row.names(set[grepl(paste(mergerows, collapse = "|"), set$Internal.LIMS.ID),][i,]),]
    }}
  return(set)
}

# This function takes a list of data frames dflist and a list of antigen names antigens. 
# It loops over each data frame in dflist and renames the column names of the data frames according to the antigen list. 
# If mode=1, the column names are set to the Antigen name 
# If mode=2, the column names are set to the Gene name. 
# The function then returns a list of data frames with updated column names.
set_colname_adapter <- function (dflist, antigens = I00_Antigens, mode=1, controls=get("controls", globalenv())){
  lapply(1:length(dflist), function(setid){
    set = dflist[[setid]]
    antigens = antigens [[setid]]
    for (i in 3:length(set)){
      analytenum = as.numeric(strsplit(colnames(set[i]), split='.', fixed = TRUE)[[1]][2])
      if (mode == 1){
        colnames(set)[i] = antigens[antigens$analyte == analytenum,]$Antigen.name
      } else if (mode == 2){
        colnames(set)[i] = antigens[antigens$analyte == analytenum,]$Gene.name
      }}
    set = set[,!(names(set) %in% controls)]
    return(set)
  })
} 


### Automatic Processing
stage_1 <- function(dflist, name){
  T01 = handle_background(dflist)
  T02 = lapply(T01,emptyadjust)
  T03 = lapply(T02,compress_duplicates, layout="data/layout.xlsx")
  T04 = set_colname_adapter(T03)
  assign(name, T04, envir = .GlobalEnv)
}
