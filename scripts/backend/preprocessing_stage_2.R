# The function adds a prefix "Set_" to the index of each data frame to create the name for that data frame. 
# The function returns a list of these generated names.
simple_names <- function (dflist){
  namelist = list()
  for (i in 1:length(dflist)){
    namelist[i] = paste0("Set_", i)
  }
  return(namelist)
}

# The function mergedown takes a list of data frames dflist and merges them into a single data frame by 
# taking the row mean of columns with the same name in each data frame. It returns the merged data frame. 
# If there are columns in a data frame that are not present in any other data frame, they are included in the merged data frame as is.
mergedown <- function (dflist){
  outputdf = dflist[[1]]
  for (setid in 2:length(dflist)){
    originlist = colnames(dflist[[1]])[-1:-2]
    mergelist = colnames(dflist[[setid]])[-1:-2]
    uniquelist = mergelist[!(mergelist %in% originlist)]
    mergelist = mergelist[mergelist %in% originlist]}
  for (name in mergelist) {
    #print(name)
    outputdf[,name] = rowMeans(data.frame(dflist[[1]][,name], dflist[[setid]][,name]), na.rm = TRUE)
  }
  if (length(uniquelist) != 0){
    for (name in uniquelist) {
      #print(name)
      outputdf[name] = dflist[[setid]][,name]
    }
  }
  return(outputdf)
}

#Automatic Processing
stage_2 <- function(dflist, name){
  set3 <- mergedown(dflist)
  dflist <- c(dflist, list(set3))
  names(dflist) <- simple_names(dflist)
  assign(name, dflist, envir = .GlobalEnv)
}