dataset <- "GLA02"
controls <- c("Anti-human IgG", "EBNA1", "Bare-bead", "His6ABP")
for (file in list.files(file.path("scripts", "backend"))) {
  source(file.path("scripts", "backend", file))
}


### Pipeline
library("readxl")
library("dplyr")
import(dataset)
stage_1(I01_Import, "P01_Preprocessed")
stage_2(P01_Preprocessed, "P02_Merged")

full_dataset <- P02_Merged$Set_3
healthy_dataset <- full_dataset[full_dataset$group == 0, ]
gc_dataset <- full_dataset[full_dataset$group == 1, ]


library(reshape2)
library(ggplot2)

gc_averages <- colMeans(gc_dataset[-1:-2])
hc_averages <- colMeans(healthy_dataset[-1:-2])
adjusted_gc_averages <- gc_averages - hc_averages
gc_averages <- gc_averages[order(gc_averages, decreasing = TRUE)]
adjusted_gc_averages <- adjusted_gc_averages[order(adjusted_gc_averages, decreasing = TRUE)]

top_20 <- names(gc_averages)[1:20]
adjused_top_20 <- names(adjusted_gc_averages)[1:20]
names_list <- replace_antigen_names(top_20)
adjusted_names_list <- replace_antigen_names(adjused_top_20)

top_60 <- read.csv("top_60.list", header = FALSE)
#check how many of the top 20 expressed genes are in the top 60 expressed genes
sum(names_list %in% top_60$V1)

#check how many of the top 20 expressed genes are in the adjusted top 20 expressed genes
sum(top_20 %in% adjused_top_20)
###Six of top_20 are in the top 60
#Identify which names are in both the top 20 and the top 60
names_list[names_list %in% top_60$V1]

top_20_expressed_gc <- gc_dataset[, c("Internal.LIMS.ID", "group", top_20)]
colnames(top_20_expressed_gc)[-1:-2] <- names_list

#change to long format
top_20_expressed_gc_melted <- melt(top_20_expressed_gc, id.vars = c("Internal.LIMS.ID", "group"), variable.name = "variable", value.name = "value")
#create a scatterplot of the top 20 expressed genes
df = top_20_expressed_gc_melted
# Create scatterplot
ggplot(df, aes(x = variable, y = value, color = Internal.LIMS.ID)) +
  geom_point() + # This adds the scatterplot points
  theme_minimal() + # Optional: This applies a minimal theme to the plot
  labs(title = "Variable Values by Internal LIMS ID", 
       x = "Variable", 
       y = "Value") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotate x-axis labels for better readability


%%R
significant_antigens <- c("HPRA034083", "HPRA000767", "HPRA019035", "HPRA006876", "HPRA003490", "HPRA022019", "HPRA017192")
head(gc_dataset)

# Function to calculate correlations with significant antigens
calculate_correlations <- function(gc_dataset, significant_antigens) {
  # Filter out non-antigen columns (keep only antigen data)
  antigen_data <- gc_dataset[ , !(names(gc_dataset) %in% c("Internal.LIMS.ID", "group"))]
  
  # Initialize a list to store correlation results
  correlation_results <- list()
  
  # Iterate over each significant antigen
  for(significant_antigen in significant_antigens) {
    # Check if the significant antigen exists in the dataset
    if(significant_antigen %in% names(antigen_data)) {
      # Calculate correlation of this antigen with all others
      correlations <- cor(antigen_data[, significant_antigen, drop = FALSE], antigen_data, use = "complete.obs", method = "pearson")
      # Store the correlations in the list with the antigen name as the key
      correlation_results[[significant_antigen]] <- correlations
    } else {
      cat(paste("Warning: Antigen", significant_antigen, "not found in dataset.\n"))
    }
  }
  
  return(correlation_results)
}

# Example usage:
full_dataset <- P03_Transformed$Set_3
healthy_dataset <- full_dataset[full_dataset$group == 0, ]
gc_dataset <- full_dataset[full_dataset$group == 1, ]
significant_antigens <- c("HPRA034083", "HPRA000767", "HPRA019035", "HPRA006876", "HPRA003490", "HPRA022019", "HPRA017192")
#correlation_results <- calculate_correlations(gc_dataset, significant_antigens)

# Revised function to ensure names are preserved in the strong correlations vector
calculate_and_filter_correlations_fixed <- function(gc_dataset, significant_antigens, correlation_threshold = 0.7) {
  # Filter out non-antigen columns (keep only antigen data)
  antigen_data <- gc_dataset[ , !(names(gc_dataset) %in% c("Internal.LIMS.ID", "group"))]
  
  # Initialize a list to store strong correlation results
  strong_correlation_results_fixed <- list()
  
  # Calculate the full correlation matrix once
  full_correlations <- cor(antigen_data, use = "complete.obs", method = "pearson")
  
  # Iterate over each significant antigen
  for(significant_antigen in significant_antigens) {
    if(significant_antigen %in% names(antigen_data)) {
      # Extract correlations for the current significant antigen
      current_correlations <- full_correlations[, significant_antigen]
      
      # Ensure names are preserved
      names(current_correlations) <- rownames(full_correlations)
      
      # Filter for strong correlations (absolute value greater than the threshold)
      strong_correlations <- current_correlations[abs(current_correlations) > correlation_threshold]
      
      # Remove the correlation of the antigen with itself, ensuring names are used for filtering
      strong_correlations <- strong_correlations[names(strong_correlations) != significant_antigen]
      
      # Store the strong correlations in the list with the antigen name as the key
      strong_correlation_results_fixed[[significant_antigen]] <- strong_correlations
    } else {
      cat(paste("Warning: Antigen", significant_antigen, "not found in dataset.\n"))
    }
  }
  
  return(strong_correlation_results_fixed)
}

# Example usage:
significant_antigens <- c("HPRA034083", "HPRA000767", "HPRA019035", "HPRA006876", "HPRA003490", "HPRA022019", "HPRA017192")
strong_correlation_results_fixed <- calculate_and_filter_correlations_fixed(gc_dataset, significant_antigens, 0.75)

# Initialize an empty vector to store all unique antigen names
all_strongly_correlated_antigens <- c()

# Iterate over the list of strong correlations for each significant antigen
for(significant_antigen in names(strong_correlation_results_fixed)) {
  # Extract names of strongly correlated antigens for the current significant antigen
  current_antigens <- names(strong_correlation_results_fixed[[significant_antigen]])
  
  # Combine with the existing list of antigen names
  all_strongly_correlated_antigens <- c(all_strongly_correlated_antigens, current_antigens)
}

# Remove duplicates to get a unique list of antigen names
all_strongly_correlated_antigens <- unique(all_strongly_correlated_antigens)

# all_strongly_correlated_antigens now contains all unique names of strongly correlated antigens
print(all_strongly_correlated_antigens)
length(all_strongly_correlated_antigens)



# Initialize an empty data frame to store antigens and their highest correlation coefficients
antigens_correlations <- data.frame(antigen = character(), correlation = numeric(), stringsAsFactors = FALSE)

# Iterate over the list of strong correlations for each significant antigen
for(significant_antigen in names(strong_correlation_results_fixed)) {
  # Extract the correlations for the current significant antigen
  correlations <- strong_correlation_results_fixed[[significant_antigen]]
  
  # Create a data frame for the current significant antigen's correlations
  current_df <- data.frame(antigen = names(correlations), correlation = as.numeric(correlations), stringsAsFactors = FALSE)
  
  # Combine with the existing data frame
  antigens_correlations <- rbind(antigens_correlations, current_df)
}

# Remove duplicates based on the antigen name, keeping the entry with the highest absolute correlation
antigens_correlations <- antigens_correlations[order(-abs(antigens_correlations$correlation)),]
antigens_correlations <- antigens_correlations[!duplicated(antigens_correlations$antigen),]

# Get the top 50 unique antigens based on their highest absolute correlation
top_50_antigens <- head(antigens_correlations, 50)

# top_50_antigens now contains the top 50 antigens and their correlations
print(top_50_antigens)

# Initialize an empty data frame to store antigens, their correlations, and the significant antigen(s) they correlate with
antigens_correlations_partners <- data.frame(antigen = character(), correlation = numeric(), partner = character(), stringsAsFactors = FALSE)

# Iterate over the list of strong correlations for each significant antigen
for(significant_antigen in names(strong_correlation_results_fixed)) {
  # Extract the correlations for the current significant antigen
  correlations <- strong_correlation_results_fixed[[significant_antigen]]
  
  # Create a data frame for the current significant antigen's correlations
  current_df <- data.frame(antigen = names(correlations), correlation = as.numeric(correlations), partner = significant_antigen, stringsAsFactors = FALSE)
  
  # Combine with the existing data frame
  antigens_correlations_partners <- rbind(antigens_correlations_partners, current_df)
}

# Order by absolute value of correlation to prioritize stronger correlations
antigens_correlations_partners <- antigens_correlations_partners[order(-abs(antigens_correlations_partners$correlation)),]

# For antigens that appear multiple times, consolidate their partner entries
library(dplyr)
antigens_correlations_partners <- antigens_correlations_partners %>%
  group_by(antigen) %>%
  summarise(correlation = first(correlation), partner = paste(unique(partner), collapse = ", ")) %>%
  ungroup()

# Now, remove duplicate antigens while keeping the highest correlation (already done by our summarise since we're taking the first() which is the highest due to our ordering)
# Get the top 50 unique antigens
top_50_antigens_partners <- head(antigens_correlations_partners, 50)

# top_50_antigens_partners now contains the top 50 antigens, their correlations, and the significant antigen(s) they correlate with
print(top_50_antigens_partners)

