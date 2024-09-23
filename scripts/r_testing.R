library("readxl")
library("dplyr")
library("MASS")
library("lumi")
library("limma")
library("broom")
library("qvalue")
dataset <- "GLA02"
controls <- c("Anti-human IgG", "EBNA1", "Bare-bead", "His6ABP")
for (file in list.files(file.path("scripts", "backend"))) {
    source(file.path("scripts", "backend", file))
}


### Pipeline
import(dataset)
stage_1(I01_Import, "P01_Preprocessed")
stage_2(P01_Preprocessed, "P02_Merged")
stage_3(P02_Merged, "P03_Transformed")


gc_corr_dataset <- P03_Transformed$Set_3
gc_corr_dataset <- gc_corr_dataset[gc_corr_dataset$group == 1, ]
significant_antigens <- c("HPRA034083", "HPRA000767", "HPRA019035", "HPRA006876", "HPRA003490", "HPRA022019", "HPRA017192")

correlation_threshold <- 0.75
# Revised function to ensure names are preserved in the strong correlations vector
calculate_and_filter_correlations <- function(gc_dataset, significant_antigens, corr_threshold = correlation_threshold) {
    # Filter out non-antigen columns (keep only antigen data)
    antigen_data <- gc_dataset[, !(names(gc_dataset) %in% c("Internal.LIMS.ID", "group"))]

    # Initialize a list to store strong correlation results
    strong_correlation_results_fixed <- list()

    # Calculate the full correlation matrix once
    full_correlations <- cor(antigen_data, use = "complete.obs", method = "pearson")

    # Iterate over each significant antigen
    for (significant_antigen in significant_antigens) {
        if (significant_antigen %in% names(antigen_data)) {
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
sink()  # Reset sink
dev.off()  # Close all graphics devices

correlation_dataframe <- calculate_and_filter_correlations(gc_corr_dataset, significant_antigens)
cd2 <- calculate_and_filter_correlations(gc_corr_dataset, significant_antigens)
# Initialize an empty vector to store all unique antigen names
all_strongly_correlated_antigens <- c()
# Iterate over the list of strong correlations for each significant antigen
for (significant_antigen in names(correlation_dataframe)) {
    # Extract names of strongly correlated antigens for the current significant antigen
    current_antigens <- names(correlation_dataframe[[significant_antigen]])

    # Combine with the existing list of antigen names
    all_strongly_correlated_antigens <- c(all_strongly_correlated_antigens, current_antigens)
}
# Remove duplicates to get a unique list of antigen names
all_strongly_correlated_antigens <- unique(all_strongly_correlated_antigens)
# now replace the names with the corresponding genes and save to a list for export to python
correlated_genes <- (replace_antigen_names(all_strongly_correlated_antigens))

library(dplyr)

pairs_list <- list()

for (antigen in names(correlation_dataframe)){
  print(antigen)
}
str(correlation_dataframe)
exists("correlation_dataframe")

# Combine all data frames into one
pairs_df <- bind_rows(pairs_list)

# Now let's sample 3 random rows (pairs)
set.seed(123)  # Setting a seed for reproducibility
random_pairs <- pairs_df[sample(nrow(pairs_df), 3), ]

# Assuming gc_corr_dataset is a data frame with columns for each antigen
# Generate scatter plots for the 3 random pairs
library(ggplot2)
for (i in 1:nrow(random_pairs)) {
  pair <- random_pairs[i, ]
  plot_data <- data.frame(Antigen1 = gc_corr_dataset[[pair$antigen1]],
                          Antigen2 = gc_corr_dataset[[pair$antigen2]])
  
  print(ggplot(plot_data, aes(x = Antigen1, y = Antigen2)) +
          geom_point(alpha = 0.6) +
          geom_smooth(method = "lm", se = FALSE, color = "blue") +
          labs(title = paste("Scatter Plot with Regression Line:", pair$antigen1, "vs.", pair$antigen2),
               x = pair$antigen1,
               y = pair$antigen2) +
          theme_minimal())
}

