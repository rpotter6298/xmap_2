from classes import *
import pandas as pd

# Step 1: Initialize the study
#study = xmap_study(study_report_xls="data/02_assay_data.xlsx", control_cols=slice(-4, None))
study2 = xmap_study(study_report_xls="data/02_assay_data.xlsx", control_cols=slice(-4, None))
study2.patient_data = study.patient_data
study2.patient_data_coldic = study.patient_data_coldic
study = study2
# Step 2: Attach patient data
patient_data = pd.read_excel("data/02_patient_data_cleaned.xlsx", index_col=0)
study.attach_patient_data(patient_data)

# Step 3: Normalize the data
normalizer = Normalizer()
normalized_data = normalizer.normalize(
    study.adj_mfi_data.drop(columns=["Patient_ID"]),
    rsn_transform=True,
    boxcox_transform=True,
)
study.normalized_data = normalized_data

# Step 4: Perform Quality Control (Variance Filtering)
qc = xmap_qc(study.normalized_data)
cleaned_proteins = qc.filter_low_variance_proteins(threshold=1e-5)
qc.plot_variance(top_n=20)
# Step 5: Perform Correlation Analysis
correlation_analyzer = CorrelationAnalyzer(cleaned_proteins, study.patient_data)

# Define column classifications based on uniqueness and data types
binary_cols = [col for col in study.patient_data.columns if study.patient_data[col].nunique() == 2]
continuous_cols = [col for col in study.patient_data.columns if study.patient_data[col].dtype in ['float64', 'int64'] and study.patient_data[col].nunique() > 2]
multi_level_cols = [col for col in study.patient_data.columns if study.patient_data[col].nunique() > 2 and col not in continuous_cols]
##Manually remove medicine list and active ingredients
multi_level_cols.remove("Medicine_List")
multi_level_cols.remove("Active_Ingredients")

# Compute different correlations
point_biserial_corr = correlation_analyzer.point_biserial_correlation(binary_cols)
kendall_tau_corr = correlation_analyzer.kendall_tau_correlation(continuous_cols)
pearson_corr = correlation_analyzer.pearson_correlation(continuous_cols)
spearman_corr = correlation_analyzer.spearman_correlation(continuous_cols)
anova_f_stat = correlation_analyzer.anova_f_statistic(multi_level_cols)

# Compute Protein-Protein Correlations
protein_protein_corr = correlation_analyzer.protein_protein_correlation(method='pearson')  # Change method as needed

# Step 6: Visualize Correlations
visualizer = Visualizer()
visualizer.plot_correlation_heatmap(point_biserial_corr, title="Point-Biserial Correlation Heatmap", cbar_label="Correlation Coefficient")
visualizer.plot_correlation_heatmap(kendall_tau_corr, title="Kendall's Tau Correlation Heatmap", cbar_label="Correlation Coefficient")
visualizer.plot_correlation_heatmap(pearson_corr, title="Pearson Correlation Heatmap", cbar_label="Correlation Coefficient")
visualizer.plot_correlation_heatmap(spearman_corr, title="Spearman Correlation Heatmap", cbar_label="Correlation Coefficient")
visualizer.plot_correlation_heatmap(anova_f_stat, title="ANOVA F-Statistic Heatmap", cmap="YlGnBu", cbar_label="F-Statistic")
visualizer.plot_correlation_heatmap(protein_protein_corr, title="Protein-Protein Pearson Correlation Heatmap", cmap="coolwarm", cbar_label="Correlation Coefficient")

# Step 7: Aggregate Data Analysis (Loop through binary columns)
# Initialize an empty DataFrame to store all t-test results
mapping_dict = {"Sex" : {"Male":1, "Female":0}}
t_test_results = study.binary_t_test(binary_cols, cleaned_proteins, mapping_dict=mapping_dict)


visualizer.plot_volcano_3d_with_labels(t_test_results, binary_cols, p_value_threshold=0.05)
t_test_report = visualizer.generate_significant_report(t_test_results, binary_cols, p_value_threshold=0.05)
print("")
# # Step 7: Aggregate Data Analysis (e.g., Compare Means Between Groups)
# group_comparator = GroupComparator(cleaned_proteins, study.patient_data['Sex'])
# # Perform t-tests between 'Male' and 'Female'
# t_test_results = group_comparator.compare_aggregate_means(method='t_test', group1=0, group2=1)
# print("T-Test Results between Male and Female:")
# print(t_test_results)



# main_study.patient_data.columns
# column_class = []
# data = main_study.patient_data
# for column in main_study.patient_data.columns:
#     column_class.append(column_classification(main_study.patient_data[column]))
# columns_with_class = pd.DataFrame(
#     {"Column": main_study.patient_data.columns, "Class": column_class})
# determinations = []
# for col in columns_with_class["Column"]:
#     if columns_with_class[columns_with_class["Column"] == col]["Class"].values[0] == "undetermined":
#         determinations.append(evaluate_column(main_study.patient_data[col]).choices[0].message.content.strip())
#     else:
#         determinations.append(columns_with_class[columns_with_class["Column"] == col]["Class"].values[0])
# col_with_class_full = pd.DataFrame(
#     {"Column": main_study.patient_data.columns, "Class": column_class, "Determination": determinations})
# response.choices[0].message.content.strip()
# column = main_study.patient_data.iloc[:, 3]
# unique_values = column.unique()
# print(unique_values)
# # select just the first 20 columns of the normalized data
# main_study.normalized_data["IL6R_HPRR2760260"].var()

# # select only numeric columns from the patient data
# patient_subset = main_study.patient_data.select_dtypes(include="number")
# protein_corr_matrix = prot_subset.corr(method="pearson")
# patient_metrics_corr = prot_subset.corrwith(patient_subset, method="pearson")

# # Step 1: Calculate Variance for Each Protein
# protein_variances = main_study.normalized_data.var()

# # Step 2: Define a Variance Threshold
# # Option 1: Absolute Threshold
# # Example: Remove proteins with variance less than 1e-5
# absolute_threshold = 1e-5
# median_variance = protein_variances.median()
# relative_threshold = median_variance * 0.01  # Adjust the multiplier as needed
# threshold = min(absolute_threshold, relative_threshold)
# low_variance_proteins = protein_variances[protein_variances < threshold].index.tolist()
# cleaned_data = main_study.normalized_data.drop(columns=low_variance_proteins)
# protein_subset = cleaned_data.iloc[:, :20]
# protein_corr_matrix = protein_subset.corr(method="pearson")
# patient_metrics_corr = protein_subset.corrwith(patient_subset, method="pearson")
# # Check if indices are aligned
# print(protein_subset.index.equals(patient_subset.index))

# # If not, align the indices
# protein_subset = protein_subset.loc[patient_subset.index]

# # Select only numeric columns from both datasets
# protein_subset_numeric = protein_subset.select_dtypes(include=[np.number])
# patient_subset_numeric = patient_subset.select_dtypes(include=[np.number])

# # Perform correlation again
# patient_metrics_corr = protein_subset_numeric.corrwith(
#     patient_subset_numeric, method="pearson"
# )
# print(patient_metrics_corr)

# # Option 1: Drop rows or columns with missing values
# protein_subset_clean = protein_subset_numeric.dropna()
# patient_subset_clean = patient_subset_numeric.dropna()

# # Option 2: Fill missing values with a specific value (e.g., mean or median)
# protein_subset_filled = protein_subset_numeric.fillna(protein_subset_numeric.mean())
# patient_subset_filled = patient_subset_numeric.fillna(patient_subset_numeric.mean())

# # Perform correlation after cleaning the data
# patient_metrics_corr = protein_subset_clean.T.corrwith(
#     patient_subset_clean, method="pearson"
# )
# print(patient_metrics_corr)
# # Check the shape of your DataFrames
# print(protein_subset_numeric.shape)
# print(patient_subset_numeric.shape)

# # Ensure proper alignment
# print(protein_subset_numeric.index)
# print(patient_subset_numeric.index)
# print(patient_subset_numeric.info())
# print(protein_subset_numeric.info())
# # Drop the 'Patient_ID' column from the patient subset
# patient_subset_clean = patient_subset_numeric.drop(columns=["Patient_ID"])

# # Recompute correlations
# patient_metrics_corr = protein_subset_numeric.corrwith(
#     patient_subset_clean,
#     method="pearson",
# )
# print(patient_metrics_corr)

# from sklearn.preprocessing import StandardScaler

# # Standardize both datasets
# scaler = StandardScaler()

# protein_subset_scaled = pd.DataFrame(
#     scaler.fit_transform(protein_subset_numeric),
#     index=protein_subset_numeric.index,
#     columns=protein_subset_numeric.columns,
# )

# patient_subset_scaled = pd.DataFrame(
#     scaler.fit_transform(patient_subset_clean),
#     index=patient_subset_clean.index,
#     columns=patient_subset_clean.columns,
# )

# # Recompute correlations
# patient_metrics_corr = protein_subset_scaled.corrwith(patient_subset_scaled, axis=0)
# print(patient_metrics_corr)

# # Check for columns with near-zero variance
# low_variance_proteins = protein_subset_numeric.var()[
#     protein_subset_numeric.var() < 1e-10
# ]
# low_variance_patients = patient_subset_clean.var()[patient_subset_clean.var() < 1e-10]

# print(f"Proteins with near-zero variance:\n{low_variance_proteins}")
# print(f"Patient metrics with near-zero variance:\n{low_variance_patients}")

# # Ensure column-wise correlation
# patient_metrics_corr = protein_subset_numeric.corrwith(patient_subset_clean, axis=0)
# print(patient_metrics_corr)

# result = pd.concat([protein_subset_numeric, patient_subset_clean], axis=1).corr()

# import pandas as pd

# # Concatenate protein and patient data along the columns
# combined_df = pd.concat([protein_subset_numeric, patient_subset_clean], axis=1)

# # Calculate the full correlation matrix
# full_corr_matrix = combined_df.corr()

# # Extract relevant correlations: between protein columns and patient metric columns
# # We want correlations where proteins (from protein_subset_numeric) are correlated with metrics (from patient_subset_clean)
# protein_columns = protein_subset_numeric.columns
# patient_columns = patient_subset_clean.columns

# # Extract the relevant part of the correlation matrix
# protein_patient_corr = full_corr_matrix.loc[protein_columns, patient_columns]

# # Display the correlation results
# print(protein_patient_corr)

# import seaborn as sns
# import matplotlib.pyplot as plt

# # Set the plot size
# plt.figure(figsize=(12, 8))

# # Create the heatmap
# sns.heatmap(
#     protein_patient_corr,
#     annot=True,
#     fmt=".2f",
#     cmap="coolwarm",
#     cbar_kws={"label": "Correlation"},
# )

# # Set the labels and title
# plt.title("Protein-Patient Correlation Heatmap", fontsize=16)
# plt.xlabel("Patient Metrics", fontsize=12)
# plt.ylabel("Proteins", fontsize=12)

# # Show the plot
# plt.tight_layout()
# plt.show()


# import pandas as pd
# import numpy as np
# import plotly.graph_objs as go
# import plotly.express as px

# # Assuming 'raw_proteins_df' is your raw protein data before normalization
# zero_var_before = main_study.adj_mfi_data.columns[
#     main_study.adj_mfi_data.var() == 0
# ].tolist()
# zero_var_columns = main_study.normalized_data.columns[
#     main_study.normalized_data.var() == 0
# ].tolist()
# print(f"Proteins with zero variance before normalization: {zero_var_before}")


# commander = regression_controller(
#     data = pd.concat([main_study.adj_mfi_data.iloc[:,:-1], main_study.patient_data.iloc[:,:-1]], axis=1,),
#     split_parameters = {"split_type": "kfold", "n_splits": 5},
#     predictors = [],
#     confounders = [],
#     outcome = "GPA"
#     )
# commander.initialize_split()
# commander.train
# lr_result = commander.linear_regression(
#     predictors = main_study.adj_mfi_data.iloc[:,:-1],
#     outcome = main_study.patient_data["IOP_Diagnosis"],
#     backward_elimination = True
# )
# lr_result.model.summary()

# main_study.adj_mfi_data.head(3)
# main_study.patient_data.head(3)
# main_study.patient_data.columns


# control_data
# control_mean.iloc[control_cols].index


# #create a bar graph of the row averages, sorted by mean value
# import matplotlib.pyplot as plt

# bead_calibrated_data.mean(axis=1).sort_values().plot(kind='bar')
# plt.show()

# #create a bar graph of the column averages, sorted by mean value, excluding the top 2 values

# bead_calibrated_data.mean(axis=0).sort_values(ascending=False)[2:].plot(kind='bar')
# plt.show()


# train = commander.train
# test = commander.test


# def backward_elimination_aic( X, y):
#     model = sm.OLS(y,X).fit()
#     current_aic = model.aic
#     while True:
#         aic_values = []
#         models = []
#         for i in range(1, X.shape[1]):
#                 X_new = np.delete(X, i, axis=1)
#                 model_new = sm.OLS(y, X_new).fit()
#                 aic_values.append(model_new.aic)
#                 models.append(model_new)
#         min_aic = min(aic_values)
#         if min_aic < current_aic:
#                 current_aic = min_aic
#                 best_model_index = aic_values.index(min_aic)
#                 X = np.delete(X, best_model_index + 1, axis=1)
#                 model = models[best_model_index]
#         else:
#             break
#     return model


# def linear_regression(
#     predictors: pd.DataFrame,
#     outcome: pd.Series,
#     confounders: pd.DataFrame = None,
#     polynomial=False,
#     degree=None,
#     holdout: float = 0.1,
#     scale=True,
#     accuracy_threshold=0.1,
#     backward_elimination=True,
# ):
#     ## Structuring the data
#     ############################
#     # Align the indices of predictors and outcome
#     common_indices = predictors.index.intersection(outcome.index)
#     predictors = predictors.loc[common_indices]
#     outcome = outcome.loc[common_indices]

#     def apply_scaling(data):
#         scaler = StandardScaler()
#         return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

#     if confounders:
#         confounders = confounders.loc[common_indices]
#         if scale:
#             confounders = apply_scaling(confounders)

#     # Scale predictors if scale is True
#     if scale:
#         predictors = apply_scaling(predictors)

#     poly = None
#     if polynomial:
#         poly = PolynomialFeatures(degree=degree, include_bias=False)
#         predictors = pd.DataFrame(
#             poly.fit_transform(predictors),
#             columns=poly.get_feature_names_out(),
#         )

#     # Remove any rows with missing values
#     combined = pd.concat([predictors, confounders, outcome], axis=1).dropna()
#     predictors = combined.iloc[:, :-1]
#     outcome = combined.iloc[:, -1]

#     X = predictors.values
#     y = outcome.values

#     # Initialize lists to store predictions and outcomes
#     all_train_preds = []
#     all_test_preds = []
#     all_test_outcomes = []
#     all_train_outcomes = []
#     mse_list = []
#     r2_list = []
#     test_indices = []

#     for train_index, test_index in zip(train, test):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         X_train_const = sm.add_constant(X_train, has_constant="add")
#         X_test_const = sm.add_constant(X_test, has_constant="add")

#         # Fit the model on the training data
#         model = sm.OLS(y_train, X_train_const).fit()
#         model.summary()
#         # Predict on the test data
#         y_pred = model.predict(X_test_const)

#         # Save predictions and actual outcomes
#         all_test_preds.extend(y_pred)  # Append predictions to the list
#         all_test_outcomes.extend(y_test)  # Append actual test outcomes

#         # calculate MSE and R2 and append to the lists
#         mse = mean_squared_error(y_test, y_pred)
#         mse_list.append(mse)
#         ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
#         ss_res = np.sum((y_test - y_pred) ** 2)
#         r2 = 1 - ss_res / ss_total
#         r2_list.append(r2)

#     average_mse = np.mean(mse_list)
#     average_r2 = np.mean(r2_list)

#     ##Build the full model. Here we are using main and holdout as terms for train and test to prevent confusion with the cross-validation loop.
#     x_main, x_holdout, y_main, y_holdout = train_test_split(
#         X, y, test_size=holdout, random_state=42
#     )
#     x_main_const = sm.add_constant(x_main, has_constant="add")
#     x_holdout_const = sm.add_constant(x_holdout, has_constant="add")

#     if backward_elimination:
#         final_model = backward_elimination_aic(x_main_const, y_main)
#         # Get the columns used in the final model (after backward elimination)
#         final_model_columns = final_model.model.exog_names
#         selected_columns = final_model.model.exog.shape[1]
#         #selected_columns = X_train_const[:, final_model.model.exog_names].shape[1]
#         x_holdout_const = x_holdout_const[:, :selected_columns]
#         y_holdout_pred = final_model.predict(x_holdout_const)
#     else:
#         final_model = sm.OLS(y_main, x_main_const).fit()
#         y_holdout_pred = final_model.predict(x_holdout_const)
#     # final_model = sm.OLS(y_main, x_main_const).fit()
#     # Evaluate specifically this model on the holdout set
#     #x_holdout_const = sm.add_constant(x_holdout, has_constant="add")
#     #y_holdout_pred = final_model.predict(x_holdout_const)
#     final_model.aic
#     model_mse = mean_squared_error(y_holdout, y_holdout_pred)
#     model_r2 = final_model.rsquared

#     # Calculate Predictive Accuracy for the holdout set
#     model_accuracy = np.mean(
#         np.abs(y_holdout - y_holdout_pred) <= accuracy_threshold
#     )

#     # Beta Coefficient and 95% CI
#     params = final_model.params
#     conf = final_model.conf_int()
#     beta = params[1:]
#     ci_lower = conf[1:, 0]
#     ci_upper = conf[1:, 1]
#     ci = (ci_lower, ci_upper)
#     p_value = final_model.pvalues[1:]

#     equation = (
#         f"{final_model.params[0]:.3f}"  # The intercept is the first parameter
#     )
#     for coef, name in zip(final_model.params[1:], predictors.columns):
#         equation += f" + {coef:.3f} * {name}"

#     return LinearRegressionResult(
#         model=final_model,
#         equation=equation,
#         average_mse=average_mse,
#         average_r2=average_r2,
#         train_preds=all_train_preds,
#         test_preds=all_test_preds,
#         train_outcomes=all_train_outcomes,
#         test_outcomes=all_test_outcomes,
#         model_accuracy=model_accuracy,
#         model_mse=model_mse,
#         model_r2=model_r2,
#         beta=beta,
#         ci=ci,
#         p_value=p_value,
#         test_indices=test_indices,
#         poly=poly,  # Return the polynomial transformer
#         plot_data={
#             "x_test": x_holdout,
#             "y_test": y_holdout,
#             "y_pred": y_holdout_pred,
#         },
#     )


# final_model = sm.OLS(y_main, x_main_const).fit()
# final_model.aic
# aic_add = []
# for i in range(x_main_const.shape[1]):
#     print(i)
#     x_new = np.delete(x_main_const, i, axis=1)
#     model_new = sm.OLS(y_main, x_new).fit()
#     if model_new.aic < final_model.aic:
#         aic_add.append(i)
#         final_model = model_new


# def backward_elimination(X: pd.DataFrame,y : pd.Series, predictor_names, criterion='aic', p_value_threshold=0.05):
#     # """
#     # Perform backward elimination by iteratively removing features based on p-value or AIC.
#     # Args:
#     #     X: The feature matrix (with a constant column for the intercept).
#     #     y: The target variable.
#     #     predictor_names: List of predictor names corresponding to the columns in X.
#     #     criterion: The criterion to use for elimination ('p-value' or 'aic').
#     #     p_value_threshold: The significance level to retain a predictor (default is 0.05, used when criterion is 'p-value').
#     # Returns:
#     #     final_model: The final OLS model after backward elimination.
#     #     selected_predictors: List of remaining predictor names.
#     # """
#     included_predictors = list(X.columns)
#     #included_names = list(predictor_names)
#     while True:
#         #fit the model with the current set of predictors
#         model = sm.OLS(y, X[included_predictors]).fit()
#         if criterion == 'p-value':
#             #get the p-values of the predictors
#             p_values = model.pvalues
#             #find the predictor with the highest p-value
#             worst_p_value = p_values.max()
#             #if the worst p-value is above the threshold, remove the predictor
#             if worst_p_value > p_value_threshold:
#                 worst_feature = p_values.idxmax()
#                 print(f"Removing {worst_feature} with p-value {worst_p_value:.4f}")
#                 included_predictors.remove(worst_feature)
#                 #included_names.pop(included_predictors.index(worst_feature))
#             else:
#                 break
#         elif criterion == 'aic':
#             #get the AIC of the model
#             current_aic = model.aic
#             aic_values = []
#             models = []
#             for i in range(1, X[included_predictors].shape[1]):
#                 X_new = X[included_predictors].iloc[:, i:]
#                 model_new = sm.OLS(y, X_new).fit()
#                 aic_values.append(model_new.aic)
#                 models.append(model_new)
#             # Find the model with the lowest AIC
#             min_aic = min(aic_values)
#             if min_aic < current_aic:
#                 best_model_index = aic_values.index(min_aic)
#                 worst_predictor = included_predictors[best_model_index + 1] # Skip the constant column
#                 print(f"Removing {worst_predictor} with AIC {current_aic:.4f}")
#                 included_predictors.remove(worst_predictor)
#                 #included_names.pop(included_predictors.index(worst_predictor))
#             else:
#                 break
#         final_model = sm.OLS(y, X[included_predictors]).fit()
#     return final_model, included_predictors

# predictors = main_study.adj_mfi_data.iloc[:,:-1]
# predictor_names = main_study.adj_mfi_data.iloc[:,:-1].columns
# outcome = main_study.patient_data["IOP_Diagnosis"].values

# model, included_predictors = backward_elimination(predictors, outcome, predictor_names, criterion='aic', p_value_threshold=0.05)
# model.summary()

# import pandas as pd
# from statsmodels.stats.outliers_influence import variance_inflation_factor

# def calculate_vif(X: pd.DataFrame):
#     """
#     Calculate the Variance Inflation Factor (VIF) for each predictor in the DataFrame.

#     Args:
#         X: The feature matrix (Pandas DataFrame).

#     Returns:
#         vif_df: A DataFrame containing the predictors and their corresponding VIF values.
#     """

#     vif_data = pd.DataFrame()
#     vif_data["Predictor"] = X.columns

#     # Calculate VIF for each predictor
#     vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

#     return vif_data

# vif_df = calculate_vif(predictors)

# import seaborn as sns
# import matplotlib.pyplot as plt

# # Generate a correlation matrix
# corr_matrix = predictors.corr()

# # Plot a heatmap to visualize the correlations
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title("Correlation Matrix of Predictors")
# plt.show()

# def remove_high_vif_predictors(X: pd.DataFrame, threshold=10.0):
#     """
#     Iteratively remove predictors with VIF values higher than the threshold.

#     Args:
#         X: The feature matrix (Pandas DataFrame).
#         threshold: The VIF threshold above which predictors will be removed.

#     Returns:
#         X_reduced: The DataFrame with predictors having VIF below the threshold.
#         removed_predictors: A list of predictors that were removed.
#     """

#     removed_predictors = []
#     while True:
#         # Calculate VIF for all predictors
#         vif_df = calculate_vif(X)

#         # Find the predictor with the highest VIF
#         max_vif = vif_df["VIF"].max()

#         if max_vif > threshold:
#             # Get the name of the predictor with the highest VIF
#             max_vif_predictor = vif_df.loc[vif_df["VIF"].idxmax(), "Predictor"]
#             print(f"Removing {max_vif_predictor} with VIF = {max_vif:.2f}")

#             # Remove the predictor from the dataset
#             X = X.drop(columns=[max_vif_predictor])
#             removed_predictors.append(max_vif_predictor)
#         else:
#             # Break the loop when all VIFs are below the threshold
#             break

#     return X, removed_predictors

# # Remove predictors with high VIF values
# X_reduced, removed_predictors = remove_high_vif_predictors(predictors, threshold=10.0)
# X_reduced.columns
