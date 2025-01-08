# Data handling
import pandas as pd
import numpy as np
import pickle

# Data analysis and clustering
from scipy.cluster.hierarchy import linkage, dendrogram

# Visualization
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Custom modules
from classes import xmap_qc, CorrelationAnalyzer, Visualizer, ClusterTools

# Load the saved study object using pickle
with open("data/study.pickle", "rb") as f:
    study = pickle.load(f)

# Perform quality control and filter proteins
qc = xmap_qc(study.normalized_data)
cleaned_proteins = qc.filter_low_variance_proteins(threshold=1e-5)


# Identify continuous and categorical columns
continuous_cols = [
    col for col in study.patient_data.columns 
    if study.patient_data_coldic[col] == "continuous"
]

categorical_cols = [
    col for col in study.patient_data.columns 
    if study.patient_data_coldic[col] in {"categorical", "binary"}
]

### Restructure the patient data
# convert VFI and VFI_3 to VFI loss
study.patient_data["VFI_Loss"] = 100 - study.patient_data["VFI_Diagnosis"]
study.patient_data["VFI_Loss_3"] = 100 - study.patient_data["VFI_3"]
study.patient_data.drop(columns=["VFI_Diagnosis", "VFI_3"], inplace=True)

# replace vfi values in the patient_data_coldic
study.patient_data_coldic["VFI_Loss"] = "continuous"
study.patient_data_coldic["VFI_Loss_3"] = "continuous"
study.patient_data_coldic.pop("VFI_Diagnosis")
study.patient_data_coldic.pop("VFI_3")

# Replace VFI columns
continuous_cols = [
    "VFI_Loss" if col == "VFI_Diagnosis" else 
    "VFI_Loss_3" if col == "VFI_3" else col
    for col in continuous_cols
]

diagnosis_cols = [col for col in continuous_cols if not col.endswith("_3") and col != "ROP_Percent"]
three_cols = [col for col in continuous_cols if col.endswith("_3")]
continuous_cols = diagnosis_cols + three_cols + ["ROP_Percent"]

# Remove "Medicine_List" and "Active_Ingredients" from categorical columns
categorical_cols.remove("Medicine_List")
categorical_cols.remove("Active_Ingredients")
study.patient_data["GPA"] = study.patient_data["GPA"].str.lower()

periodic_cols = [col for col in categorical_cols if "_" in col]
other_categorical_cols = [col for col in categorical_cols if col not in periodic_cols + ["GPA"]]
# sort periodic columns based on the value after the underscore
periodic_cols = sorted(periodic_cols, key=lambda x: int(x.split("_")[1]))
categorical_cols = other_categorical_cols + periodic_cols + ["GPA"]


#scale the cleaned_proteins row-wise
cleaned_proteins_scaled = cleaned_proteins.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
cleaned_proteins_unscaled = cleaned_proteins

correlation_analyzer = CorrelationAnalyzer(cleaned_proteins_scaled, study.patient_data)
correlation_analyzer_unscaled = CorrelationAnalyzer(cleaned_proteins_unscaled, study.patient_data)

pp_corr = correlation_analyzer.correlation_continuous(single_dataset=True)
spearman_corr = correlation_analyzer.correlation_continuous(
    continuous_cols, method="spearman",
)
spearman_corr_unscaled = correlation_analyzer_unscaled.correlation_continuous(
    continuous_cols, method="spearman",
)

#Get the table of top correlations
def get_top_correlations(corr_matrix, pval_matrix, significance_level=0.05, n=10, save_path=None):
    """
    Extracts the top n significant correlation pairs from a correlation matrix, 
    based on a p-value threshold, excluding pairs where the base protein names are the same.

    Parameters:
    - corr_matrix (pd.DataFrame): Square dataframe with protein correlation values.
    - pval_matrix (pd.DataFrame): Square dataframe with adjusted p-values for each correlation.
    - significance_level (float): P-value threshold to consider significance.
    - n (int): Number of top correlations to return.
    - save_path (str, optional): Path to save the resulting Excel file. If None, returns a DataFrame.

    Returns:
    - pd.DataFrame: DataFrame containing the top n protein pairs, their correlation, and adjusted p-value.
    """
    # Ensure the correlation matrix has appropriate index
    if corr_matrix.index.name is None:
        corr_matrix = corr_matrix.reset_index().rename(columns={'index': 'Protein1'})
    else:
        corr_matrix = corr_matrix.reset_index()
    if pval_matrix.index.name is None:
        pval_matrix = pval_matrix.reset_index().rename(columns={'index': 'Protein1'})
    else:
        pval_matrix = pval_matrix.reset_index()

    # Melt the correlation and p-value matrices to long format
    corr_long = corr_matrix.melt(id_vars=corr_matrix.columns[0], var_name='Protein2', value_name='Correlation')
    corr_long = corr_long.rename(columns={corr_matrix.columns[0]: 'Protein1'})
    pval_long = pval_matrix.melt(id_vars=pval_matrix.columns[0], var_name='Protein2', value_name='Adjusted_pval')
    pval_long = pval_long.rename(columns={pval_matrix.columns[0]: 'Protein1'})

    # Merge the correlation and p-value data on Protein1 and Protein2
    df_long = pd.merge(corr_long, pval_long, on=['Protein1', 'Protein2'])

    # Extract base protein names to filter out self-correlations
    df_long['Protein1_base'] = df_long['Protein1'].str.split('_').str[0]
    df_long['Protein2_base'] = df_long['Protein2'].str.split('_').str[0]

    # Filter out pairs with the same base protein and non-significant p-values
    df_filtered = df_long[(df_long['Protein1_base'] != df_long['Protein2_base']) & 
                          (df_long['Adjusted_pval'] < significance_level)].copy()

    # To avoid duplicate pairs (e.g., (A, B) and (B, A)), sort the protein names and drop duplicates
    df_filtered['Protein_pair'] = df_filtered.apply(
        lambda row: tuple(sorted([row['Protein1'], row['Protein2']])), axis=1
    )
    df_unique = df_filtered.drop_duplicates(subset='Protein_pair')

    # Sort by Adjusted p-value (significance) first, then by correlation (descending)
    df_sorted = df_unique.sort_values(by=['Adjusted_pval', 'Correlation'], ascending=[True, False])

    # Select the top n correlations
    df_top = df_sorted.head(n)

    # Split the protein pairs back into separate columns and reorder the columns
    result = df_top[['Protein1', 'Protein2', 'Correlation', 'Adjusted_pval']].reset_index(drop=True)

    # Save the results to an Excel file if a save path is provided
    if save_path:
        result.to_excel(save_path, index=False)
        print(f"Top {n} significant correlations saved to '{save_path}'")
    else:
        return result

top_protein_correlations = get_top_correlations(
    corr_matrix=pp_corr.corr,
    pval_matrix=pp_corr.adj_p,
    significance_level=0.05,
    n=20,
    # save_path="reports/top_protein_correlations.xlsx"
)
top_clinical = get_top_correlations(
    corr_matrix=spearman_corr_unscaled.corr, 
    pval_matrix=spearman_corr_unscaled.adj_p, 
    significance_level=0.1, 
    n=20, 
    # save_path="reports/top_clinical_correlations.xlsx"
    )

def get_correlations_to_target(corr_matrix, pval_matrix, corr_parameter, n=10, save_path=None):
    """
    Extracts the top n significant correlations involving a specified target parameter (e.g., clinical parameter) 
    from a correlation matrix, based on their adjusted p-values.

    Parameters:
    - corr_matrix (pd.DataFrame): Square dataframe with correlation values.
    - pval_matrix (pd.DataFrame): Square dataframe with adjusted p-values for each correlation.
    - corr_parameter (str): Name of the target parameter for which to find correlations.
    - n (int): Number of top correlations to return.
    - save_path (str, optional): Path to save the resulting Excel file. If None, returns a DataFrame.

    Returns:
    - pd.DataFrame: DataFrame containing the top n correlations for the specified parameter.
    """
    import pandas as pd

    # Ensure the correlation matrix and p-value matrix have appropriate indices
    #if corr_matrix.index.name is None:
    corr_matrix.index.name = 'Parameter1'
    #if pval_matrix.index.name is None:
    pval_matrix.index.name = 'Parameter1'

    # Melt the correlation and p-value matrices into long format
    corr_long = corr_matrix.reset_index().melt(id_vars='Parameter1', var_name='Parameter2', value_name='Correlation')
    pval_long = pval_matrix.reset_index().melt(id_vars='Parameter1', var_name='Parameter2', value_name='Adjusted_pval')

    # Merge the correlation and p-value dataframes
    df_long = pd.merge(corr_long, pval_long, on=['Parameter1', 'Parameter2'])

    # Filter for rows involving the specified parameter
    df_filtered = df_long[(df_long['Parameter1'] == corr_parameter) | (df_long['Parameter2'] == corr_parameter)]

    # Avoid self-correlation
    df_filtered = df_filtered[df_filtered['Parameter1'] != df_filtered['Parameter2']]

    # Sort by adjusted p-value (ascending) and then by correlation (descending)
    df_sorted = df_filtered.sort_values(by=['Adjusted_pval', 'Correlation'], ascending=[True, False])

    # Select the top n correlations
    df_top = df_sorted.head(n).reset_index(drop=True)

    # Update the output DataFrame to make the target parameter explicit
    df_top['Target'] = corr_parameter
    df_top['Correlated_with'] = df_top.apply(
        lambda row: row['Parameter1'] if row['Parameter2'] == corr_parameter else row['Parameter2'], axis=1
    )

    # Reorder columns for clarity
    result = df_top[['Target', 'Correlated_with', 'Correlation', 'Adjusted_pval']]

    # Save the results to an Excel file if a save path is provided
    if save_path:
        result.to_excel(save_path, index=False)
        print(f"Top {n} correlations for target '{corr_parameter}' saved to '{save_path}'")
    else:
        return result

def compute_correlations_for_all_targets(corr_matrix, pval_matrix, continuous_cols, n=10):
    """
    Computes the top correlations for each target in continuous_cols and creates a dictionary 
    where the key is the target, and the value is a list of parameters it is most strongly correlated with.

    Parameters:
    - corr_matrix (pd.DataFrame): Square dataframe with correlation values.
    - pval_matrix (pd.DataFrame): Square dataframe with adjusted p-values for each correlation.
    - continuous_cols (list): List of target parameters to compute correlations for.
    - n (int): Number of top correlations to extract for each target.

    Returns:
    - dict: Dictionary where keys are target parameters and values are lists of correlated parameters.
    """
    correlation_dict = {}

    for target in continuous_cols:
        # Get the top correlations for the current target
        result = get_correlations_to_target(corr_matrix, pval_matrix, corr_parameter=target, n=n)

        # Extract the 'Correlated_with' column as a list
        correlated_with_list = result['Correlated_with'].tolist()
        correlated_with_list = [item.split("_")[0] for item in correlated_with_list]
        # Store the list in the dictionary
        correlation_dict[target] = correlated_with_list

    return correlation_dict

top_correlation_dict = compute_correlations_for_all_targets(
    corr_matrix=spearman_corr_unscaled.corr,
    pval_matrix=spearman_corr_unscaled.adj_p,
    continuous_cols=continuous_cols,
    n=10,
)
top_correlation_proteins = top_protein_correlations["Protein1"].tolist() + top_protein_correlations["Protein2"].tolist()
top_correlation_dict["Proteins"] = list(set([protein.split("_")[0] for protein in top_correlation_proteins]))

#save the top_correlation_dict
with open("data/top_correlation_proteins.pickle", "wb") as f:
    pickle.dump(top_correlation_dict, f)


cluster_tool = ClusterTools(study)
cluster_tool.build_clusters(method="complete")

#Figure 1
Visualizer.plot_clustered_heatmap(spearman_corr_unscaled.corr, spearman_corr_unscaled.adj_p, clean_X_names=True, cbar_pos=(0.05,0.6,0.02,0.2), text_x=0.06, method="complete", column_order=cluster_tool.col_order, save_path="plots/figure_01.png")

#Figure 2
cluster_tool.dendrogram.savefig("plots/figure_02.png", dpi=300)

#Figure 3
Visualizer.plot_clustered_heatmap(pp_corr.corr, pp_corr.adj_p, clean_X_names=True, clean_Y_names=True, cbar_pos=(0.05,0.6,0.02,0.2), text_x=0.06, save_path="plots/figure_03.png", figsize=(24, 24), asterisk_size=12)


### Save cluster lists to files
cluster_1 = cluster_tool.clusters_dict[1]
cluster_2 = cluster_tool.clusters_dict[2]
cluster_1_proteins = list(set([protein.split("_")[0] for protein in cluster_1]))
cluster_2_proteins = list(set([protein.split("_")[0] for protein in cluster_2]))
with open("data/cluster_1_proteins.pickle", "wb") as f:
    pickle.dump(cluster_1_proteins, f)
#save cluster_2_proteins to a file
with open("data/cluster_2_proteins.pickle", "wb") as f:
    pickle.dump(cluster_2_proteins, f)

#CCA
cca_full_set = correlation_analyzer_unscaled.cca_continuous_variables(continuous_cols)
cca_cluster_1 = correlation_analyzer_unscaled.cca_continuous_variables(continuous_cols, X_df=cleaned_proteins[cluster_1])
cca_cluster_2 = correlation_analyzer_unscaled.cca_continuous_variables(continuous_cols, X_df=cleaned_proteins[cluster_2])

#PERMANOVA
permanova_full_set = correlation_analyzer_unscaled.permanova_categorical_variables(categorical_cols)
permanova_cluster_1 = correlation_analyzer_unscaled.permanova_categorical_variables(categorical_cols, X_df=cleaned_proteins[cluster_1])
permanova_cluster_2 = correlation_analyzer_unscaled.permanova_categorical_variables(categorical_cols, X_df=cleaned_proteins[cluster_2])

#Table 1
# Initialize empty DataFrames for all analysis tables
cca_tables = {
    "Full Set": pd.DataFrame(columns=["Variable", "Test Statistic", "P-value"]),
    "Cluster 1": pd.DataFrame(columns=["Variable", "Test Statistic", "P-value"]),
    "Cluster 2": pd.DataFrame(columns=["Variable", "Test Statistic", "P-value"]),
}

permanova_tables = {
    "Full Set": pd.DataFrame(columns=["Variable", "Test Statistic", "P-value"]),
    "Cluster 1": pd.DataFrame(columns=["Variable", "Test Statistic", "P-value"]),
    "Cluster 2": pd.DataFrame(columns=["Variable", "Test Statistic", "P-value"]),
}

# Populate CCA tables
cca_results = {
    "Full Set": cca_full_set,
    "Cluster 1": cca_cluster_1,
    "Cluster 2": cca_cluster_2,
}

for key, result in cca_results.items():
    cca_tables[key]["Variable"] = result[1].index
    cca_tables[key]["Test Statistic"] = result[1]["Canonical Correlation"].round(3).values
    cca_tables[key]["P-value"] = result[1]["p-value"].round(3).values

# Populate PERMANOVA tables
permanova_results = {
    "Full Set": permanova_full_set,
    "Cluster 1": permanova_cluster_1,
    "Cluster 2": permanova_cluster_2,
}

for key, result in permanova_results.items():
    for variable, stats in result.items():
        test_statistic = round(stats["test statistic"], 3)
        p_value = round(stats["p-value"], 3)
        line_vals = [variable, test_statistic, p_value]
        permanova_tables[key].loc[len(permanova_tables[key])] = line_vals


def create_combined_table_bold_headers(cca_tables, permanova_tables):
    """
    Creates a combined table for CCA and PERMANOVA results, reporting only Variables and P-Values,
    with bold headers and row names, and minimal whitespace between tables.

    Args:
        cca_tables (dict): Dictionary with CCA dataframes for "Full Set", "Cluster 1", and "Cluster 2".
        permanova_tables (dict): Dictionary with PERMANOVA dataframes for "Full Set", "Cluster 1", and "Cluster 2".

    Returns:
        None
    """
    def preprocess_table(table, column_name="P-value"):
        # Ensure "P-value" column is numeric and handle invalid values
        table[column_name] = pd.to_numeric(table[column_name], errors='coerce')
        return table

    # Preprocess tables to ensure numeric P-values
    for key in cca_tables:
        cca_tables[key] = preprocess_table(cca_tables[key])
    for key in permanova_tables:
        permanova_tables[key] = preprocess_table(permanova_tables[key])

    # Prepare combined data for CCA
    cca_combined = cca_tables["Full Set"][["Variable", "P-value"]].rename(columns={"P-value": "P-Value (Full Set)"})
    cca_combined["P-Value (Cluster 1)"] = cca_tables["Cluster 1"]["P-value"].values
    cca_combined["P-Value (Cluster 2)"] = cca_tables["Cluster 2"]["P-value"].values

    # Prepare combined data for PERMANOVA
    permanova_combined = permanova_tables["Full Set"][["Variable", "P-value"]].rename(columns={"P-value": "P-Value (Full Set)"})
    permanova_combined["P-Value (Cluster 1)"] = permanova_tables["Cluster 1"]["P-value"].values
    permanova_combined["P-Value (Cluster 2)"] = permanova_tables["Cluster 2"]["P-value"].values

    # Plot combined tables
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1])

    # Add CCA Table
    ax_cca = fig.add_subplot(gs[0, 0])
    ax_cca.axis('off')
    ax_cca.set_title("CCA Results", fontsize=14, weight='bold', pad=0, y=0.95)
    cca_table = ax_cca.table(
        cellText=cca_combined.values,
        colLabels=cca_combined.columns,
        cellLoc='center',
        loc='center'
    )
    cca_table.auto_set_font_size(False)
    cca_table.set_fontsize(14)
    cca_table.auto_set_column_width(col=list(range(len(cca_combined.columns))))

    # Make headers and row names bold
    for (i, j), cell in cca_table.get_celld().items():
        if i == 0 or j == 0:  # Header row or first column
            cell.set_text_props(weight='bold')
        if i > 0 and j > 0:  # Highlight significant p-values
            try:
                p_value = float(cca_combined.iloc[i - 1, j - 1])
                if p_value < 0.05:
                    cell.set_facecolor('#ffcccc')
            except ValueError:
                continue

    # Add PERMANOVA Table
    ax_permanova = fig.add_subplot(gs[1, 0])
    ax_permanova.axis('off')
    ax_permanova.set_title("PERMANOVA Results", fontsize=14, weight='bold', pad=0, y=0.95)
    permanova_table = ax_permanova.table(
        cellText=permanova_combined.values,
        colLabels=permanova_combined.columns,
        cellLoc='center',
        loc='center'
    )
    permanova_table.auto_set_font_size(False)
    permanova_table.set_fontsize(14)
    permanova_table.auto_set_column_width(col=list(range(len(permanova_combined.columns))))

    # Make headers and row names bold
    for (i, j), cell in permanova_table.get_celld().items():
        if i == 0 or j == 0:  # Header row or first column
            cell.set_text_props(weight='bold')
        if i > 0 and j > 0:  # Highlight significant p-values
            try:
                p_value = float(permanova_combined.iloc[i - 1, j - 1])
                if p_value < 0.05:
                    cell.set_facecolor('#ffcccc')
            except ValueError:
                continue

    # Adjust layout to minimize whitespace
    plt.subplots_adjust(hspace=0)  # Further reduced top spacing
    # plt.tight_layout()
    plt.show()


create_combined_table_bold_headers(cca_tables, permanova_tables)

