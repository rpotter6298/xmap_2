# File: classes/Eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, kendalltau, pearsonr, spearmanr, ttest_ind, f_oneway
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class VarianceAnalyzer:
    """
    A class to compute and analyze the variance of proteins.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the VarianceAnalyzer with the provided protein data.

        Args:
            data (pd.DataFrame): Normalized protein expression data.
        """
        self.data = data

    def compute_variance(self) -> pd.Series:
        """
        Computes the variance for each protein.

        Returns:
            pd.Series: Variance of each protein.
        """
        try:
            variance = self.data.var()
            return variance
        except Exception as e:
            logging.error(f"Error computing variance: {e}")
            return pd.Series(dtype=float)

    def get_high_variance_proteins(self, threshold: float) -> list:
        """
        Retrieves a list of proteins with variance above the specified threshold.

        Args:
            threshold (float): Variance threshold.

        Returns:
            list: List of protein names with high variance.
        """
        variance = self.compute_variance()
        high_variance = variance[variance > threshold].index.tolist()
        logging.info(f"Proteins with variance above {threshold}: {len(high_variance)}")
        return high_variance

    def plot_top_variances(self, top_n: int = 20):
        """
        Plots the top N proteins with the highest variance.

        Args:
            top_n (int, optional): Number of top proteins to plot. Defaults to 20.
        """
        try:
            variance = self.compute_variance().sort_values(ascending=False).head(top_n)
            plt.figure(figsize=(12, 8))
            sns.barplot(x=variance.values, y=variance.index, palette="viridis")
            plt.xlabel("Variance")
            plt.ylabel("Proteins")
            plt.title(f"Top {top_n} Proteins by Variance")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting top variances: {e}")


class CorrelationAnalyzer:
    """
    A class to compute various types of correlations between proteins and patient metrics,
    as well as protein-protein correlations.
    """

    def __init__(self, protein_data: pd.DataFrame, patient_data: pd.DataFrame):
        """
        Initializes the CorrelationAnalyzer with protein and patient data.

        Args:
            protein_data (pd.DataFrame): High-variance protein data.
            patient_data (pd.DataFrame): Patient metrics data.
        """
        self.protein_data = protein_data
        self.patient_data = patient_data

    def point_biserial_correlation(self, binary_columns: list, mapping_dict: dict = None) -> pd.DataFrame:
        """
        Computes point-biserial correlations for binary patient variables.
        Automatically encodes non-numeric binary columns to numeric.

        Args:
            binary_columns (list): List of binary patient metric column names.
            mapping_dict (dict, optional): Dictionary mapping column names to their value mappings.
                                           Example: {'Gender': {'Male': 0, 'Female': 1}}

        Returns:
            pd.DataFrame: Correlation coefficients with proteins as rows and binary variables as columns.
        """
        correlations = {}
        for col in binary_columns:
            # Copy the column to avoid modifying the original DataFrame
            patient_col = self.patient_data[col].copy()

            # Check if the column is numeric
            if not pd.api.types.is_numeric_dtype(patient_col):
                unique_vals = patient_col.dropna().unique()
                if len(unique_vals) != 2:
                    logging.warning(f"Column '{col}' is not binary (found {len(unique_vals)} unique values). Skipping.")
                    continue

                # Use provided mapping if available
                if mapping_dict and col in mapping_dict:
                    patient_col = patient_col.map(mapping_dict[col])
                    logging.info(f"Column '{col}' was mapped using provided mapping: {mapping_dict[col]}")
                else:
                    # Create a default mapping: first unique value to 0, second to 1
                    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                    patient_col = patient_col.map(mapping)
                    logging.info(f"Column '{col}' was auto-mapped to numeric values: {mapping}")

                if patient_col.isnull().any():
                    logging.warning(f"Column '{col}' contains values outside the binary mapping. These will be treated as NaN.")

            # Initialize list to store correlation coefficients for this column
            correlations[col] = []
            for protein in self.protein_data.columns:
                protein_values = self.protein_data[protein]
                # Align indices to handle any potential mismatches
                aligned_patient, aligned_protein = patient_col.align(protein_values, join='inner')
                # Drop any NaN values resulting from alignment or mapping
                valid = aligned_patient.notna() & aligned_protein.notna()
                if valid.sum() == 0:
                    corr = np.nan
                    logging.warning(f"No valid data for correlation between '{col}' and '{protein}'.")
                else:
                    corr, _ = pointbiserialr(aligned_patient[valid], aligned_protein[valid])
                correlations[col].append(corr)

        # Convert the dictionary to a DataFrame
        if correlations:
            corr_df = pd.DataFrame(correlations, index=self.protein_data.columns).T
            return corr_df
        else:
            logging.warning("No valid binary columns provided for point-biserial correlation.")
            return pd.DataFrame()

    def kendall_tau_correlation(self, continuous_columns: list) -> pd.DataFrame:
        """
        Computes Kendall's Tau correlations for continuous patient variables.

        Args:
            continuous_columns (list): List of continuous patient metric column names.

        Returns:
            pd.DataFrame: Correlation coefficients.
        """
        correlations = {}
        for col in continuous_columns:
            correlations[col] = []
            for protein in self.protein_data.columns:
                tau, _ = kendalltau(self.patient_data[col], self.protein_data[protein])
                correlations[col].append(tau)
        if correlations:
            corr_df = pd.DataFrame(correlations, index=self.protein_data.columns).T
            return corr_df
        else:
            logging.warning("No continuous columns provided for Kendall's Tau correlation.")
            return pd.DataFrame()

    def pearson_correlation(self, continuous_columns: list) -> pd.DataFrame:
        """
        Computes Pearson correlations for continuous patient variables.

        Args:
            continuous_columns (list): List of continuous patient metric column names.

        Returns:
            pd.DataFrame: Correlation coefficients.
        """
        correlations = {}
        for col in continuous_columns:
            correlations[col] = []
            for protein in self.protein_data.columns:
                corr, _ = pearsonr(self.patient_data[col], self.protein_data[protein])
                correlations[col].append(corr)
        if correlations:
            corr_df = pd.DataFrame(correlations, index=self.protein_data.columns).T
            return corr_df
        else:
            logging.warning("No continuous columns provided for Pearson correlation.")
            return pd.DataFrame()

    def spearman_correlation(self, continuous_columns: list) -> pd.DataFrame:
        """
        Computes Spearman correlations for continuous patient variables.

        Args:
            continuous_columns (list): List of continuous patient metric column names.

        Returns:
            pd.DataFrame: Correlation coefficients.
        """
        correlations = {}
        for col in continuous_columns:
            correlations[col] = []
            for protein in self.protein_data.columns:
                corr, _ = spearmanr(self.patient_data[col], self.protein_data[protein])
                correlations[col].append(corr)
        if correlations:
            corr_df = pd.DataFrame(correlations, index=self.protein_data.columns).T
            return corr_df
        else:
            logging.warning("No continuous columns provided for Spearman correlation.")
            return pd.DataFrame()

    def protein_protein_correlation(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Computes protein-protein correlations using the specified method.

        Args:
            method (str, optional): Correlation method ('pearson', 'spearman', 'kendall'). Defaults to 'pearson'.

        Returns:
            pd.DataFrame: Correlation matrix between proteins.
        """
        valid_methods = ['pearson', 'spearman', 'kendall']
        if method not in valid_methods:
            logging.error(f"Invalid correlation method '{method}'. Choose from {valid_methods}.")
            return pd.DataFrame()

        try:
            if method == 'pearson':
                corr_matrix = self.protein_data.corr(method='pearson')
            elif method == 'spearman':
                corr_matrix = self.protein_data.corr(method='spearman')
            elif method == 'kendall':
                corr_matrix = self.protein_data.corr(method='kendall')
            return corr_matrix
        except Exception as e:
            logging.error(f"Error computing protein-protein correlations: {e}")
            return pd.DataFrame()

    def anova_f_statistic(self, categorical_columns: list) -> pd.DataFrame:
        """
        Computes ANOVA F-statistics for multi-level categorical patient variables.

        Args:
            categorical_columns (list): List of multi-level categorical patient metric column names.

        Returns:
            pd.DataFrame: ANOVA F-statistics.
        """
        f_statistics = {}
        for col in categorical_columns:
            f_statistics[col] = []
            for protein in self.protein_data.columns:
                # Prepare the data
                data = pd.concat([self.protein_data[protein], self.patient_data[col]], axis=1).dropna()
                if data[col].nunique() < 2:
                    logging.warning(f"Column '{col}' has less than 2 unique values after dropping NaNs. Skipping ANOVA.")
                    f_stat = np.nan
                else:
                    try:
                        model = ols(f'{protein} ~ C({col})', data=data).fit()
                        anova = sm.stats.anova_lm(model, typ=2)
                        f_stat = anova['F'][0]  # Assuming the first row corresponds to the categorical variable
                    except Exception as e:
                        logging.warning(f"ANOVA failed for {protein} vs {col}: {e}")
                        f_stat = np.nan
                f_statistics[col].append(f_stat)
        if f_statistics:
            f_df = pd.DataFrame(f_statistics, index=self.protein_data.columns).T
            return f_df
        else:
            logging.warning("No categorical columns provided for ANOVA.")
            return pd.DataFrame()

    def cramers_v(self, categorical_col1: str, categorical_col2: str) -> float:
        """
        Computes Cramér's V for two categorical variables.

        Args:
            categorical_col1 (str): First categorical column name.
            categorical_col2 (str): Second categorical column name.

        Returns:
            float: Cramér's V statistic.
        """
        confusion_matrix = pd.crosstab(self.patient_data[categorical_col1], self.patient_data[categorical_col2])
        if confusion_matrix.size == 0:
            logging.warning(f"No data to compute Cramér's V between '{categorical_col1}' and '{categorical_col2}'.")
            return np.nan
        chi2 = sm.stats.chisquare(confusion_matrix, axis=None)[0]
        n = confusion_matrix.sum().sum()
        r, k = confusion_matrix.shape
        if n == 0 or min(r, k) <= 1:
            logging.warning(f"Insufficient data to compute Cramér's V between '{categorical_col1}' and '{categorical_col2}'.")
            return np.nan
        return np.sqrt(chi2 / (n * (min(r, k) - 1)))


class Visualizer:
    """
    A class to handle visualization tasks for exploratory data analysis.
    """

    def __init__(self):
        """
        Initializes the Visualizer class.
        """
        pass

    def plot_correlation_heatmap(self, correlation_df: pd.DataFrame, title: str, cmap: str = "coolwarm",
                                  annot: bool = True, fmt: str = ".2f", cbar_label: str = "Correlation"):
        """
        Plots a heatmap for the provided correlation DataFrame.

        Args:
            correlation_df (pd.DataFrame): DataFrame containing correlation coefficients.
            title (str): Title of the heatmap.
            cmap (str, optional): Colormap to use. Defaults to "coolwarm".
            annot (bool, optional): Whether to annotate the heatmap cells. Defaults to True.
            fmt (str, optional): String formatting code for annotations. Defaults to ".2f".
            cbar_label (str, optional): Label for the color bar. Defaults to "Correlation".
        """
        try:
            plt.figure(figsize=(14, 10))
            sns.heatmap(
                correlation_df,
                annot=annot,
                fmt=fmt,
                cmap=cmap,
                cbar_kws={"label": cbar_label},
                linewidths=.5,
                linecolor='gray'
            )
            plt.title(title, fontsize=16)
            plt.xlabel("Patient Metrics", fontsize=12)
            plt.ylabel("Proteins", fontsize=12)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting correlation heatmap: {e}")

    def plot_distribution(self, data: pd.Series, title: str, xlabel: str = "Values",
                          ylabel: str = "Frequency"):
        """
        Plots the distribution of a given data series.

        Args:
            data (pd.Series): Data to plot.
            title (str): Title of the plot.
            xlabel (str, optional): Label for the x-axis. Defaults to "Values".
            ylabel (str, optional): Label for the y-axis. Defaults to "Frequency".
        """
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(data, bins=30, kde=True, color='skyblue')
            plt.title(title, fontsize=16)
            plt.xlabel(xlabel, fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting distribution: {e}")

    def plot_boxplot(self, data: pd.DataFrame, title: str, orient: str = "h"):
        """
        Plots boxplots for each protein to visualize distribution and detect outliers.

        Args:
            data (pd.DataFrame): Protein expression data.
            title (str): Title of the plot.
            orient (str, optional): Orientation of the boxplots ('h' for horizontal, 'v' for vertical). Defaults to "h".
        """
        try:
            plt.figure(figsize=(14, 8))
            sns.boxplot(data=data, orient=orient, palette="Set2")
            plt.title(title, fontsize=16)
            plt.xlabel("Proteins", fontsize=12)
            plt.ylabel("Expression Levels", fontsize=12)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting boxplot: {e}")

    def plot_volcano_3d_with_labels(self, t_test_results: pd.DataFrame, binary_cols: list, p_value_threshold: float = None):
        """
        Creates a 3D volcano plot for multiple binary attributes with persistent labels for significant proteins.

        Args:
            t_test_results (pd.DataFrame): DataFrame containing t-statistics and p-values for binary columns.
            binary_cols (list): List of binary column names.
            p_value_threshold (float, optional): Threshold for p-values to consider significance. If None, no threshold is applied.
        """
        # Initialize lists to collect data
        x_vals = []
        y_vals = []
        z_vals = []
        hover_text = []
        colors = []
        persistent_labels = []

        for col in binary_cols:
            # Construct column names
            t_stat_col = f"{col}_t_stat"
            p_val_col = f"{col}_p_value"
            
            # Check if columns exist
            if t_stat_col not in t_test_results.columns or p_val_col not in t_test_results.columns:
                logging.warning(f"Columns '{t_stat_col}' or '{p_val_col}' not found in t_test_results. Skipping.")
                continue
            
            # Extract relevant data
            df = t_test_results[[t_stat_col, p_val_col]].copy()
            df = df.dropna(subset=[t_stat_col, p_val_col])
            
            # Ensure p-values are numeric
            df[p_val_col] = pd.to_numeric(df[p_val_col], errors='coerce')
            df = df.dropna(subset=[p_val_col])
            
            # Replace p-values of 0 with a very small number to avoid -log10(0)
            df[p_val_col] = df[p_val_col].replace(0, 1e-300)
            
            # Compute -log10(p-value)
            df['-Log10(P-Value)'] = -np.log10(df[p_val_col])  # Updated to use pd.np for compatibility
            
            # Iterate through each protein
            for protein, row in df.iterrows():
                x_vals.append(row[t_stat_col])
                y_vals.append(row['-Log10(P-Value)'])
                z_vals.append(col)
                hover_text.append(f"Protein: {protein}<br>Attribute: {col}<br>T-Stat: {row[t_stat_col]:.2f}<br>P-Value: {row[p_val_col]:.3f}")
                
                # Add persistent label only for significant proteins
                if p_value_threshold is not None and row[p_val_col] < p_value_threshold:
                    persistent_labels.append(protein.split("_")[0])  # Adjust based on your protein naming convention
                    colors.append('red')  # Significant proteins in red
                else:
                    persistent_labels.append('')  # No label for non-significant proteins
                    colors.append('blue')  # Non-significant proteins in blue

        if not x_vals:
            logging.error("No data available for plotting.")
            return

        # Create 3D scatter plot
        fig = go.Figure()

        # Scatter plot with hover information and colors
        fig.add_trace(go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='markers+text',  # Enable markers and text (for labels)
            text=persistent_labels,  # Persistent labels for significant proteins
            textposition="top center",  # Adjust label position
            marker=dict(
                size=5,
                color=colors,
                opacity=0.8
            ),
            hovertext=hover_text,
            hoverinfo='text'
        ))

        # Customize layout
        fig.update_layout(
            title="3D Volcano Plot of Protein-Attribute Relationships",
            scene=dict(
                xaxis_title="T-Statistic",
                yaxis_title="-Log10(P-Value)",
                zaxis_title="Attribute"
            ),
            margin=dict(l=0, r=0, b=0, t=50)
        )

        # Show plot
        fig.show()

    def generate_significant_report(self, t_test_results: pd.DataFrame, binary_cols: list, p_value_threshold: float = 0.05, output_path: str = None) -> pd.DataFrame:
            """
            Generates a report of proteins with significant attributes based on t-testing.

            The report includes:
                - Protein Name
                - List of Significant Attributes
                - Corresponding T-Statistics and P-Values as a list of tuples
                - Effect Directions (Increase/Decrease)
                - Minimum P-Value (for sorting)

            Args:
                t_test_results (pd.DataFrame): DataFrame containing t-statistics and p-values for binary columns.
                binary_cols (list): List of binary attribute column names.
                p_value_threshold (float, optional): Threshold for p-values to consider significance. Defaults to 0.05.
                output_path (str, optional): File path to save the report as a CSV. If None, the report is not saved.

            Returns:
                pd.DataFrame: DataFrame containing the significant report.
            """
            report_data = []

            for protein in t_test_results.index:
                significant_attrs = []
                t_p_values = []
                effect_directions = []
                min_p_value = np.inf  # Initialize with infinity

                for col in binary_cols:
                    t_stat_col = f"{col}_t_stat"
                    p_val_col = f"{col}_p_value"

                    # Check if columns exist
                    if t_stat_col not in t_test_results.columns or p_val_col not in t_test_results.columns:
                        logging.warning(f"Columns '{t_stat_col}' or '{p_val_col}' not found in t_test_results. Skipping.")
                        continue

                    # Retrieve p-value
                    p_val = t_test_results.at[protein, p_val_col]

                    # Check if p-value is below the threshold
                    if pd.notnull(p_val) and p_val < p_value_threshold:
                        significant_attrs.append(col)
                        t_stat = t_test_results.at[protein, t_stat_col]
                        t_p_values.append((t_stat, p_val))
                        direction = "Increase" if t_stat > 0 else "Decrease"
                        effect_directions.append(direction)

                        # Update minimum p-value
                        if p_val < min_p_value:
                            min_p_value = p_val

                if significant_attrs:
                    # Combine the attributes and their statistics
                    attrs_stats = [f"{attr}: ({t:.2f}, {p:.3f}) [{dir_}]" for attr, (t, p), dir_ in zip(significant_attrs, t_p_values, effect_directions)]
                    report_data.append({
                        'Protein': protein.split("_")[0],
                        'Significant Attributes': significant_attrs,
                        'T-Stats and P-Values': t_p_values,
                        'Effect Directions': effect_directions,
                        'Minimum P-Value': min_p_value
                    })

            # Create the report DataFrame
            report_df = pd.DataFrame(report_data)

            if report_df.empty:
                logging.info("No significant protein-attribute relationships found based on the specified threshold.")
                return report_df

            # Sort the DataFrame by 'Minimum P-Value' in ascending order
            report_df = report_df.sort_values(by='Minimum P-Value', ascending=True).reset_index(drop=True)

            # Optional: Format the lists as strings for better readability in CSV
            report_df['Significant Attributes'] = report_df['Significant Attributes'].apply(lambda x: ', '.join(x))
            report_df['T-Stats and P-Values'] = report_df['T-Stats and P-Values'].apply(lambda x: ', '.join([f"({t:.2f}, {p:.3f})" for t, p in x]))
            report_df['Effect Directions'] = report_df['Effect Directions'].apply(lambda x: ', '.join(x))
            report_df['Minimum P-Value'] = report_df['Minimum P-Value'].apply(lambda x: f"{x:.3e}")  # Scientific notation for p-values

            # Reorder columns for better readability
            report_df = report_df[['Protein', 'Significant Attributes', 'T-Stats and P-Values', 'Effect Directions', 'Minimum P-Value']]

            #optional: Remove effect directions and minimum p-value
            report_df = report_df.drop(columns=['Effect Directions', 'Minimum P-Value'])
            # Display the report
            print("Significant Protein-Attribute Relationships Report:")
            print(report_df)

            # Save the report to a CSV file if an output path is provided
            if output_path:
                try:
                    report_df.to_csv(output_path, index=False)
                    logging.info(f"Report successfully saved to '{output_path}'.")
                except Exception as e:
                    logging.error(f"Failed to save report to '{output_path}': {e}")

            return report_df


class GroupComparator:
    """
    A class to perform aggregate data comparisons between groups.
    """

    def __init__(self, data: pd.DataFrame, group_labels: pd.Series):
        """
        Initializes the GroupComparator with data and group labels.

        Args:
            data (pd.DataFrame): Data to compare across groups.
            group_labels (pd.Series): Series containing group labels corresponding to the data rows.
        """
        self.data = data
        self.group_labels = group_labels

    def t_test(self, group1: str, group2: str, protein: str) -> dict:
        """
        Performs an independent t-test between two groups for a specific protein.

        Args:
            group1 (str): Label of the first group.
            group2 (str): Label of the second group.
            protein (str): Protein column name to test.

        Returns:
            dict: Dictionary containing t-statistic and p-value.
        """
        try:
            data1 = self.data.loc[self.group_labels == group1, protein].dropna()
            data2 = self.data.loc[self.group_labels == group2, protein].dropna()
            t_stat, p_val = ttest_ind(data1, data2, equal_var=False)  # Welch's t-test
            return {'t_statistic': t_stat, 'p_value': p_val}
        except Exception as e:
            logging.error(f"Error performing t-test on protein '{protein}': {e}")
            return {'t_statistic': np.nan, 'p_value': np.nan}

    def anova_test(self, protein: str) -> dict:
        """
        Performs one-way ANOVA across multiple groups for a specific protein.

        Args:
            protein (str): Protein column name to test.

        Returns:
            dict: Dictionary containing F-statistic and p-value.
        """
        try:
            groups = self.group_labels.unique()
            group_data = [self.data.loc[self.group_labels == group, protein].dropna() for group in groups]
            if len(group_data) < 2:
                logging.warning(f"Not enough groups to perform ANOVA for protein '{protein}'.")
                return {'F_statistic': np.nan, 'p_value': np.nan}
            f_stat, p_val = f_oneway(*group_data)
            return {'F_statistic': f_stat, 'p_value': p_val}
        except Exception as e:
            logging.error(f"Error performing ANOVA on protein '{protein}': {e}")
            return {'F_statistic': np.nan, 'p_value': np.nan}

    def compare_aggregate_means(self, method: str = 't_test', group1: str = None, group2: str = None) -> pd.DataFrame:
        """
        Compares aggregate means across groups for all proteins using the specified method.

        Args:
            method (str, optional): Comparison method ('t_test', 'anova'). Defaults to 't_test'.
            group1 (str, optional): Label of the first group (required for 't_test').
            group2 (str, optional): Label of the second group (required for 't_test').

        Returns:
            pd.DataFrame: DataFrame containing test statistics and p-values for each protein.
        """
        results = {}
        for protein in self.data.columns:
            if method == 't_test':
                if group1 is None or group2 is None:
                    logging.error("Both 'group1' and 'group2' must be specified for t_test.")
                    return pd.DataFrame()
                test_result = self.t_test(group1, group2, protein)
                results[protein] = test_result
            elif method == 'anova':
                test_result = self.anova_test(protein)
                results[protein] = test_result
            else:
                logging.error(f"Unsupported comparison method '{method}'. Choose 't_test' or 'anova'.")
                return pd.DataFrame()
        results_df = pd.DataFrame(results).T
        return results_df

