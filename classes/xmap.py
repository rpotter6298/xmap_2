import pandas as pd
from pathlib import Path
from classes.openai_calls import xmap_openai
from classes.xmap_qc import xmap_qc
import json
from classes.group_comparator import GroupComparator
import logging
import seaborn as sns
import matplotlib.pyplot as plt

class xmap_study:
    """
    A class to manage and analyze xMAP study data. This includes loading data,
    performing background adjustments, and aligning patient data with MFI data.

    Attributes:
        study_report_xls (str): Path to the study report Excel file.
        mfi_data (pd.DataFrame): MFI data loaded from the Excel file.
        bead_count (pd.DataFrame): Bead count data loaded from the Excel file.
        antigen_info (pd.DataFrame): Antigen information loaded from the Excel file.
        sample_info (pd.DataFrame): Sample information loaded from the Excel file.
        pool (control): Control instance for the 'pool' tag.
        empty (control): Control instance for the 'empty' tag.
        adj_mfi_data (pd.DataFrame): Background-adjusted MFI data.
        control_cols (pd.DataFrame): Control columns extracted from adjusted MFI data.
    """

    def __init__(self, study_report_xls: str, control_cols: slice = slice(None)):
        """
        Initializes the xmap_study class by loading the data from the provided Excel file,
        and setting up controls and background adjustments.

        Args:
            study_report_xls (str): The path to the study report Excel file.
            control_cols (slice, optional): Slice object representing control columns. Defaults to all columns.
        """
        self.study_report_xls = study_report_xls
        self.mfi_data = self.load_data("MFI", force_numeric=True)  # Load MFI data
        self.bead_count = self.load_data(
            "COUNT", force_numeric=True
        )  # Load bead count data
        self.antigen_info = self.load_data("ANTIGENS")  # Load antigen information
        self.sample_info = self.load_data("SAMPLES")  # Load sample information
        self.pool = self.control(
            "pool", self.mfi_data, control_cols
        )  # Initialize 'pool' control
        self.empty = self.control(
            "empty", self.mfi_data, control_cols
        )  # Initialize 'empty' control
        self.adj_mfi_data = self.background_adjust_data(
            self.mfi_data, self.empty
        )  # Perform background adjustment
        self.openai = xmap_openai(json.load(open(Path("keys", ".keychain")))["openai"]["xmap_2"])
        self.qc = xmap_qc
    def load_data(self, sheet_name, force_numeric=False):
        """
        Loads data from the specified sheet in the study report Excel file.

        Args:
            sheet_name (str): The name of the sheet to load.
            force_numeric (bool, optional): If True, attempts to coerce data to numeric values. Defaults to False.

        Returns:
            pd.DataFrame: The loaded data.
        """
        # Read the Excel sheet and load data
        data = pd.read_excel(self.study_report_xls, sheet_name=sheet_name, index_col=0)

        # If force_numeric is True, attempt to convert data to numeric types
        if force_numeric:
            try:
                data = data.apply(pd.to_numeric, errors="coerce")
            except Exception as e:
                print(f"Unable to convert {sheet_name} data to numeric: {e}")
                raise
        return data

    class control:
        """
        A nested class to represent control data, calculate means and background values
        for 'pool' or 'empty' controls.
        """

        def __init__(
            self, tag: str, data: pd.DataFrame, control_cols: slice = slice(None)
        ):
            """
            Initializes the control instance by filtering the relevant rows using the tag
            and calculating the mean for the control columns.

            Args:
                tag (str): A string to filter relevant control rows (e.g., 'pool', 'empty').
                data (pd.DataFrame): The MFI data.
                control_cols (slice, optional): Slice object to define control columns. Defaults to all columns.
            """
            # Select the data matching the tag
            self.data = data[data.index.str.contains(tag, case=False)]
            self.mean = self.data.mean()  # Calculate the mean for each column
            self.control_cols = control_cols  # Control columns

        def calculate_background(self):
            """
            Calculates the background mean and standard deviation by excluding control columns.

            Returns:
                tuple: Background mean and standard deviation (bg_mean, bg_sd).
            """
            # If control_cols is a slice, slice the control columns
            if isinstance(self.control_cols, slice):
                control_data = self.mean.iloc[self.control_cols]
            else:
                control_data = self.mean

            # Drop control columns to calculate background on the remaining data
            bg_data = self.mean.drop(control_data.index)
            bg_mean = bg_data.mean()  # Calculate the background mean
            bg_sd = bg_data.std()  # Calculate the background standard deviation
            return (bg_mean, bg_sd)

    def background_adjust_data(self, data, control, sd=1):
        """
        Adjusts the MFI data by subtracting the background noise and adding a constant
        to avoid log(0) errors. Also drops rows related to pool and empty controls.

        Args:
            data (pd.DataFrame): The MFI data to be adjusted.
            control (control): The control instance (pool or empty).
            sd (int, optional): The number of standard deviations to add to the background mean. Defaults to 1.

        Returns:
            pd.DataFrame: The background-adjusted data, excluding control columns.
        """
        # Calculate background mean and standard deviation
        bg_mean, bg_sd = control.calculate_background()
        bg_estimate = bg_mean + sd * bg_sd  # Estimate background level
        adj_data = data.sub(bg_estimate)  # Subtract background

        # Set negative values to 0
        adj_data[adj_data <= 0] = 1e-6

        # Add 1 to avoid log(0) errors
        adj_data = adj_data.add(1)

        # Drop pool and empty rows from the adjusted data
        adj_data = adj_data.drop(self.pool.data.index)
        adj_data = adj_data.drop(self.empty.data.index)

        # Extract control columns and store in self.control_cols
        self.control_cols = adj_data.iloc[:, control.control_cols]

        # Remove control columns from the adjusted data
        adj_data = adj_data.drop(self.control_cols.columns, axis=1)

        return adj_data

    def attach_patient_data(self, patient_data: pd.DataFrame):
        """
        Aligns patient data with the adjusted MFI data by matching patient indices.

        Args:
            patient_data (pd.DataFrame): The patient data to align with adjusted MFI data.
        """
        self.patient_data = patient_data

        # Align the index of the adjusted MFI data to match the patient data by stripping letters and converting to integers
        self.adj_mfi_data.index = self.adj_mfi_data.index.str.replace(
            "Sample", ""
        ).astype(int)

        # Find the common patients between MFI and patient data
        common_patients = self.adj_mfi_data.index.intersection(self.patient_data.index)

        # Filter adjusted MFI data to keep only common patients
        self.adj_mfi_data = self.adj_mfi_data.loc[common_patients]

        # for both the patient data and the adjusted mfi data, copy the index to a column called 'Patient ID'
        self.patient_data["Patient_ID"] = self.patient_data.index
        self.adj_mfi_data["Patient_ID"] = self.adj_mfi_data.index

        # reset both indexes
        self.patient_data = self.patient_data.reset_index(drop=True)
        self.adj_mfi_data = self.adj_mfi_data.reset_index(drop=True)

        #reorder the columns in the patient data
        self.patient_data, self.patient_data_coldic = self.openai.order_columns(self.patient_data)

    def binary_t_test(self, binary_cols: list, cleaned_proteins: pd.DataFrame, mapping_dict: dict = None) -> pd.DataFrame:
        """
        Performs t-tests for each binary column against all proteins and compiles the results.

        Args:
            binary_cols (list): List of binary column names in patient_data.
            cleaned_proteins (pd.DataFrame): DataFrame of cleaned protein expression data.
            mapping_dict (dict, optional): Dictionary for custom binary encoding.
                                           Example: {'Gender': {'Male': 0, 'Female': 1}}

        Returns:
            pd.DataFrame: Combined t-test results with each binary column's t-statistic and p-value for all proteins.
        """
        # Initialize an empty DataFrame to store all t-test results
        all_t_test_results = pd.DataFrame()

        # Loop through each binary column
        for col in binary_cols:
            # Get the unique values for the binary column
            unique_vals = self.patient_data[col].dropna().unique()

            # Ensure there are exactly two unique values for a valid t-test
            if len(unique_vals) != 2:
                logging.warning(f"Column '{col}' does not have exactly two unique values. Skipping.")
                continue

            group1, group2 = unique_vals[0], unique_vals[1]

            # Instantiate GroupComparator with the current binary column as group labels
            group_comparator = GroupComparator(cleaned_proteins, self.patient_data[col])

            # Perform t-tests between the two groups
            t_test_results = group_comparator.compare_aggregate_means(method='t_test', group1=group1, group2=group2)

            # Rename columns to include the binary column name and statistic type
            t_test_results = t_test_results.rename(columns={
                't_statistic': f"{col}_t_stat",
                'p_value': f"{col}_p_value"
            })

            # Concatenate the results horizontally
            all_t_test_results = pd.concat([all_t_test_results, t_test_results], axis=1)

        # Optionally, you can transpose the DataFrame for better readability
        # all_t_test_results = all_t_test_results.T

        return all_t_test_results
    
    def binary_t_test_visualization(self, t_test_results: pd.DataFrame, cleaned_proteins: pd.DataFrame, binary_cols: list, p_value_threshold: float = 0.05):
        """
        Visualizes proteins with significant t-test results for each binary attribute using dual box plots.
        
        Args:
            t_test_results (pd.DataFrame): DataFrame containing t-statistics and p-values for binary columns.
            cleaned_proteins (pd.DataFrame): DataFrame containing protein expression data (samples x proteins).
            binary_cols (list): List of binary column names in patient_data to visualize.
            p_value_threshold (float, optional): Threshold for p-values to consider significance. Defaults to 0.05.
        
        Returns:
            None
        """
        # Configure logging within the method if not already configured
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        # Initialize an empty DataFrame to store all t-test results
        all_t_test_results = pd.DataFrame()
        
        # Loop through each binary column
        for col in binary_cols:
            # Get the unique values for the binary column
            unique_vals = self.patient_data[col].dropna().unique()
    
            # Ensure there are exactly two unique values for a valid t-test
            if len(unique_vals) != 2:
                logging.warning(f"Column '{col}' does not have exactly two unique values. Skipping.")
                continue
    
            group1, group2 = unique_vals[0], unique_vals[1]
    
            # Construct the p-value column name
            p_value_col = f"{col}_p_value"
    
            # Check if the p-value column exists in t_test_results
            if p_value_col not in t_test_results.columns:
                logging.warning(f"p-value column '{p_value_col}' not found in t_test_results. Skipping.")
                continue
    
            # Filter proteins with p_value < threshold
            significant_proteins = t_test_results[t_test_results[p_value_col] < p_value_threshold].index.tolist()
    
            if not significant_proteins:
                logging.info(f"No significant proteins found for binary column '{col}' with p-value < {p_value_threshold}.")
                continue
    
            logging.info(f"Found {len(significant_proteins)} significant proteins for binary column '{col}'.")
    
            # Prepare data for plotting
            # Select expression data for the significant proteins
            expr_data = cleaned_proteins[significant_proteins].copy()
    
            # Add group labels
            expr_data['Group'] = self.patient_data[col].values  # Assumes alignment between cleaned_proteins and patient_data
    
            # Melt the DataFrame for seaborn
            expr_melted = expr_data.melt(id_vars='Group', var_name='Protein', value_name='Expression')
    
            # Define palette: map group1 to red, group2 to blue
            palette = {group1: 'red', group2: 'blue'}
    
            # Create box plot
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='Protein', y='Expression', hue='Group', data=expr_melted, palette=palette)
            plt.title(f"Expression Levels by {col} for Significant Proteins (p < {p_value_threshold})", fontsize=16)
            plt.xlabel("Protein", fontsize=14)
            plt.ylabel("Expression Level", fontsize=14)
            plt.legend(title=col, loc='best')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()

