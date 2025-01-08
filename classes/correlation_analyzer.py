import pandas as pd
import numpy as np
import logging
from scipy.stats import spearmanr, pearsonr, kendalltau
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import permanova, DistanceMatrix
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class CorrelationResults:
    """
    A class to store and manage correlation results, including p-values and adjusted p-values.
    """
    def __init__(self, corr_df: pd.DataFrame, pval_df: pd.DataFrame, adj_pval_df: pd.DataFrame):
        """
        Initializes the CorrelationResults object.

        Args:
            corr_df (pd.DataFrame): DataFrame of correlation coefficients.
            pval_df (pd.DataFrame): DataFrame of p-values.
            adj_pval_df (pd.DataFrame): DataFrame of adjusted p-values after multiple testing correction.
        """
        self.corr = corr_df
        self.pval = pval_df
        self.adj_p = adj_pval_df

    def get_significant_results(self, alpha=0.05):
        """
        Retrieves significant correlations based on adjusted p-values.

        Args:
            alpha (float): Significance level threshold (default is 0.05).

        Returns:
            pd.DataFrame: DataFrame containing significant correlations.
        """
        mask = self.adj_p <= alpha
        significant_corrs = self.corr.where(mask)
        return significant_corrs.dropna(how='all')

class CorrelationAnalyzer:
    """
    A class for analyzing correlations between protein data and patient metrics.
    """
    def __init__(self, protein_data: pd.DataFrame, patient_data: pd.DataFrame):
        """
        Initializes the CorrelationAnalyzer with protein and patient data.

        Args:
            protein_data (pd.DataFrame): DataFrame containing protein expression data.
            patient_data (pd.DataFrame): DataFrame containing patient metrics.
        """
        self.protein_data = protein_data
        self.patient_data = patient_data

    def correlation_continuous(self, continuous_columns: list = None, method: str = 'spearman', correction_method: str = 'fdr_bh', single_dataset: bool = False) -> CorrelationResults:
        """
        Computes correlations using the specified method, and adjusts p-values for multiple testing.
        When single_dataset=True, computes correlations within the protein_data DataFrame.

        Args:
            continuous_columns (list, optional): List of continuous patient metric column names.
                                                Not required if single_dataset=True.
            method (str): Correlation method to use. Options are 'spearman', 'pearson', 'kendall'.
                        Default is 'spearman'.
            correction_method (str): Method for multiple testing correction.
                                    Options include 'bonferroni', 'fdr_bh' (default), etc.
            single_dataset (bool): If True, computes correlations within protein_data.
                                If False, computes correlations between continuous_columns and protein_data.

        Returns:
            CorrelationResults: An object containing DataFrames of correlation coefficients,
                                p-values, and adjusted p-values.
        """
        # Map method names to functions
        methods = {
            'spearman': spearmanr,
            'pearson': pearsonr,
            'kendall': kendalltau
        }

        if method not in methods:
            raise ValueError(f"Method '{method}' is not recognized. Choose from 'spearman', 'pearson', 'kendall'.")

        correlation_func = methods[method]

        if single_dataset:
            # Compute correlations within protein_data
            data = self.protein_data
            variables = data.columns
            n = len(variables)

            # Initialize DataFrames
            corr_df = pd.DataFrame(index=variables, columns=variables, dtype=float)
            pval_df = pd.DataFrame(index=variables, columns=variables, dtype=float)
            adj_pval_df = pd.DataFrame(index=variables, columns=variables, dtype=float)

            # Compute pairwise correlations and p-values
            for i in range(n):
                for j in range(i, n):
                    x = data.iloc[:, i]
                    y = data.iloc[:, j]
                    # Remove NaN values
                    mask = (~pd.isna(x)) & (~pd.isna(y))
                    x_clean = x[mask]
                    y_clean = y[mask]
                    if len(x_clean) > 1 and len(y_clean) > 1:
                        if method == 'kendall':
                            corr_result = correlation_func(x_clean, y_clean)
                            corr, pval = corr_result.correlation, corr_result.pvalue
                        else:
                            corr, pval = correlation_func(x_clean, y_clean)
                    else:
                        corr, pval = np.nan, np.nan
                    # Fill symmetric matrices
                    corr_df.iloc[i, j] = corr_df.iloc[j, i] = corr
                    pval_df.iloc[i, j] = pval_df.iloc[j, i] = pval

            # Multiple testing correction
            pvals = pval_df.values.flatten()
            valid_indices = ~np.isnan(pvals)
            corrected_pvals = np.full_like(pvals, np.nan)
            if valid_indices.any():
                corrected_pvals[valid_indices] = multipletests(pvals[valid_indices], method=correction_method)[1]
            adj_pval_df.values[:] = corrected_pvals.reshape(n, n)

            # Return results
            return CorrelationResults(corr_df, pval_df, adj_pval_df)

        else:
            # Validate continuous_columns
            if continuous_columns is None or not continuous_columns:
                raise ValueError("continuous_columns must be provided when single_dataset=False.")

            # Initialize DataFrames
            corr_df = pd.DataFrame(index=continuous_columns, columns=self.protein_data.columns, dtype=float)
            pval_df = pd.DataFrame(index=continuous_columns, columns=self.protein_data.columns, dtype=float)
            adj_pval_df = pd.DataFrame(index=continuous_columns, columns=self.protein_data.columns, dtype=float)

            for cont_col in continuous_columns:
                for protein in self.protein_data.columns:
                    # Prepare data
                    x = self.patient_data[cont_col]
                    y = self.protein_data[protein]
                    # Remove NaN values
                    mask = (~pd.isna(x)) & (~pd.isna(y))
                    x_clean = x[mask]
                    y_clean = y[mask]

                    if len(x_clean) > 1 and len(y_clean) > 1:
                        # Compute correlation and p-value
                        if method == 'kendall':
                            corr_result = correlation_func(x_clean, y_clean)
                            corr, pval = corr_result.correlation, corr_result.pvalue
                        else:
                            corr, pval = correlation_func(x_clean, y_clean)
                    else:
                        corr, pval = np.nan, np.nan

                    corr_df.loc[cont_col, protein] = corr
                    pval_df.loc[cont_col, protein] = pval

                # Multiple testing correction for this continuous variable across all proteins
                pvals = pval_df.loc[cont_col].astype(float).values
                valid_indices = ~np.isnan(pvals)
                corrected_pvals = np.full_like(pvals, np.nan)

                if valid_indices.any():
                    corrected_pvals[valid_indices] = multipletests(pvals[valid_indices], method=correction_method)[1]
                    adj_pval_df.loc[cont_col] = corrected_pvals
                else:
                    adj_pval_df.loc[cont_col] = np.nan

            # Return results
            return CorrelationResults(corr_df, pval_df, adj_pval_df)
        
    def cca(self, X_df, Y_df, n_components=None, n_permutations=1000, random_state=None):
        """
        Performs Canonical Correlation Analysis between two dataframes and calculates p-values
        using permutation testing.

        Args:
            X_df (pd.DataFrame): The first set of variables (e.g., protein data).
            Y_df (pd.DataFrame): The second set of variables (e.g., clinical data).
            n_components (int, optional): Number of components to keep. Defaults to the minimum of the number of variables in X_df and Y_df.
            n_permutations (int, optional): Number of permutations for the permutation test. Default is 1000.
            random_state (int or RandomState, optional): Seed for reproducibility.

        Returns:
            cca (sklearn.cross_decomposition.CCA): The fitted CCA object with canonical correlations and p-values stored in cca.corr and cca.p_values.
            X_c (np.ndarray): The canonical variates for X_df.
            Y_c (np.ndarray): The canonical variates for Y_df.
        """
        from sklearn.cross_decomposition import CCA
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        import numpy as np

        # Align the dataframes on the index (samples)
        X_df, Y_df = X_df.align(Y_df, join='inner', axis=0)

        # Handle missing values by dropping samples with missing data
        data = pd.concat([X_df, Y_df], axis=1)
        data = data.dropna()
        X_df = data[X_df.columns]
        Y_df = data[Y_df.columns]

        # Standardize the data
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_df)
        Y_scaled = scaler_Y.fit_transform(Y_df)

        n_samples = X_scaled.shape[0]
        p = X_scaled.shape[1]
        q = Y_scaled.shape[1]

        # Determine the number of components
        if n_components is None:
            n_components = min(p, q)

        # Initialize and fit the CCA model
        cca = CCA(n_components=n_components)
        cca.fit(X_scaled, Y_scaled)

        # Transform the data to get the canonical variates
        X_c, Y_c = cca.transform(X_scaled, Y_scaled)

        # Calculate observed canonical correlations
        canonical_correlations = []
        for i in range(n_components):
            corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
            canonical_correlations.append(corr)

        # Permutation testing
        permuted_correlations = np.zeros((n_permutations, n_components))
        rng = np.random.RandomState(random_state)

        for perm in range(n_permutations):
            # Permute Y data
            permuted_indices = rng.permutation(n_samples)
            Y_permuted = Y_scaled[permuted_indices, :]

            # Fit CCA on permuted data
            cca_perm = CCA(n_components=n_components)
            cca_perm.fit(X_scaled, Y_permuted)

            # Transform data
            X_c_perm, Y_c_perm = cca_perm.transform(X_scaled, Y_permuted)

            # Calculate canonical correlations for permuted data
            for i in range(n_components):
                corr_perm = np.corrcoef(X_c_perm[:, i], Y_c_perm[:, i])[0, 1]
                permuted_correlations[perm, i] = corr_perm

        # Calculate p-values
        p_values = []
        for i in range(n_components):
            greater = np.sum(np.abs(permuted_correlations[:, i]) >= np.abs(canonical_correlations[i]))
            p_value = (greater + 1) / (n_permutations + 1)  # Add 1 for continuity correction
            p_values.append(p_value)

        # Store canonical correlations and p-values in the cca object
        cca.corr = canonical_correlations
        cca.p_values = p_values

        # Store results in the instance for later use if desired
        self.cca_model = cca
        self.X_c = X_c
        self.Y_c = Y_c
        self.canonical_correlations = canonical_correlations
        self.canonical_p_values = p_values

        return cca, X_c, Y_c
    
    def cca_continuous_variables(self, continuous_cols, X_df=None, n_components=1, n_permutations=1000, random_state=None):
        """
        Performs CCA between each continuous variable in self.continuous_cols and the protein data,
        and assembles the canonical weights into a DataFrame.

        Args:
            X_df (pd.DataFrame, optional): The first set of variables (e.g., protein data).
                                        Defaults to self.cleaned_proteins.
            n_components (int, optional): Number of components to keep. Defaults to 1.
            n_permutations (int, optional): Number of permutations for p-value calculation. Default is 1000.
            random_state (int or RandomState, optional): Seed for reproducibility.

        Returns:
            weights_df (pd.DataFrame): DataFrame containing canonical weights for proteins.
                                    Columns are continuous variable names, rows are proteins.
            correlations_df (pd.DataFrame): DataFrame containing canonical correlations and p-values.
                                            Index is continuous variable names.
        """
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        import numpy as np

        if X_df is None:
            X_df = self.protein_data

        # Check that continuous_cols is not empty
        if not continuous_cols:
            raise ValueError("No continuous variables provided.")
        # Initialize DataFrames to store results
        weights_df = pd.DataFrame(index=X_df.columns)
        correlations = {}

        # Iterate over continuous variables
        for var in continuous_cols:
            print(var)
            # Prepare Y_df as a DataFrame
            Y_df = self.patient_data[[var]]

            # Align the dataframes on the index (samples)
            X_aligned, Y_aligned = X_df.align(Y_df, join='inner', axis=0)

            # Handle missing values by dropping samples with missing data
            data = pd.concat([X_aligned, Y_aligned], axis=1)
            data = data.dropna()
            X_aligned = data[X_df.columns]
            Y_aligned = data[Y_df.columns]

            # Check if there are enough samples
            if X_aligned.shape[0] < 3:
                print(f"Not enough samples after dropping missing data for variable '{var}'. Skipping.")
                continue

            # Standardize the data
            scaler_X = StandardScaler()
            scaler_Y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_aligned)
            Y_scaled = scaler_Y.fit_transform(Y_aligned)

            # Convert standardized data back to DataFrames
            X_scaled_df = pd.DataFrame(X_scaled, index=X_aligned.index, columns=X_aligned.columns)
            Y_scaled_df = pd.DataFrame(Y_scaled, index=Y_aligned.index, columns=Y_aligned.columns)

            # Perform CCA using the updated cca method with permutation testing
            cca_model, X_c, Y_c = self.cca(
                X_scaled_df,
                Y_scaled_df,
                n_components=n_components,
                n_permutations=n_permutations,
                random_state=random_state
            )

            # Get the canonical correlation and p-value
            corr = cca_model.corr[0]
            p_value = cca_model.p_values[0]
            correlations[var] = {'Canonical Correlation': corr, 'p-value': p_value}

            # Get canonical weights for proteins
            weights = pd.Series(cca_model.x_weights_[:, 0], index=X_df.columns, name=var)

            # Add weights to the DataFrame
            weights_df[var] = weights

        # Convert correlations to DataFrame
        correlations_df = pd.DataFrame.from_dict(correlations, orient='index')

        # Store results in the instance if desired
        self.cca_weights_df = weights_df
        self.canonical_correlations_df = correlations_df

        return weights_df, correlations_df
    
    def run_permanova(self, X_df, Y_vector, distance_metric='euclidean', permutations=999):
        """
        Perform PERMANOVA analysis on the provided data.
        
        Parameters:
        - X_df (pd.DataFrame): DataFrame where rows are samples and columns are features.
        - Y_vector (pd.Series): Grouping variable (categorical) for each sample (must align with X_df rows).
        - distance_metric (str): Distance metric to use (default is 'euclidean').
        - permutations (int): Number of permutations for the PERMANOVA test.
        
        Returns:
        - result (skbio.stats.distance.PERMANOVAResults): Results of the PERMANOVA analysis.
        """
        # Ensure indices align between X_df and Y_vector
        X_df = X_df.loc[Y_vector.index]
        if X_df.shape[0] != len(Y_vector):
            raise ValueError("X_df and Y_vector must have the same number of rows (samples).")
        
        # Check for missing values in Y_vector
        if Y_vector.isnull().any():
            print("Missing values detected in the grouping variable. Removing corresponding samples.")
            valid_indices = Y_vector.dropna().index
            X_df = X_df.loc[valid_indices]
            Y_vector = Y_vector.loc[valid_indices]
        
        # Calculate pairwise distances between rows of X_df
        distance_matrix = pdist(X_df, metric=distance_metric)
        distance_matrix = squareform(distance_matrix)
        
        # Create a labeled DistanceMatrix object
        dm = DistanceMatrix(distance_matrix, ids=X_df.index)
        
        # Perform PERMANOVA
        result = permanova(dm, grouping=Y_vector, permutations=permutations)
        
        return result
    
    def permanova_categorical_variables(self, categorical_cols, X_df=None, distance_metric='euclidean', permutations=999):
        """
        Perform PERMANOVA analysis for each categorical variable in self.categorical_cols against the protein data.
        
        Parameters:
        - categorical_cols (list): List of categorical variable names.
        - X_df (pd.DataFrame, optional): The first set of variables (e.g., protein data).
                                        Defaults to self.cleaned_proteins.
        - distance_metric (str): Distance metric to use (default is 'euclidean').
        - permutations (int): Number of permutations for the PERMANOVA test.
        
        Returns:
        - results (dict): Dictionary containing PERMANOVA results for each categorical variable.
        """
        results = {}
        
        if X_df is None:
            X_df = self.protein_data
        
        for var in categorical_cols:
            print(var)
            Y_vector = self.patient_data[var]
            result = self.run_permanova(X_df, Y_vector, distance_metric=distance_metric, permutations=permutations)
            results[var] = result
        
        return results