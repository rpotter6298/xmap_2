import pandas as pd
import numpy as np
import logging
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from classes import xmap_qc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ClusterTools:
    def __init__(self, study):
        self.study = study
        self.qc = xmap_qc(study.normalized_data)
        self.cleaned_proteins = self.qc.filter_low_variance_proteins(threshold=1e-5)
        self.patient_data = study.patient_data
        
        self.continuous_cols = [
            col for col in study.patient_data.columns
            if study.patient_data_coldic[col] == "continuous"
        ]
        
        self.categorical_cols = [
            col for col in study.patient_data.columns
            if study.patient_data_coldic[col] in {"categorical", "binary"}
        ]
        # Remove "Medicine_List" from categorical columns
        self.categorical_cols.remove("Medicine_List")

    def build_clusters(self, input_df=None, metric="correlation", method="average"):
        """
        Performs hierarchical clustering on the columns (proteins) of the input DataFrame 
        using correlation distance and assigns clusters based on a distance threshold.

        Generates two dendrograms:
        1. Full names: Used to save column order for alignment with heatmap.
        2. Cleaned names: Used for visualization.

        Args:
            input_df (pd.DataFrame, optional): The input DataFrame. Defaults to self.cleaned_proteins.
            metric (str, optional): Distance metric for clustering. Defaults to 'correlation'.
            method (str, optional): Linkage method for hierarchical clustering. Defaults to 'average'.

        Returns:
            dict: A dictionary where keys are cluster labels and values are lists of protein names.
        """
        if input_df is None:
            input_df = self.cleaned_proteins

        correlation_distance_matrix = pdist(input_df.T, metric=metric)
        correlation_linkage = linkage(correlation_distance_matrix, method=method)
        names = input_df.columns

        # Save column order using full names
        plt.figure(figsize=(16, 10))
        dendro_full = dendrogram(
            correlation_linkage,
            labels=names,  # Full names
            leaf_rotation=90,
            leaf_font_size=14
        )
        self.col_order = dendro_full["ivl"]  # Save column order (full names)
        plt.close()  # Don't display this dendrogram

        # Generate dendrogram with cleaned names for visualization
        clean_names = [name.split("_")[0] for name in names]
        plt.figure(figsize=(16, 10))
        dendrogram(
            correlation_linkage,
            labels=clean_names,  # Cleaned names
            leaf_rotation=90,
            leaf_font_size=14
        )
        plt.title("Hierarchical Clustering of Proteins (Correlation Distance)")
        plt.xlabel("Proteins")
        plt.ylabel("Correlation Distance")
        plt.tight_layout()

        # Save the cleaned dendrogram figure
        self.dendrogram = plt

        # Assign clusters
        color_threshold = 0.7 * max(correlation_distance_matrix)
        cluster_labels = fcluster(correlation_linkage, color_threshold, criterion="distance")

        protein_clusters = pd.DataFrame({"Protein": names, "Cluster": cluster_labels})
        self.clusters_dict = protein_clusters.groupby("Cluster")["Protein"].apply(list).to_dict()

        return self.clusters_dict

    def c3(self, series, labels=['low', 'med', 'high'], method='quantile', bins=None):
        """
        Converts a continuous variable into a categorical variable with three categories.

        Args:
            series (pd.Series): The continuous data to convert.
            labels (list, optional): Labels for the categories. Default is ['low', 'med', 'high'].
            method (str, optional): Method to define bins ('quantile', 'equal', 'custom'). Default is 'quantile'.
            bins (list or array, optional): Custom bin edges if method is 'custom'.

        Returns:
            pd.Series: A categorical Series with the converted values.
        """
        if not isinstance(series, pd.Series):
            raise ValueError("Input 'series' must be a Pandas Series.")
        if method not in ['quantile', 'equal', 'custom']:
            raise ValueError("Method must be one of 'quantile', 'equal', or 'custom'.")
        if method == 'custom' and bins is None:
            raise ValueError("Bins must be provided when method is 'custom'.")

        valid_series = series.dropna()

        if method == 'quantile':
            quantiles = [0, 1/3, 2/3, 1]
            bins = valid_series.quantile(quantiles).values
            bins = np.unique(bins)
            if len(bins) - 1 < len(labels):
                labels = labels[:len(bins) - 1]
        elif method == 'equal':
            min_val = valid_series.min()
            max_val = valid_series.max()
            bins = np.linspace(min_val, max_val, num=4)
        elif method == 'custom':
            bins = np.sort(bins)
            if len(bins) - 1 != len(labels):
                raise ValueError("Number of bins must be equal to number of labels plus one.")

        categorized_series = pd.cut(
            series,
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        return categorized_series
