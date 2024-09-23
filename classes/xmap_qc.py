# File: classes/xmap_qc.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class xmap_qc:
    """
    A class dedicated to Quality Control (QC) tasks for the xMAP project.
    
    Attributes:
        data (pd.DataFrame): The protein expression data to perform QC on.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the xmap_qc class with the provided protein data.
        
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
        return self.data.var()

    def filter_low_variance_proteins(self, threshold: float = 1e-5) -> pd.DataFrame:
        """
        Filters out proteins with variance below the specified threshold.
        
        Args:
            threshold (float, optional): Variance threshold. Defaults to 1e-5.
        
        Returns:
            pd.DataFrame: DataFrame containing only high-variance proteins.
        """
        protein_variances = self.compute_variance()
        low_variance_proteins = protein_variances[protein_variances < threshold].index.tolist()
        print(f"Removing {len(low_variance_proteins)} low-variance proteins.")
        cleaned_data = self.data.drop(columns=low_variance_proteins)
        print(f"Remaining proteins after filtering: {cleaned_data.shape[1]}")
        return cleaned_data

    def plot_variance(self, top_n: int = 20):
        """
        Plots the top N proteins with the highest variance.
        
        Args:
            top_n (int, optional): Number of top proteins to plot. Defaults to 20.
        """
        protein_variances = self.compute_variance().sort_values(ascending=False)
        top_variances = protein_variances.head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_variances.values, y=top_variances.index, palette="viridis")
        plt.xlabel("Variance")
        plt.ylabel("Proteins")
        plt.title(f"Top {top_n} Proteins by Variance")
        plt.tight_layout()
        plt.show()
