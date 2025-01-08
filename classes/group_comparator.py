import pandas as pd
import numpy as np
import logging
from scipy.stats import ttest_ind, f_oneway

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

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
        self.group_labels = group_labels.astype(str)

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
