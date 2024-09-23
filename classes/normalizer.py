import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects import pandas2ri

# Activate the automatic conversion between R and Python data structures
rpy2.robjects.numpy2ri.activate()
pandas2ri.activate()


class Normalizer:
    """
    A class that provides various normalization and transformation methods for data preprocessing.
    Each method can be called independently, or multiple methods can be chained together using the 'normalize' method.
    """

    def __init__(self):
        pass

    def log_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies log1p (natural logarithm) transformation to the data.

        Args:
            data (pd.DataFrame): The data to be transformed.

        Returns:
            pd.DataFrame: The log-transformed data.
        """
        print("Applying log1p transformation...")
        transformed_data = np.log1p(data)
        print("Log1p transformation completed.")
        return transformed_data

    def boxcox_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Box-Cox transformation to the data.

        Args:
            data (pd.DataFrame): The data to be transformed.

        Returns:
            pd.DataFrame: The Box-Cox transformed data.
        """
        print("Applying Box-Cox transformation...")
        transformed_data = data.copy()
        for column in transformed_data.columns:
            # Ensure all values are positive
            min_value = transformed_data[column].min()
            if min_value <= 0:
                transformed_data[column] += abs(min_value) + 1e-6
            # Apply Box-Cox transformation
            transformed_data[column], _ = boxcox(transformed_data[column])
        print("Box-Cox transformation completed.")
        return transformed_data

    def rsn_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies RSN normalization using the lumi package in R via rpy2.

        Args:
            data (pd.DataFrame): The data to be RSN normalized.

        Returns:
            pd.DataFrame: The RSN normalized data.
        """
        print("Applying RSN normalization using the lumi package in R via rpy2...")

        # Exclude non-numeric columns if any
        numeric_data = data.select_dtypes(include=[np.number])

        # Convert pandas DataFrame to R matrix
        r_matrix = pandas2ri.py2rpy(numeric_data)

        # Import R functions
        ro.r("suppressMessages(library(lumi))")

        # Assign the data to an R variable
        ro.globalenv["r_matrix"] = r_matrix

        # Create an ExpressionSet object in R
        ro.r('exprs_data <- new("ExpressionSet", exprs = as.matrix(r_matrix))')

        # Apply RSN normalization
        ro.r('rsn_data <- lumiN(exprs_data, method = "rsn")')

        # Get the normalized data back to Python
        rsn_normalized = ro.r("exprs(rsn_data)")
        rsn_normalized_df = pd.DataFrame(
            rsn_normalized, index=data.index, columns=numeric_data.columns
        )

        # If there were non-numeric columns, add them back
        non_numeric_cols = data.select_dtypes(exclude=[np.number])
        if not non_numeric_cols.empty:
            rsn_normalized_df = pd.concat(
                [non_numeric_cols.reset_index(drop=True), rsn_normalized_df], axis=1
            )

        print("RSN normalization completed.")
        return rsn_normalized_df

    def zscore_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Z-score normalization (standardization) to the data.

        Args:
            data (pd.DataFrame): The data to be normalized.

        Returns:
            pd.DataFrame: The normalized data.
        """
        print("Applying Z-score normalization...")
        scaler = StandardScaler()
        transformed_data = pd.DataFrame(
            scaler.fit_transform(data), index=data.index, columns=data.columns
        )
        print("Z-score normalization completed.")
        return transformed_data

    def normalize(
        self,
        data: pd.DataFrame,
        log_transform=False,
        boxcox_transform=False,
        rsn_transform=False,
        zscore_normalize=False,
    ) -> pd.DataFrame:
        """
        Applies a series of transformations to the data based on the provided flags.

        Args:
            data (pd.DataFrame): The data to be transformed.
            log_transform (bool): If True, applies log1p transformation.
            boxcox_transform (bool): If True, applies Box-Cox transformation.
            rsn_transform (bool): If True, applies RSN normalization.
            zscore_normalize (bool): If True, applies Z-score normalization.

        Returns:
            pd.DataFrame: The transformed data.
        """
        transformed_data = data.copy()

        # Apply RSN Normalization
        if rsn_transform:
            transformed_data = self.rsn_transform(transformed_data)

        # Apply Box-Cox Transformation
        if boxcox_transform:
            transformed_data = self.boxcox_transform(transformed_data)

        # Apply Log Transformation
        if log_transform:
            transformed_data = self.log_transform(transformed_data)

        # Apply Z-score Normalization
        if zscore_normalize:
            transformed_data = self.zscore_normalize(transformed_data)

        return transformed_data
