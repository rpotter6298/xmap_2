import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Union

class FeatureSelector:
    """
    A class that provides various feature selection methods for data preprocessing.
    """

    def __init__(self, target_column: Optional[str] = None):
        """
        Initializes the FeatureSelector.

        Args:
            target_column (str, optional): The name of the target variable. Required for supervised methods.
        """
        self.target_column = target_column
        self.encoder = LabelEncoder()
        self.selected_features = None
        self.scaler = StandardScaler()

    def variance_threshold(self, data: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
        """
        Removes features with variance below the specified threshold.

        Args:
            data (pd.DataFrame): The input data.
            threshold (float, optional): The variance threshold. Defaults to 0.0.

        Returns:
            pd.DataFrame: Data with low-variance features removed.
        """
        print(f"Applying Variance Threshold with threshold={threshold}...")
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(data)
        features = data.columns[selector.get_support()]
        high_variance_data = data[features]
        removed_features = set(data.columns) - set(features)
        print(f"Removed {len(removed_features)} low-variance features: {removed_features}\n")
        return high_variance_data

    def predict_alpha(self, model_type: str = 'lasso', alphas: Optional[List[float]] = None, cv: int = 5) -> float:
        """
        Determines the optimal alpha value for Lasso or Elastic Net using cross-validation.

        Args:
            model_type (str, optional): Type of model ('lasso' or 'elasticnet'). Defaults to 'lasso'.
            alphas (Optional[List[float]], optional): List of alpha values to try. Defaults to None.
            cv (int, optional): Number of cross-validation folds. Defaults to 5.

        Returns:
            float: The optimal alpha value.
        """
        if self.target_column is None:
            raise ValueError("Target column must be specified for supervised feature selection.")

        X = self.scaler.fit_transform(self.data.drop(columns=[self.target_column]))
        y = self.data[self.target_column]

        # Encode target if it's categorical
        if y.dtype == "object" or y.dtype.name == "category":
            y = self.encoder.fit_transform(y)

        if model_type == 'lasso':
            model = LassoCV(alphas=alphas, cv=cv, random_state=42, max_iter=10000)
        elif model_type == 'elasticnet':
            model = ElasticNetCV(alphas=alphas, cv=cv, l1_ratio=0.5, random_state=42, max_iter=10000)
        else:
            raise ValueError("Unsupported model_type. Use 'lasso' or 'elasticnet'.")

        model.fit(X, y)
        optimal_alpha = model.alpha_
        print(f"Optimal alpha for {model_type}: {optimal_alpha}\n")
        return optimal_alpha

    def select_k_features(self, data: pd.DataFrame, k: int, model_type: str = 'rf') -> pd.DataFrame:
        """
        Selects the top k features based on the specified model.

        Args:
            data (pd.DataFrame): The input data including the target column.
            k (int): Number of top features to select.
            model_type (str, optional): The feature selection model ('rf', 'elasticnet', 'lasso'). Defaults to 'rf'.

        Returns:
            pd.DataFrame: Data with only the selected features.
        """
        if model_type not in ['rf', 'elasticnet', 'lasso']:
            raise ValueError("Unsupported model_type. Use 'rf', 'elasticnet', or 'lasso'.")

        self.data = data.copy()

        # Remove target column
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Encode target if it's categorical
        if y.dtype == "object" or y.dtype.name == "category":
            y = self.encoder.fit_transform(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            importances = model.feature_importances_
            feature_scores = pd.Series(importances, index=X.columns)
        elif model_type == 'lasso':
            alpha = self.predict_alpha(model_type='lasso')
            model = LassoCV(alphas=[alpha], cv=5, random_state=42, max_iter=10000)
            model.fit(X_scaled, y)
            importances = np.abs(model.coef_)
            feature_scores = pd.Series(importances, index=X.columns)
        elif model_type == 'elasticnet':
            alpha = self.predict_alpha(model_type='elasticnet')
            model = ElasticNetCV(alphas=[alpha], l1_ratio=0.5, cv=5, random_state=42, max_iter=10000)
            model.fit(X_scaled, y)
            importances = np.abs(model.coef_)
            feature_scores = pd.Series(importances, index=X.columns)

        # Select top k features
        top_features = feature_scores.sort_values(ascending=False).head(k).index.tolist()
        self.selected_features = top_features
        print(f"Selected top {k} features using {model_type}: {top_features}\n")
        return self.data[top_features]

    def feature_importance(self, data: pd.DataFrame, model_type: str = 'rf') -> pd.Series:
        """
        Returns feature importances based on the specified model.

        Args:
            data (pd.DataFrame): The input data including the target column.
            model_type (str, optional): The feature selection model ('rf', 'elasticnet', 'lasso'). Defaults to 'rf'.

        Returns:
            pd.Series: Feature importances sorted in descending order.
        """
        if model_type not in ['rf', 'elasticnet', 'lasso']:
            raise ValueError("Unsupported model_type. Use 'rf', 'elasticnet', or 'lasso'.")

        self.data = data.copy()

        # Remove target column
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Encode target if it's categorical
        if y.dtype == "object" or y.dtype.name == "category":
            y = self.encoder.fit_transform(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            importances = model.feature_importances_
            feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        elif model_type == 'lasso':
            alpha = self.predict_alpha(model_type='lasso')
            model = LassoCV(alphas=[alpha], cv=5, random_state=42, max_iter=10000)
            model.fit(X_scaled, y)
            importances = np.abs(model.coef_)
            feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        elif model_type == 'elasticnet':
            alpha = self.predict_alpha(model_type='elasticnet')
            model = ElasticNetCV(alphas=[alpha], l1_ratio=0.5, cv=5, random_state=42, max_iter=10000)
            model.fit(X_scaled, y)
            importances = np.abs(model.coef_)
            feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

        self.selected_features = feature_importances[feature_importances > 0].index.tolist()
        print(f"Feature importances using {model_type}:\n{feature_importances}\n")
        return feature_importances

    def plot_feature_selection_scores(self, scores: Union[pd.Series, List[float]], features: Optional[List[str]] = None, title: str = "Feature Importances"):
        """
        Plots feature selection scores.

        Args:
            scores (Union[pd.Series, List[float]]): The scores of features.
            features (Optional[List[str]], optional): List of feature names. Defaults to None.
            title (str, optional): The title of the plot. Defaults to "Feature Importances".
        """
        if isinstance(scores, pd.Series):
            scores = scores.sort_values(ascending=False)
            features = scores.index.tolist()
            values = scores.values
        elif isinstance(scores, list) or isinstance(scores, np.ndarray):
            if features is None:
                raise ValueError("Feature names must be provided if scores are not a pandas Series.")
            values = scores
        else:
            raise ValueError("Scores must be a pandas Series or a list of values.")

        plt.figure(figsize=(12, 8))
        sns.barplot(x=values, y=features)
        plt.title(title)
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()
