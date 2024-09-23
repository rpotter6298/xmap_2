import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureSelector:
    """
    A class that provides various feature selection methods for data preprocessing.
    """

    def __init__(self, target_column=None):
        """
        Initializes the FeatureSelector.

        Args:
            target_column (str, optional): The name of the target variable. Required for supervised methods.
        """
        self.target_column = target_column
        self.encoder = LabelEncoder()
        self.selected_features = None

    def variance_threshold(
        self, data: pd.DataFrame, threshold: float = 0.0
    ) -> pd.DataFrame:
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
        print(
            f"Removed {len(removed_features)} low-variance features: {removed_features}\n"
        )
        return high_variance_data

    def select_k_best(
        self, data: pd.DataFrame, k: int = 10, score_func="f_classif"
    ) -> pd.DataFrame:
        """
        Selects the top k features based on univariate statistical tests.

        Args:
            data (pd.DataFrame): The input data including the target column.
            k (int, optional): Number of top features to select. Defaults to 10.
            score_func (str, optional): The scoring function ('f_classif' or 'mutual_info_classif'). Defaults to 'f_classif'.

        Returns:
            pd.DataFrame: Data with only the top k features.
        """
        if self.target_column is None:
            raise ValueError(
                "Target column must be specified for supervised feature selection."
            )

        print(f"Applying SelectKBest with k={k} and score_func={score_func}...")
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        # Encode target if it's categorical
        if y.dtype == "object" or y.dtype.name == "category":
            y = self.encoder.fit_transform(y)

        if score_func == "f_classif":
            selector = SelectKBest(score_func=f_classif, k=k)
        elif score_func == "mutual_info_classif":
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            raise ValueError(
                "Unsupported score_func. Use 'f_classif' or 'mutual_info_classif'."
            )

        selector.fit(X, y)
        features = X.columns[selector.get_support()]
        selected_data = X[features]
        self.selected_features = features
        print(f"Selected top {k} features: {list(features)}\n")
        return selected_data

    def recursive_feature_elimination(
        self, data: pd.DataFrame, model=None, n_features_to_select: int = 10
    ) -> pd.DataFrame:
        """
        Recursively eliminates features based on model importance.

        Args:
            data (pd.DataFrame): The input data including the target column.
            model (sklearn estimator, optional): The model to use for feature ranking. Defaults to RandomForestClassifier().
            n_features_to_select (int, optional): Number of features to select. Defaults to 10.

        Returns:
            pd.DataFrame: Data with only the selected features.
        """
        if self.target_column is None:
            raise ValueError(
                "Target column must be specified for supervised feature selection."
            )

        print(
            f"Applying Recursive Feature Elimination with {n_features_to_select} features..."
        )
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        # Encode target if it's categorical
        if y.dtype == "object" or y.dtype.name == "category":
            y = self.encoder.fit_transform(y)

        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        selector = RFE(model, n_features_to_select=n_features_to_select, step=1)
        selector = selector.fit(X, y)
        features = X.columns[selector.support_]
        selected_data = X[features]
        self.selected_features = features
        print(f"Selected features: {list(features)}\n")
        return selected_data

    def feature_importance(
        self, data: pd.DataFrame, model=None, top_n: int = 10, plot: bool = False
    ) -> pd.DataFrame:
        """
        Selects top_n features based on feature importance from a model.

        Args:
            data (pd.DataFrame): The input data including the target column.
            model (sklearn estimator, optional): The model to use for feature importance. Defaults to RandomForestClassifier().
            top_n (int, optional): Number of top features to select. Defaults to 10.
            plot (bool, optional): Whether to plot the feature importances. Defaults to False.

        Returns:
            pd.DataFrame: Data with only the top_n features.
        """
        if self.target_column is None:
            raise ValueError(
                "Target column must be specified for supervised feature selection."
            )

        print(f"Applying Feature Importance selection with top_n={top_n}...")
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        # Encode target if it's categorical
        if y.dtype == "object" or y.dtype.name == "category":
            y = self.encoder.fit_transform(y)

        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X, y)
        importances = model.feature_importances_
        feature_importances = pd.Series(importances, index=X.columns)
        top_features = (
            feature_importances.sort_values(ascending=False).head(top_n).index
        )
        selected_data = X[top_features]
        self.selected_features = top_features
        print(
            f"Selected top {top_n} features based on importance: {list(top_features)}\n"
        )

        if plot:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feature_importances[top_features], y=top_features)
            plt.title(f"Top {top_n} Feature Importances")
            plt.xlabel("Importance Score")
            plt.ylabel("Features")
            plt.tight_layout()
            plt.show()

        return selected_data

    def plot_feature_selection_scores(
        self, scores: pd.Series, title: str = "Feature Scores"
    ):
        """
        Plots feature selection scores.

        Args:
            scores (pd.Series): The scores of features.
            title (str, optional): The title of the plot. Defaults to "Feature Scores".
        """
        plt.figure(figsize=(12, 8))
        sns.barplot(x=scores.values, y=scores.index)
        plt.title(title)
        plt.xlabel("Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()
