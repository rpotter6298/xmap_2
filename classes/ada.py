# ada.py

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DimensionalityReducer:
    """
    A class for performing dimensionality reduction using PCA and t-SNE,
    along with associated visualization methods.
    
    Attributes:
        pca_model (PCA): Fitted PCA model.
        tsne_model (TSNE): Fitted t-SNE model.
    """
    
    def __init__(self, random_state=42):
        """
        Initializes the DimensionalityReducer with a specified random state.
        
        Parameters:
            random_state (int): Seed for reproducibility.
        """
        self.pca_model = None
        self.tsne_model = None
        self.random_state = random_state
    
    # ---------------- PCA Methods ---------------- #
    
    def fit_pca(self, X, n_components=None, variance_threshold=0.95):
        """
        Fits a PCA model to the data.
        
        Parameters:
            X (array-like or pd.DataFrame): Preprocessed feature data.
            n_components (int or float, optional): 
                - If int, number of principal components to keep.
                - If float, percentage of variance to retain.
                - If None, all components are kept.
                Defaults to None.
            variance_threshold (float, optional): 
                If n_components is float, the variance threshold to retain.
                Defaults to 0.95.
        """
        if isinstance(n_components, float):
            self.pca_model = PCA(n_components=variance_threshold, random_state=self.random_state)
        else:
            self.pca_model = PCA(n_components=n_components, random_state=self.random_state)
        
        self.pca_model.fit(X)
        print(f"PCA fitted with {self.pca_model.n_components_} components.")
    
    def transform_pca(self, X):
        """
        Transforms the data using the fitted PCA model.
        
        Parameters:
            X (array-like or pd.DataFrame): Preprocessed feature data.
        
        Returns:
            np.ndarray: PCA-transformed data.
        """
        if not self.pca_model:
            raise ValueError("PCA model has not been fitted yet. Call 'fit_pca' first.")
        return self.pca_model.transform(X)
    
    def plot_pca_scree(self):
        """
        Plots the Scree plot showing the explained variance ratio of each principal component.
        """
        if not self.pca_model:
            raise ValueError("PCA model has not been fitted yet. Call 'fit_pca' first.")
        
        plt.figure(figsize=(10,6))
        plt.plot(range(1, len(self.pca_model.explained_variance_ratio_) + 1),
                 self.pca_model.explained_variance_ratio_, marker='o', linestyle='--')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(range(1, len(self.pca_model.explained_variance_ratio_) + 1))
        plt.grid(True)
        plt.show()
    
    def plot_pca_scatter(self, X_pca, target=None, hue=None, title='PCA Scatter Plot'):
        """
        Creates a scatter plot of the first two principal components.
        
        Parameters:
            X_pca (np.ndarray): PCA-transformed data.
            y (array-like, optional): Target variable for coloring.
            hue (str, optional): Column name for hue if y is a DataFrame.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(10,8))
        if target is not None:
            sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=target, palette='viridis', edgecolor='k', s=100)
        else:
            sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], palette='viridis', edgecolor='k', s=100)
        plt.title(title)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Target' if y is not None else '')
        plt.grid(True)
        plt.show()
    
    # ---------------- t-SNE Methods ---------------- #
    
    def fit_tsne(self, X, n_components=2, perplexity=30, learning_rate=200, n_iter=1000):
        """
        Fits a t-SNE model to the data.
        
        Parameters:
            X (array-like or pd.DataFrame): Preprocessed feature data.
            n_components (int): Dimension of the embedded space. Typically 2 or 3.
                Defaults to 2.
            perplexity (float): Perplexity parameter for t-SNE. 
                Typically between 5 and 50. Defaults to 30.
            learning_rate (float): Learning rate for t-SNE. Defaults to 200.
            n_iter (int): Number of iterations for optimization. Defaults to 1000.
        """
        self.tsne_model = TSNE(n_components=n_components,
                               perplexity=perplexity,
                               learning_rate=learning_rate,
                               n_iter=n_iter,
                               random_state=self.random_state)
        X_tsne = self.tsne_model.fit_transform(X)
        print(f"t-SNE fitted with {n_components} components.")
        return X_tsne
    
    def plot_tsne_scatter(self, X_tsne, y=None, hue=None, title='t-SNE Scatter Plot'):
        """
        Creates a scatter plot of the t-SNE components.
        
        Parameters:
            X_tsne (np.ndarray): t-SNE-transformed data.
            y (array-like, optional): Target variable for coloring.
            hue (str, optional): Column name for hue if y is a DataFrame.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(10,8))
        if y is not None:
            sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette='viridis', edgecolor='k', s=100)
        else:
            sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], palette='viridis', edgecolor='k', s=100)
        plt.title(title)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Target' if y is not None else '')
        plt.grid(True)
        plt.show()
    
    # ---------------- Utility Methods ---------------- #
    
    def save_model(self, model_name='pca_tsne_models.pkl'):
        """
        Saves the fitted PCA and t-SNE models to a file.
        
        Parameters:
            model_name (str): Filename to save the models. Defaults to 'pca_tsne_models.pkl'.
        """
        models = {
            'pca_model': self.pca_model,
            'tsne_model': self.tsne_model
        }
        joblib.dump(models, model_name)
        print(f"Models saved to {model_name}")
    
    def load_model(self, model_name='pca_tsne_models.pkl'):
        """
        Loads the PCA and t-SNE models from a file.
        
        Parameters:
            model_name (str): Filename from which to load the models. Defaults to 'pca_tsne_models.pkl'.
        """
        models = joblib.load(model_name)
        self.pca_model = models.get('pca_model', None)
        self.tsne_model = models.get('tsne_model', None)
        print(f"Models loaded from {model_name}")

class PCARegression:
    """
    A class that integrates PCA for dimensionality reduction with Linear Regression modeling.
    
    Parameters:
        n_components (int or float): Number of principal components to retain.
                                     If int, specifies the number of components.
                                     If float, specifies the variance ratio to retain (e.g., 0.95).
                                     Defaults to 0.95.
        test_size (float): Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int): Random state for reproducibility. Defaults to 42.
        verbose (bool): If True, prints progress messages. Defaults to False.
    """
    
    def __init__(self, n_components=0.95, test_size=0.2, random_state=42, verbose=False):
        self.n_components = n_components
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.regressor = LinearRegression()
        self.pipeline = None
        
        # Data placeholders
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Transformed data
        self.X_train_pca = None
        self.X_test_pca = None
        
        # Model performance
        self.metrics = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the PCA and Linear Regression models on the training data.
        
        Parameters:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series): Continuous target variable.
        """
        if self.verbose:
            print("Starting PCA Regression fitting process...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        if self.verbose:
            print(f"Data split into training and testing sets with test size = {self.test_size}")
        
        # Create a pipeline: Scaling -> PCA -> Linear Regression
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('pca', self.pca),
            ('regressor', self.regressor)
        ])
        
        # Fit the pipeline on training data
        self.pipeline.fit(self.X_train, self.y_train)
        
        if self.verbose:
            print("PCA and Linear Regression models fitted successfully.")
        
        # Transform the data for evaluation
        self.X_train_pca = self.pipeline.named_steps['pca'].transform(
            self.scaler.transform(self.X_train)
        )
        self.X_test_pca = self.pipeline.named_steps['pca'].transform(
            self.scaler.transform(self.X_test)
        )
        
        # Evaluate the model
        self.evaluate()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions using the fitted regression model.
        
        Parameters:
            X (pd.DataFrame): New feature data.
        
        Returns:
            np.ndarray: Predicted values.
        """
        return self.pipeline.predict(X)
    
    def evaluate(self):
        """
        Evaluates the model on the test set and stores the performance metrics.
        """
        predictions = self.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, predictions)
        mae = mean_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        
        self.metrics = {
            'Mean Squared Error': mse,
            'Mean Absolute Error': mae,
            'R² Score': r2
        }
        
        if self.verbose:
            print("Model Evaluation Metrics:")
            for metric, value in self.metrics.items():
                print(f"{metric}: {value:.4f}")
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv=5):
        """
        Performs cross-validation and updates the metrics.
        
        Parameters:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series): Continuous target variable.
            cv (int): Number of cross-validation folds. Defaults to 5.
        """
        mse_scores = -cross_val_score(
            self.pipeline, X, y, cv=cv, scoring='neg_mean_squared_error'
        )
        mae_scores = -cross_val_score(
            self.pipeline, X, y, cv=cv, scoring='neg_mean_absolute_error'
        )
        r2_scores = cross_val_score(
            self.pipeline, X, y, cv=cv, scoring='r2'
        )
        
        self.metrics = {
            'Cross-Validated Mean MSE': mse_scores.mean(),
            'Cross-Validated Mean MAE': mae_scores.mean(),
            'Cross-Validated Mean R² Score': r2_scores.mean()
        }
        
        if self.verbose:
            print("Cross-Validation Metrics:")
            for metric, value in self.metrics.items():
                print(f"{metric}: {value:.4f}")
    
    def plot_actual_vs_predicted(self):
        """
        Plots Actual vs. Predicted values for the test set.
        """
        predictions = self.predict(self.X_test)
        actuals = self.y_test
        
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=actuals, y=predictions, alpha=0.7, edgecolor='k')
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs. Predicted Values')
        plt.grid(True)
        plt.show()
    
    def plot_residuals(self):
        """
        Plots Residuals vs. Predicted values.
        """
        predictions = self.predict(self.X_test)
        actuals = self.y_test
        residuals = actuals - predictions
        
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=predictions, y=residuals, alpha=0.7, edgecolor='k')
        plt.hlines(y=0, xmin=predictions.min(), xmax=predictions.max(), colors='r', linestyles='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs. Predicted Values')
        plt.grid(True)
        plt.show()
    
    def plot_pca_scatter(self, y=None, hue_label='ROP_Percent', title='PCA Scatter Plot'):
        """
        Plots the first two principal components colored by the target variable.
        
        Parameters:
            y (array-like, optional): Target variable for coloring.
            hue_label (str): Label for the hue legend. Defaults to 'ROP_Percent'.
            title (str): Title of the plot. Defaults to 'PCA Scatter Plot'.
        """
        if self.X_test_pca is None:
            raise ValueError("The model has not been fitted yet.")
        
        plt.figure(figsize=(10,8))
        if y is not None:
            sns.scatterplot(x=self.X_test_pca[:,0], y=self.X_test_pca[:,1], 
                            hue=y, palette='viridis', edgecolor='k', s=100)
            plt.legend(title=hue_label)
        else:
            sns.scatterplot(x=self.X_test_pca[:,0], y=self.X_test_pca[:,1], 
                            palette='viridis', edgecolor='k', s=100)
        plt.title(title)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.show()
    
    def get_pca_components(self) -> pd.DataFrame:
        """
        Retrieves the principal components of the test set.
        
        Returns:
            pd.DataFrame: DataFrame containing the principal components.
        """
        if self.X_test_pca is None:
            raise ValueError("The model has not been fitted yet.")
        
        components = self.pca.components_
        feature_names = self.X_train.columns if isinstance(self.X_train, pd.DataFrame) else [f"Feature_{i}" for i in range(self.X_train.shape[1])]
        pc_names = [f'PC{i+1}' for i in range(self.pca.n_components_)]
        
        pca_df = pd.DataFrame(self.X_test_pca, columns=pc_names)
        return pca_df
    
    def save_pipeline(self, filepath='pca_regression_pipeline.pkl'):
        """
        Saves the entire pipeline (Scaler, PCA, Regressor) to a file.
        
        Parameters:
            filepath (str): Path to save the pipeline. Defaults to 'pca_regression_pipeline.pkl'.
        """
        joblib.dump(self.pipeline, filepath)
        if self.verbose:
            print(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath='pca_regression_pipeline.pkl'):
        """
        Loads a saved pipeline from a file.
        
        Parameters:
            filepath (str): Path from which to load the pipeline. Defaults to 'pca_regression_pipeline.pkl'.
        """
        self.pipeline = joblib.load(filepath)
        if self.verbose:
            print(f"Pipeline loaded from {filepath}")
