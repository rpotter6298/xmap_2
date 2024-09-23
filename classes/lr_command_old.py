import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from tabulate import tabulate
import matplotlib.pyplot as plt

class LinearRegressionResult:
    def __init__(
        self,
        model,
        equation,
        average_mse,
        average_r2,
        train_preds,
        test_preds,
        train_outcomes,
        test_outcomes,
        model_accuracy,
        model_mse,
        model_r2,
        beta,
        ci,
        p_value,
        test_indices,
        poly=None,  # Add polynomial transformer if applicable
        plot_data=None,
    ):
        self.model = model
        self.equation = equation
        self.average_mse = average_mse
        self.model_mse = model_mse
        self.average_r2 = average_r2
        self.model_r2 = model_r2
        self.train_preds = train_preds
        self.test_preds = test_preds
        self.train_outcomes = train_outcomes
        self.test_outcomes = test_outcomes
        self.beta = beta
        self.ci = ci
        self.p_value = p_value
        self.test_indices = test_indices
        self.accuracy = model_accuracy
        self.poly = poly  # Save the polynomial transformer
        self.plot_data = plot_data

    # def summary(self):
    #     print(f"Average MSE: {self.average_mse}")
    #     print(f"Model Coefficients: {self.model.coef_}")
    #     print(f"Model Intercept: {self.model.intercept_}")
    #     print(f"Beta Coefficient: {self.beta} (95% CI: {self.ci})")
    #     print(f"P-value: {self.p_value}")
    #     print(f"R-squared: {self.r_squared}")

class LogisticRegressionResult:
    def __init__(
        self,
        model,
        equation,
        model_auc,
        model_ci,
        validation_auc,
        validation_ci,
        coefficients,
        ci,
        p_value,
        fpr_list=None,  # fpr_list and tpr_list are optional
        tpr_list=None,
        pre_cv_fpr=None,
        pre_cv_tpr=None,
    ):
        self.model = model
        self.equation = equation
        self.model_auc = model_auc
        self.model_ci = model_ci
        self.validation_auc = validation_auc
        self.validation_ci = validation_ci
        self.coefficients = coefficients
        self.ci = ci
        self.p_value = p_value
        self.fpr_list = fpr_list
        self.tpr_list = tpr_list
        self.pre_cv_fpr = pre_cv_fpr
        self.pre_cv_tpr = pre_cv_tpr

    # def summary(self):
    #     print(f"Model Equation: {self.equation}")
    #     print(
    #         f"Model AUC: {self.model_auc}, 95% CI: ({self.model_ci[0]}, {self.model_ci[1]})"
    #     )
    #     print(
    #         f"Validation AUC: {self.validation_auc}, 95% CI: ({self.validation_ci[0]}, {self.validation_ci[1]})"
    #     )
    #     print(f"Model Coefficients (Beta): {self.coefficients}")
    #     print(f"95% CI for Coefficients: {self.ci}")
    #     print(f"P-values for Coefficients: {self.p_value}")

class regression_controller:
    def __init__(
        self,
        data: pd.DataFrame,
        random_state: int = 42,
        split_parameters: dict = {"split_type": "loocv"},
        predictors=None,
        confounders=None,
        outcome=None,
    ):
        self.data = data
        self.random_state = random_state
        self.predictors = [] if predictors is None else predictors
        self.confounders = [] if confounders is None else confounders
        self.outcome = None if outcome is None else outcome
        self.train = []
        self.test = []
        self.split_parameters = split_parameters if split_parameters is not None else {}
    
    def backward_elimination_aic(self, X, y):
        """
        Perform backward elimination by iteratively removing features to minimize AIC.

        Args:
            X: The feature matrix (with a constant column for the intercept).
            y: The target variable.

        Returns:
            The final OLS model after backward elimination using AIC.
        """
        model = sm.OLS(y, X).fit()
        current_aic = model.aic
        while True:
            # Test removing each feature one by one
            aic_values = []
            models = []
            for i in range(1, X.shape[1]):
                X_new = np.delete(X, i, axis=1)
                model_new = sm.OLS(y, X_new).fit()  
                aic_values.append(model_new.aic)
                models.append(model_new)
            min_aic = min(aic_values)
            if min_aic < current_aic:
                current_aic = min_aic
                best_model_index = aic_values.index(min_aic)
                X = np.delete(X, best_model_index + 1, axis=1)
                model = models[best_model_index]    
            else:
                break
        return model

    # def initialize_split(self):
    #     if self.data is None:
    #         raise ValueError("Data must be loaded before initializing the split.")

    #     X = self.data[self.predictors].values
    #     y = self.data[self.outcome].values
    #     if self.split_parameters["split_type"] == "loocv":
    #         from sklearn.model_selection import LeaveOneOut
    #         self.split = LeaveOneOut()
    #         self.train = []
    #         self.test = []
    #         for train_index, test_index in self.split.split(X):
    #             self.train.append(train_index)
    #             self.test.append(test_index)
    #     elif self.split_parameters["split_type"] == "train_test":
    #         from sklearn.model_selection import train_test_split
    #         assert (
    #             "test_size" in self.split_parameters.keys()
    #         ), "test_size parameter must be provided for train_test split"
    #         X_train, X_test, y_train, y_test = train_test_split(
    #             X,
    #             y,
    #             test_size=self.split_parameters["test_size"],
    #             random_state=self.random_state,
    #         )
    #         self.train = [np.array(range(len(y_train)))]
    #         self.test = [np.array(range(len(y_train), len(y_train) + len(y_test)))]
    #     elif self.split_parameters["split_type"] == "kfold":
    #         from sklearn.model_selection import KFold
    #         assert ("n_splits" in self.split_parameters.keys()), "n_splits parameter must be provided for kfold split"
    #         self.split = KFold(
    #             n_splits=self.split_parameters["n_splits"],
    #             random_state=self.random_state,
    #             shuffle=True,
    #         )
    #         self.train = []
    #         self.test = []
    #         for train_index, test_index in self.split.split(X):
    #             self.train.append(train_index)
    #             self.test.append(test_index)

    def assign_predictors(self, num_columns=3, predictors=None):
        if predictors is not None:
            self.predictors = predictors
            print(f"Predictors: {self.predictors}")
            return
        else:
            from tabulate import tabulate

            # Print the column names in a neat table with specified number of columns
            column_names = list(self.data.columns)
            num_rows = len(column_names) // num_columns + (
                len(column_names) % num_columns > 0
            )

            # Create a list of rows, each containing up to num_columns column names
            rows = []
            for row in range(num_rows):
                current_row = []
                for col in range(num_columns):
                    index = row + col * num_rows
                    if index < len(column_names):
                        current_row.append(f"{index + 1}. {column_names[index]}")
                    else:
                        current_row.append("")
                rows.append(current_row)

            # Print the table
            print(tabulate(rows, tablefmt="grid"))
            # Get the user's choice of predictors
            predictor_indices = input(
                "Enter the indices of the predictors (separated by commas): "
            )
            self.predictors = [
                column_names[int(i) - 1] for i in predictor_indices.split(",")
            ]
            print(f"Predictors: {self.predictors}")

    def assign_confounders(self, num_columns=3, confounders=None):
        if confounders is not None:
            self.confounders = confounders
            print(f"Confounders: {self.confounders}")
            return
        else:
            from tabulate import tabulate

            # Print the column names in a neat table with specified number of columns
            column_names = list(self.data.columns)
            num_rows = len(column_names) // num_columns + (
                len(column_names) % num_columns > 0
            )

            # Create a list of rows, each containing up to num_columns column names
            rows = []
            for row in range(num_rows):
                current_row = []
                for col in range(num_columns):
                    index = row + col * num_rows
                    if index < len(column_names):
                        current_row.append(f"{index + 1}. {column_names[index]}")
                    else:
                        current_row.append("")
                rows.append(current_row)

            # Print the table
            print(tabulate(rows, tablefmt="grid"))
            # Get the user's choice of confounders
            confounder_indices = input(
                "Enter the indices of the confounders (separated by commas): "
            )
            self.confounders = [
                column_names[int(i) - 1] for i in confounder_indices.split(",")
            ]
            print(f"Confounders: {self.confounders}")

    def add_predictor(self, predictor):
        self.predictors.append(predictor)
        print(f"Predictors: {self.predictors}")

    def add_confounder(self, confounder):
        self.confounders.append(confounder)
        print(f"Confounders: {self.confounders}")

    def remove_predictor(self, predictor):
        self.predictors.remove(predictor)
        print(f"Predictors: {self.predictors}")

    def remove_confounder(self, confounder):
        self.confounders.remove(confounder)
        print(f"Confounders: {self.confounders}")

    def assign_outcome(self, outcome=None):
        if outcome is not None:
            self.outcome = outcome
            print(f"Outcome: {self.outcome}")
            return
        else:
            from tabulate import tabulate

            # Print the column names in a neat table with specified number of columns
            column_names = list(self.data.columns)
            num_columns = 3
            num_rows = len(column_names) // num_columns + (
                len(column_names) % num_columns > 0
            )

            # Create a list of rows, each containing up to num_columns column names
            rows = []
            for row in range(num_rows):
                current_row = []
                for col in range(num_columns):
                    index = row + col * num_rows
                    if index < len(column_names):
                        current_row.append(f"{index + 1}. {column_names[index]}")
                    else:
                        current_row.append("")
                rows.append(current_row)

            # Print the table
            print(tabulate(rows, tablefmt="grid"))
            # Get the user's choice of outcome
            outcome_index = input("Enter the index of the outcome: ")
            self.outcome = column_names[int(outcome_index) - 1]
            print(f"Outcome: {self.outcome}")

    def view_config(self):
        print(f"Predictors: {self.predictors}")
        print(f"Confounders: {self.confounders}")
        print(f"Outcome: {self.outcome}")

    class ROP:
        def __init__(self, input_trait: str, data: pd.DataFrame, time_limit=None):
            self.data = data
            self.trait = input_trait
            if input_trait is None:
                while True:
                    input_trait = input("Enter the trait to examine: ")
                    # identify columns beginning with the trait
                    trait_columns = [
                        col for col in self.data.columns if col.startswith(input_trait)
                    ]
                    if len(trait_columns) == 0:
                        print(f"No columns found starting with '{input_trait}'")
                    else:
                        break
            else:
                trait_columns = [
                    col for col in self.data.columns if col.startswith(input_trait)
                ]
                if len(trait_columns) == 0:
                    raise ValueError(f"No columns found starting with '{input_trait}'")
            # do the same thing but in one line
            column_numbers = [
                (
                    "".join(filter(str.isdigit, col))
                    if "".join(filter(str.isdigit, col))
                    else 0
                )
                for col in trait_columns
            ]
            self.df = self.data[trait_columns]
            self.df.columns = column_numbers
            if time_limit is not None:
                # only include columns with numbers lower than the time limit
                self.df = self.df[
                    [col for col in self.df.columns if int(col) <= time_limit]
                ]
            self.df_normalized = self.scale_to_time_0(self.df)
            self.df_delta = self.delta_adjust(self.df)
            self.df_normalized_delta = self.delta_adjust(self.df_normalized)

        #            self.delta_mean = self.df_delta.mean(axis=1)

        @staticmethod
        def delta_adjust(df):
            # Compute the difference between each timepoint and the previous timepoint
            df = df.diff(axis=1)
            # Replace the first column with zeroes
            df.iloc[:, 0] = 0
            return df

        @staticmethod
        def scale_to_time_0(df):
            # Divide each value by the value at time 0
            return df.div(df[0], axis=0)

    def get_delta_mean(self, trait, time_limit=None, normalized=False):
        if normalized:
            delta = self.ROP(
                trait, self.data, time_limit=time_limit
            ).df_normalized_delta
        else:
            delta = self.ROP(trait, self.data, time_limit=time_limit).df_delta
        time = max(delta.columns.astype(int)) / 12
        return delta.sum(axis=1) / time

    def get_delta_max(self, trait, time_limit=None, normalized=False):
        if normalized:
            delta = self.ROP(
                trait, self.data, time_limit=time_limit
            ).df_normalized_delta
        else:
            delta = self.ROP(trait, self.data, time_limit=time_limit).df_delta

        return delta.abs().max(axis=1)

    def get_category_vector(self):

        return self.data[self.outcome].values

    def rop_plot(self, trait=None, scale=False, delta=False):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        from scipy.stats import pearsonr

        rop = self.ROP(trait, self.data)
        if scale:
            rop_df = rop.df_normalized
        elif delta:
            rop_df = rop.df_delta
        else:
            rop_df = rop.df
        category_vector = self.get_category_vector()
        unique_values = len(np.unique(category_vector))

        # Correlate the mean rate of progression with the outcome variable
        mean_rop = self.get_delta_mean(trait, normalized=True)
        correlation, pvalue = pearsonr(mean_rop, category_vector)

        # Determine if we should use a discrete or continuous colormap
        if unique_values <= 4:
            colormap = plt.get_cmap("coolwarm", unique_values)
            norm = mcolors.BoundaryNorm(
                boundaries=np.arange(unique_values + 1) - 0.5, ncolors=unique_values
            )
        else:
            colormap = plt.get_cmap("coolwarm")
            norm = mcolors.Normalize(
                vmin=category_vector.min(), vmax=category_vector.max()
            )

        # Plotting each patient's values over time with colors based on category_vector
        plt.close("all")
        fig, ax = plt.subplots(figsize=(10, 6))
        for index, row in rop_df.iterrows():
            ax.plot(
                rop_df.columns,
                row,
                label=f"Patient {index + 1}",
                color=colormap(norm(category_vector[index])),
            )

        # Adjust the title position and add correlation info
        if scale:
            title = f"Patient {trait} Values Scaled to Diagnosis"
        elif delta:
            title = f"Rate of change in {trait} Patient Values"
        else:
            title = f"Patient {trait} Values Over Time"

        ax.set_title(
            f"{title}\nCorrelation with {self.outcome}: {correlation:.2f} (p-value: {pvalue:.3e})",
            pad=20,
        )

        ax.set_xlabel("Time (months)")
        ax.set_ylabel("Values")
        ax.grid(True)

        # Creating a legend for categories
        if unique_values <= 4:
            handles = [
                plt.Line2D([0], [0], color=colormap(norm(i)), lw=2)
                for i in range(unique_values)
            ]
            labels = [f"{self.outcome} category {i}" for i in range(unique_values)]
            ax.legend(
                handles,
                labels,
                title="Categories",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
        else:
            # Create a colorbar for the continuous colormap
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(self.outcome)

        # Show the plot
        # plt.show()
        return fig

    def linear_regression(
        self,
        predictors: pd.DataFrame,
        outcome: pd.Series,
        confounders: pd.DataFrame = None,
        polynomial=False,
        degree=None,
        holdout: float = 0.1,
        scale=True,
        accuracy_threshold=0.1,
    ):
        ## Structuring the data
        ############################
        # Align the indices of predictors and outcome
        common_indices = predictors.index.intersection(outcome.index)
        predictors = predictors.loc[common_indices]
        outcome = outcome.loc[common_indices]

        def apply_scaling(data):
            scaler = StandardScaler()
            return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

        if confounders:
            confounders = confounders.loc[common_indices]
            if scale:
                confounders = apply_scaling(confounders)

        # Scale predictors if scale is True
        if scale:
            predictors = apply_scaling(predictors)

        poly = None
        if polynomial:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            predictors = pd.DataFrame(
                poly.fit_transform(predictors),
                columns=poly.get_feature_names_out(),
            )

        # Remove any rows with missing values
        combined = pd.concat([predictors, confounders, outcome], axis=1).dropna()
        predictors = combined.iloc[:, :-1]
        outcome = combined.iloc[:, -1]

        X = predictors.values
        y = outcome.values

        # Initialize lists to store predictions and outcomes
        all_train_preds = []
        all_test_preds = []
        all_test_outcomes = []
        all_train_outcomes = []
        mse_list = []
        r2_list = []
        test_indices = []

        for train_index, test_index in zip(self.train, self.test):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train_const = sm.add_constant(X_train)
            X_test_const = sm.add_constant(X_test)

            # Fit the model on the training data
            model = sm.OLS(y_train, X_train_const).fit()

            # Predict on the test data
            y_pred = model.predict(X_test_const)

            # Save predictions and actual outcomes
            all_test_preds.extend(y_pred)  # Append predictions to the list
            all_test_outcomes.extend(y_test)  # Append actual test outcomes

            # calculate MSE and R2 and append to the lists
            mse = mean_squared_error(y_test, y_pred)
            mse_list.append(mse)
            ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
            ss_res = np.sum((y_test - y_pred) ** 2)
            r2 = 1 - ss_res / ss_total
            r2_list.append(r2)

        average_mse = np.mean(mse_list)
        average_r2 = np.mean(r2_list)

        ##Build the full model. Here we are using main and holdout as terms for train and test to prevent confusion with the cross-validation loop.
        x_main, x_holdout, y_main, y_holdout = train_test_split(
            X, y, test_size=holdout, random_state=self.random_state
        )
        x_main_const = sm.add_constant(x_main, has_constant="add")
        x_holdout_const = sm.add_constant(x_holdout, has_constant="add")

        #DEBUG
        print(f"Main set shape: {x_main.shape}")
        print(f"Holdout set shape: {x_holdout.shape}")

        final_model = sm.OLS(y_main, x_main_const).fit()
        # Evaluate specifically this model on the holdout set
        y_holdout_pred = final_model.predict(x_holdout_const)

        model_mse = mean_squared_error(y_holdout, y_holdout_pred)
        model_r2 = final_model.rsquared

        # Calculate Predictive Accuracy for the holdout set
        model_accuracy = np.mean(
            np.abs(y_holdout - y_holdout_pred) <= accuracy_threshold
        )

        # Beta Coefficient and 95% CI
        params = final_model.params
        conf = final_model.conf_int()
        beta = params[1:]
        ci_lower = conf[1:, 0]
        ci_upper = conf[1:, 1]
        ci = (ci_lower, ci_upper)
        p_value = final_model.pvalues[1:]

        equation = (
            f"{final_model.params[0]:.3f}"  # The intercept is the first parameter
        )
        for coef, name in zip(final_model.params[1:], predictors.columns):
            equation += f" + {coef:.3f} * {name}"

        return LinearRegressionResult(
            model=final_model,
            equation=equation,
            average_mse=average_mse,
            average_r2=average_r2,
            train_preds=all_train_preds,
            test_preds=all_test_preds,
            train_outcomes=all_train_outcomes,
            test_outcomes=all_test_outcomes,
            model_accuracy=model_accuracy,
            model_mse=model_mse,
            model_r2=model_r2,
            beta=beta,
            ci=ci,
            p_value=p_value,
            test_indices=test_indices,
            poly=poly,  # Return the polynomial transformer
            plot_data={
                "x_test": x_holdout,
                "y_test": y_holdout,
                "y_pred": y_holdout_pred,
            },
        )

    def optimize_polynomial_degree(self, predictors, outcome, max_degree=5):
        best_degree = 1
        best_mse = float("inf")
        best_lr_object = None

        for degree in range(1, max_degree + 1):
            print(f"Evaluating polynomial degree {degree}...")
            lr_object = self.linear_regression(
                predictors, outcome, polynomial=True, degree=degree
            )
            if lr_object.average_mse < best_mse:
                best_mse = lr_object.average_mse
                best_degree = degree
                best_lr_object = lr_object

        print(f"Optimal Polynomial Degree: {best_degree} with MSE: {best_mse}")
        return best_degree, best_lr_object


    def plot_linear_regression(self, predictors, outcome, polynomial=True, degree=None):
        if degree:
            lr_object = self.linear_regression(
                predictors, outcome, polynomial=polynomial, degree=degree
            )
        else:
            degree, lr_object = self.optimize_polynomial_degree(predictors, outcome)

        poly = lr_object.poly if polynomial else None

        if polynomial and poly:
            predictors_poly = pd.DataFrame(
                poly.transform(predictors),
                columns=poly.get_feature_names_out(),
            )
            predicted_values = lr_object.model.predict(predictors_poly.values)

            # Create a range for each predictor
            x_range = {
                col: np.linspace(predictors[col].min(), predictors[col].max(), 100)
                for col in predictors.columns
            }
            x_range_df = pd.DataFrame(x_range)

            # Apply polynomial transformation to the generated range
            x_range_poly = poly.transform(x_range_df)

            # Predict outcomes for the generated polynomial features
            y_range = lr_object.model.predict(x_range_poly)
        else:
            predicted_values = lr_object.model.predict(predictors.values)
            x_range = np.linspace(
                predictors.values.min(), predictors.values.max(), 100
            ).reshape(-1, 1)
            y_range = lr_object.model.predict(x_range)
        plt.close("all")
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(predictors.mean(axis=1), outcome, color="blue", label="True Values")
        ax.scatter(
            predictors.mean(axis=1),
            predicted_values,
            color="green",
            label="Predicted Values",
            marker="x",
        )
        ax.plot(x_range_df.mean(axis=1), y_range, color="red", label="Prediction Line")
        ax.set_xlabel("Composite Predictor")
        ax.set_ylabel("Outcome")
        ax.set_title("True vs Predicted Values with Prediction Line")
        ax.legend()

        summary_text = (
            f"Average MSE: {lr_object.average_mse}\n"
            f"R-squared: {lr_object.r_squared}"
            f"Model Intercept: {lr_object.model.intercept_}\n"
            f"Model Coefficients: {lr_object.model.coef_}\n"
            f"95% CI for Coefficients: {lr_object.ci}\n"
            f"P-value for Coefficients: {lr_object.p_value}\n"
            f"Beta Coefficients: {lr_object.beta}\n"
        )

        return fig, summary_text

    def logistic_regression(
        self,
        predictors: pd.DataFrame,
        outcome: pd.Series,
        scale=False,
    ):
        from sklearn.metrics import roc_auc_score, roc_curve
        from classes import compare_auc_delong_xu
        import scipy

        ## Structuring the data
        ############################
        # Align the indices of predictors and outcome
        common_indices = predictors.index.intersection(outcome.index)
        predictors = predictors.loc[common_indices]
        outcome = outcome.loc[common_indices]

        if scale:
            predictors = StandardScaler().fit_transform(predictors)
            # return to the same structure as before
            predictors = pd.DataFrame(predictors)

        X = predictors.values
        y = outcome.values

        # **AUC Before Cross-Validation** (Using the entire dataset)
        model = sm.Logit(y, sm.add_constant(X)).fit()
        y_pred_prob = model.predict(sm.add_constant(X))
        pre_cv_auc = roc_auc_score(y, y_pred_prob)
        # Calculate the variance and confidence interval for AUC using DeLong's method
        auc, delongcov = compare_auc_delong_xu.delong_roc_variance(y, y_pred_prob)
        auc_std = np.sqrt(delongcov)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - 0.95) / 2)
        model_ci = scipy.stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
        model_ci[model_ci > 1] = 1

        pre_cv_fpr, pre_cv_tpr, _ = roc_curve(
            y, y_pred_prob
        )  # Get FPR and TPR for ROC curve

        # Initialize lists to store predictions and outcomes
        all_test_preds = []
        all_test_outcomes = []

        for train_index, test_index in zip(self.train, self.test):
            # Add constant term to X_train and X_test
            X_train = sm.add_constant(X[train_index], has_constant="add")
            X_test = sm.add_constant(X[test_index], has_constant="add")
            y_train, y_test = y[train_index], y[test_index]

            # Fit the model on the training data
            model = sm.Logit(y_train, X_train).fit()

            # Predict on the test data
            y_pred_prob = model.predict(X_test)

            # Collect predictions and true outcomes for later evaluation
            all_test_preds.extend(y_pred_prob)  # Append predictions
            all_test_outcomes.extend(y_test)  # Append true outcomes

        # Convert lists to arrays
        all_test_outcomes = np.array(all_test_outcomes)
        all_test_preds = np.array(all_test_preds)

        # Calculate AUC, FPR, and TPR
        if (
            len(np.unique(all_test_outcomes)) > 1
        ):  # Ensure there are at least two classes
            validation_auc, validation_delong = (
                compare_auc_delong_xu.delong_roc_variance(
                    all_test_outcomes, all_test_preds
                )
            )
            validation_auc_std = np.sqrt(validation_delong)
            lower_upper_q = np.abs(np.array([0, 1]) - (1 - 0.95) / 2)
            validation_ci = scipy.stats.norm.ppf(
                lower_upper_q, loc=validation_auc, scale=validation_auc_std
            )
            validation_ci[validation_ci > 1] = 1
            fpr, tpr, _ = roc_curve(all_test_outcomes, all_test_preds)
        else:
            auc = None  # AUC cannot be computed
            fpr, tpr = None, None

        # Beta Coefficient and 95% CI
        params = model.params
        conf = model.conf_int()
        beta = params[1:]
        ci_lower = conf[1:, 0]
        ci_upper = conf[1:, 1]
        ci = (ci_lower, ci_upper)
        p_value = model.pvalues[1:]

        # Equation as log-odds representation
        equation = (
            f"log-odds = {model.params[0]:.3f}"  # The intercept is the first parameter
        )
        for coef, name in zip(model.params[1:], predictors.columns):
            equation += f" + {coef:.3f} * {name}"

        return LogisticRegressionResult(
            model=model,
            equation=equation,
            model_auc=pre_cv_auc,
            model_ci=model_ci,
            validation_auc=validation_auc,
            validation_ci=validation_ci,
            coefficients=beta,
            ci=ci,
            p_value=p_value,
            fpr_list=fpr,
            tpr_list=tpr,
            pre_cv_fpr=pre_cv_fpr,
            pre_cv_tpr=pre_cv_tpr,
        )
        

