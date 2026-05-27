import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, cross_validate
from tqdm.auto import tqdm
import pickle


def check_decimals(val, threshold=3):
    """Evaluate the number of decimal places in a numeric value.

    Checks if the string representation of a value has more decimal places than
    a specified threshold. If it does, it returns np.nan to effectively filter
    out excessive precision, otherwise returning the original value.

    Args:
        val (int, float, str): The numeric value or its string representation
          to check.
        threshold (int, optional): The maximum allowed number of decimal
          places. Defaults to 3 as it is the relevant threshold for the
          problem.

    Returns:
        float or original type: The original value, or np.nan if the threshold
        is exceeded.
    """

    # Convert to string and split by the decimal point
    str_val = str(val)
    if "." in str_val:
        decimals = str_val.split(".")[1]
        if len(decimals) > threshold:
            return np.nan
    return val


class DataCleaner(BaseEstimator, TransformerMixin):
    """Evaluate and clean the raw donor dataset features.

    Preprocesses the raw dataset by removing irrelevant tracking columns,
    handling structural placeholders, validating data bounds for categorical
    and rating variables.

    Args:
        categorical_cols_values (dict): A dictionary mapping categorical
          column names to lists of their admissible values.
    """

    def __init__(self, categorical_cols_values: dict):
        self.categorical_cols_values = categorical_cols_values
        self.feature_names_in_ = []

    def fit(self, X, y=None):
        """Fit the transformer on the dataset by mapping input feature headers.

        Args:
            X (pd.DataFrame): The input dataset to fit.
            y (pd.Series, optional): The target labels. Defaults to None.

        Returns:
            self: Returns the instance itself.
        """
        self.feature_names_in = []
        return self

    def transform(self, X):
        """Transform and sanitize the features of a dataset.

        Drops structural identifiers, checks and masks precision outliers,
        applies logic transformations on flags, filters out-of-bound ordinal
        values, and ensures appropriate numeric types.

        Args:
            X (pd.DataFrame): The dataset to clean.

        Returns:
            pd.DataFrame: The processed DataFrame without anomalous entries.
        """
        self.feature_names_in_ = np.array(X.columns)
        # Create a copy of the array as a precaution
        X = X.copy()
        # Drop the CONTROL_NUMBER column, as it's an identifier and, thus
        # should never be included in a model
        X.drop(["CONTROL_NUMBER"], axis=1, inplace=True)
        # Replace '?' values with a missing values
        X = X.replace("?", np.nan)
        # Set SES as a float datatype, since it was a string datatype
        # before as it contained '?' values and it contains missing values
        # which the default pandas integer datatypes don't accept.
        X["SES"] = X["SES"].astype("float")
        # Select numerical columns so that specific transformations can be
        # performed on those columns
        numerical_cols = X.select_dtypes(include=np.number).columns
        # Apply the check_decimals function to the numerical columns tp
        # transform the long decimal incorrect values into missing values
        # and then use the absolute value to flip the sign of any remanining
        # missing values, as from our exploration these appeared to be valid
        # values where the sign was simply flipped
        X[numerical_cols] = X[numerical_cols].map(check_decimals)
        X[numerical_cols] = X[numerical_cols].abs()
        # Transform the 'HOME_OWNER' variable into a binary flag column
        X["HOME_OWNER"] = X["HOME_OWNER"].apply(lambda x: 1 if x == "H" else 0)
        # Change all of the (remaining) values which are greater than 1
        # in RECENT_STAR_STATUS to 1
        X.loc[(X["RECENT_STAR_STATUS"] > 1), "RECENT_STAR_STATUS"] = 1
        # Check the categorical columns and change any unexpected value
        # to a missing value
        for var, values in self.categorical_cols_values.items():
            X.loc[(~X[var].isin(values)), var] = np.nan
        return X

    # Adding this allows set_output to work
    def get_feature_names_out(self, input_features=None):
        """Determine output features remaining after data cleaning.

        Args:
            input_features (list-like, optional): Input features descriptor.
              Defaults to None.

        Returns:
            numpy.ndarray: An array of feature names with 'CONTROL_NUMBER'
            removed.
        """
        return np.array([col for col in self.feature_names_in_ if col !=
                         "CONTROL_NUMBER"])


class CategoricalFeatureSelector(BaseEstimator, TransformerMixin):
    """Select categorical features based on a chi-squared test of independence.

    Evaluates the association between each input categorical feature and a
    target variable using a contingency table and a chi-squared test. Features
    are statistically selected and retained only if their computed p-value
    falls below a designated significance threshold.

    Args:
        p_value (float, optional): The statistical significance threshold
          (alpha level) used to reject the null hypothesis of independence.
          Features with a p-value strictly less than this threshold are
          retained. Defaults to 0.05.
    """

    def __init__(self, p_value: float = 0.05):
        self.p_value = p_value
        self.cols_to_keep_ = []

    def fit(self, X, y):
        """Fit the selector by evaluating feature associations with the target.

        Iterates through each column in the dataset, constructs a cross
        -tabulation contingency table with the target variable, performs a
        chi-squared test,and saves the name of the feature if it shows a
        statistically significant association.

        Args:
            X (pd.DataFrame): The input dataset containing categorical
              features.
            y (pd.Series or array-like): The binary or categorical target
              labels used to compute associations.

        Returns:
            self: Returns the instance itself.
        """
        # Initating the columns to keep to ensure no information
        # is leaked from previous folds
        self.cols_to_keep_ = []

        for var in X.columns:
            # Perform the chi-squared test and keep the variable if the
            # test p-value is lower than the threshold and thus the null
            # hypothesis is rejected indicating an association between the
            # variable and the target.
            if chi2_contingency(pd.crosstab(y, X[var]))[1] < self.p_value:
                self.cols_to_keep_.append(var)
        return self

    def transform(self, X):
        """Slice the input dataset to retain only statistically chosen
        features.

        Args:
            X (pd.DataFrame): The dataset to perform feature reduction on.

        Returns:
            pd.DataFrame: A subset of the input DataFrame consisting only of
              the statistically significant features determined during fitting.
        """
        return X[self.cols_to_keep_]

    def get_feature_names_out(self, input_features=None):
        """Determine output features remaining after feature selection.

        Args:
            input_features (list-like, optional): Input features descriptor.
              Defaults to None.

        Returns:
            numpy.ndarray: An array of strings containing the names of the
              retained features.
        """
        return np.array(self.cols_to_keep_)


class NumericalFeatureSelector(BaseEstimator, TransformerMixin):
    """Select numerical features using an ensemble ensemble-voting strategy.

    Combines three distinct feature screening paradigms to dynamically rate and
    prune redundant or low-value continuous variables:
    1. Spearman rank correlation to eliminate collinearity.
    2. Cross-validated Recursive Feature Elimination (RFECV) using a
       Decision Tree to isolate structural importance.
    3. L1-regularized Logistic Regression (Lasso) to enforce absolute sparsity.

    Features accumulate "removal votes" across all three assessments and are
    dropped only if flagged by a majority (2 or more) of the methods.

    Args:
        corr_threshold (float, optional): The absolute Spearman correlation
          cutoff point. Pairwise variables exceeding this bound will trigger a
          removal vote for the column carrying lower target affinity. Defaults
          to 0.8.
        cv (scikit-learn splitter, optional): Cross-validation generator used
          to safely stratify evaluations across internal selectors. Defaults to
          StratifiedKFold(n_splits=5, shuffle=True, random_state=23).
        random_state (int, optional): Control seed injected into stochastic
          underlying estimators to yield deterministic feature outcomes.
          Defaults to 42.
    """

    def __init__(
        self,
        corr_threshold: float = 0.8,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=23),
        random_state: int = 42,
    ):
        self.corr_threshold = corr_threshold
        self.cv = cv
        self.random_state = random_state
        self.cols_to_keep_ = []

    def fit(self, X, y):
        """Fit the selector by gathering ensemble votes across numeric
        variables.

        Runs Spearman correlation filtering, cross-validated RFE tree splits,
        and L1 logistic regularizations. Features with 2 or more accumulated
        votes for deletion are pruned from the downstream execution list.

        Args:
            X (pd.DataFrame): Input numerical features training matrix.
            y (pd.Series or array-like): Target classification labels.

        Returns:
            self: Returns the instance itself.
        """
        # Initating the columns to keep and the votes
        # to ensure no information is leaked from
        # previous folds
        votes_to_remove = {col: 0 for col in X.columns}

        # Method 1 - Correlation Based feature selection
        # If two features are considered highly correlated then the feature
        # with the lowest correlation with the target is voted to be removed
        # We use spearman correlation since it captures non-linear correlations

        # Getting the absolute value of the correlations, since for this step
        # only the magnitude is relevant and not the sign
        corr_matrix = X.corr(method="spearman").abs()
        # Getting the correlation of features with the target
        target_corr = X.apply(lambda x: x.corr(y, method="spearman")).abs()
        # Filtering the correlation matrix to include only the upper triangle
        # so that each feature is only tested once
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        for col in upper.columns:
            # Finding rows where correlation is above the defined threshold
            high_corr_cols = upper.index[
                upper[col] > self.corr_threshold
                ].tolist()
            for row in high_corr_cols:
                # Voting to remove the feature with lower correlation with the
                # target
                to_drop = col if target_corr[col] < target_corr[row] else row
                # Ensuring that clusters of highly correlated columns don't
                # result in extreme amounts of votes, by essentially capping
                # votes at 1 for this test.
                if votes_to_remove[to_drop] == 0:
                    votes_to_remove[to_drop] += 1

        # Method 2 - Simple Recursive Feature Elimination
        # We use a simple Decision Tree model to avoid drastically increasing
        # the running time To increase robustness we use a crossvalidated RFE,
        # this will test the best number of features using cross validation,
        # leading to a much more robust outcome.

        # Creating and fitting the RFE object
        rfe = RFECV(
            estimator=DecisionTreeClassifier(max_depth=5,
                                             class_weight="balanced",
                                             random_state=self.random_state),
            cv=self.cv,
            scoring="f1")
        rfe.fit(X, y)
        for col, keep in zip(X.columns, rfe.support_):
            # Voting to remove the feature if RFE doesn't consider it worth
            # keeping
            if not keep:
                votes_to_remove[col] += 1

        # Method 3 - Lasso
        # Note: We use LogisticRegression with L1 because TARGET_B is binary
        # This method uses L1 regularization to reduce coeficients of low-value
        # predictors to values close to 0

        # Creating and fitting the RFE object
        lasso = LogisticRegressionCV(
            cv=self.cv,
            penalty="l1",
            solver="liblinear",
            max_iter=1000,
            scoring="f1",
            random_state=self.random_state
        ).fit(X, y)

        for col, coef in zip(X.columns, lasso.coef_[0]):
            # Voting to remove the feature if its Lasso coefficient
            # is below 0.00001
            if abs(coef) < 0.00001:
                votes_to_remove[col] += 1

        # Since we're only running 3 tests, we're going to keep any variable
        # that has 1 or no votes for removal, meaning that any variables with
        # 2 or more votes is excluded
        self.cols_to_keep_ = [
            col
            for col, n_votes_to_remove in votes_to_remove.items()
            if n_votes_to_remove <= 1
        ]
        return self

    def transform(self, X):
        """Slice the input dataset to keep variables with a passing vote score.

        Args:
            X (pd.DataFrame): The dataset matrix to filter.

        Returns:
            pd.DataFrame: Sliced feature matrix holding only non-voted out
              columns.
        """
        return X[self.cols_to_keep_]

    def get_feature_names_out(self, input_features=None):
        """Determine output features remaining after ensemble voting reduction.

        Args:
            input_features (list-like, optional): Input features descriptor.
              Defaults to None.

        Returns:
            numpy.ndarray: An array of retained feature name strings.
        """
        return np.array(self.cols_to_keep_)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip out-of-bounds continuous outliers to specified structural limits.

    Caps extreme anomalies at the higher and lower tails of numeric feature
    distributions. It supports automated calculation via the Interquartile
    Range (IQR) approach, statistical quantile thresholds, or hardcoded
    boundary constraints passed through configuration rules.

    Args:
        rules (dict, optional): Mapping configuration utilized when `method` is
          set to 'percentile' or static rules. The keys correspond to column
          names and values represent inner dictionaries specifying 'lower' and
          'upper' target conditions. Defaults to None.
        method (str, optional): The strategy variant used to identify bounds.
          Options are 'iqr', 'percentile', or custom literal settings.
          Defaults to 'iqr'.
        iqr_multiplier (float, optional): Scaling coefficient applied to the
          Interquartile Range for determining IQR boundaries. Defaults to 1.5.
    """

    def __init__(
        self,
        rules: dict = None,
        method: str = "iqr",
        iqr_multiplier: float = 1.5,
    ):
        self.method = method
        self.rules = rules
        self.learned_limits_ = {}
        self.iqr_multiplier = iqr_multiplier
        self.feature_names_in_ = []

    def fit(self, X, y=None):
        """Fit the transformer by establishing clipping parameters for
        variables.

        Computes the target lower and upper containment values based on the
        selected strategy variant and records them within an internal
        dictionary state.

        Args:
            X (pd.DataFrame): Training matrix containing features to cap.
            y (pd.Series, optional): Target labels. Defaults to None.

        Returns:
            self: Returns the instance itself.
        """
        self.feature_names_in_ = np.array(X.columns)
        # If the Method for cliiping is the Interquartile range method
        # Compute the Quartiles, the IQR and then calculate the limits based
        # on the multiplier defined when instatiating the clipper.
        if self.method.lower() == "iqr":
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            lower_threshold = Q1 - self.iqr_multiplier * IQR
            upper_threshold = Q3 + self.iqr_multiplier * IQR
            self.learned_limits_ = (
                pd.merge(
                    lower_threshold.to_frame("lower"),
                    upper_threshold.to_frame("upper"),
                    left_index=True,
                    right_index=True,
                )
                .to_dict(orient="index")
            )
        # If another method is selected use the rules dictionary to compute
        # the limits
        else:
            for var, rules in self.rules.items():
                lower_rule = rules.get("lower")
                upper_rule = rules.get("upper")
                # If percentile method is elected obtain the value that
                # corresponds to that given percentile.
                if self.method.lower() == "percentile":
                    lower_limit = (
                        X[var].quantile(lower_rule)
                        if lower_rule is not None
                        else None
                    )
                    upper_limit = (
                        X[var].quantile(upper_rule)
                        if upper_rule is not None
                        else None
                    )
                    self.learned_limits_[var] = {
                        "lower": lower_limit,
                        "upper": upper_limit,
                    }
                # If not simply take the rules from the dictionary.
                else:
                    self.learned_limits_[var] = {
                        "lower": lower_rule,
                        "upper": upper_rule,
                    }
        return self

    def transform(self, X):
        """Apply learned upper and lower constraints onto features.

        Args:
            X (pd.DataFrame): Data matrix containing elements to cap.

        Returns:
            pd.DataFrame: A modified dataset copy containing clipped numeric
              values.
        """
        X = X.copy()
        # Apply the limits to the data
        for var, limits in self.learned_limits_.items():
            # Round the limits for integer columns to maintain
            # consistency
            if X[var].dtype == int or X[var].dtype == "Int64":
                lower = np.round(limits.get("lower"))
                upper = np.round(limits.get("upper"))
            else:
                lower = limits.get("lower")
                upper = limits.get("upper")
            X[var] = X[var].clip(lower=lower, upper=upper)
        return X

    # Adding this allows set_output to work
    def get_feature_names_out(self, input_features=None):
        """Determine output features remaining after outlier transformation.

        Returns:
            numpy.ndarray: An array matching the internal input structure.
        """
        return self.feature_names_in_


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer for donor feature engineering.

    This class calculates interaction features for donor metrics while
    ensuring missing source values propagate naturally as NaN.
    """
    def __init__(self):
        self.feature_names_in_ = None
        self.engineered_features_ = None

    def fit(self, X, y=None):
        """
        Saves the input column names and sets up the list of new features.
        """
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.engineered_features_ = [
            'LIFETIME_AVG_GIFT_AMT',
            'LIFETIME_GIFT_AMT_RANGE',
            'AVG_TO_LAST_GIFT_RATIO',
            'DONOR_LIFESPAN_MONTHS',
            'GIFTS_PER_MONTH_LIFESPAN',
            'GIFT_TO_HOUSEHOLD_INCOME_RATIO'
        ]
        return self

    def transform(self, X):
        """
        Computes the new variables. Division operations include a safeguard
        to replace 0 values with NaN to avoid infinity values.
        """
        X = X.copy()
        # Calculating the engineered features
        # We use the np.wheres to avoid mathematical problems infinity
        # erros coming from situations of division-by-zero

        # Average gift amount per transaction
        X["LIFETIME_AVG_GIFT_AMT"] = np.where(
            X["LIFETIME_GIFT_COUNT"] == 0,
            np.nan,
            X["LIFETIME_GIFT_AMOUNT"] / X["LIFETIME_GIFT_COUNT"],
        )

        # Value range between a donor's maximum and minimum gifts
        X["LIFETIME_GIFT_AMT_RANGE"] = (
            X["LIFETIME_MAX_GIFT_AMT"] - X["LIFETIME_MIN_GIFT_AMT"]
        )

        # Ratio of a donor's average gift to their very last gift
        X["AVG_TO_LAST_GIFT_RATIO"] = np.where(
            X["LAST_GIFT_AMT"] == 0,
            np.nan,
            X["LIFETIME_AVG_GIFT_AMT"] / X["LAST_GIFT_AMT"],
        )

        # Total span of donor activity in months
        X["DONOR_LIFESPAN_MONTHS"] = (
            X["MONTHS_SINCE_FIRST_GIFT"] - X["MONTHS_SINCE_LAST_GIFT"]
        )

        # Frequency of gifts relative to the donor's lifespan
        X["GIFTS_PER_MONTH_LIFESPAN"] = np.where(
            X["DONOR_LIFESPAN_MONTHS"] == 0,
            np.nan,
            X["LIFETIME_GIFT_COUNT"] / X["DONOR_LIFESPAN_MONTHS"],
        )

        # Average gift size relative to local neighborhood income
        X["GIFT_TO_HOUSEHOLD_INCOME_RATIO"] = np.where(
            X["MEDIAN_HOUSEHOLD_INCOME"] == 0,
            np.nan,
            X["LIFETIME_AVG_GIFT_AMT"] / X["MEDIAN_HOUSEHOLD_INCOME"],
        )
        return X

    def get_feature_names_out(self, input_features=None):
        """
        Returns the combined list of original and engineered feature names.
        """
        return np.append(self.feature_names_in_, self.engineered_features_)


def run_parameter_search(grid: dict,
                         cv: any,
                         X: any,
                         y: any,
                         model: any,
                         metrics: list,
                         results_file_dir: str = None,
                         model_file_dir: str = None,
                         refit: bool = True,
                         n_jobs: int = None) -> pd.DataFrame:
    """Performs a manual parameter search with cross-validation.

    This allows the inclusion of a TQDM progress bar to track hyperparameter
    tuning and evaluation progress.

    Args:
        grid (dict): Dictionary with parameter names as keys and lists of
            settings to try as values.
        cv (any): Determines the cross-validation splitting strategy (e.g., an
            integer or a scikit-learn CV splitter).
        X (any): The training input samples.
        y (any): The target values.
        model (any): The estimator object to use to fit the data.
        metrics (list): The list of metric functions or names to be used for
            evaluating the model. The first metric is the one used to sort
            results at the end.
        results_file_dir (str, optional): Directory path where the search
            results DataFrame will be saved. Defaults to None.
        model_file_dir (str, optional): Directory path where the best fitted
            model. Defaults to None.
        refit (bool, optional): If True, refits the best estimator using the
            entire dataset. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the parameter combinations,
          mean/std scores for training and validation splits, and execution
          status.
    """
    # Build a ParameterGrid based on the parameters to test
    params = ParameterGrid(grid)
    # Initializing the results and the primary metric, which will
    # be the one used to sort runs at the end.
    results = []
    primary_metric = metrics[0]
    for param in tqdm(params, desc="Tuning Hyperparameters"):
        # Initializing the run record with the full dictionary of parameters
        # as well as unpacking that dictionary to create a column for each
        # parameter as that may be useful.
        run_record = {'params_config': param.copy(),
                      **param}
        # We use the try/except block to ensure that if there's any issue with
        # one parameter combination (certain parameter settings are
        # incompatible) we still test all valid ones
        try:
            # Cloning the input model, applying the paramters
            # and the pandas output mode
            current_model = clone(model)
            current_model.set_params(**param)
            current_model.set_output(transform="pandas")
            # Running the cross-validation and processing
            # all of the relevant metrics (scores and fit-time)
            crossval_results = cross_validate(current_model,
                                              X,
                                              y,
                                              cv=cv,
                                              return_train_score=True,
                                              scoring=metrics,
                                              n_jobs=n_jobs,
                                              error_score='raise')
            run_record['mean_fit_time'] = np.mean(crossval_results['fit_time'])
            for metric in metrics:
                run_record[f'mean_val_{metric}'] = np.mean(
                    crossval_results[f'test_{metric}'])
                run_record[f'std_val_{metric}'] = np.std(
                    crossval_results[f'test_{metric}'])
                run_record[f'mean_train_{metric}'] = np.mean(
                    crossval_results[f'train_{metric}'])
                run_record[f'std_train_{metric}'] = np.std(
                    crossval_results[f'train_{metric}'])
            # Classifying the combination as successfull
            run_record['status'] = 'Success'
        except Exception as e:
            # If the parameter combination fails put missing values in all
            # of the metric columns
            run_record['mean_fit_time'] = np.nan
            for metric in metrics:
                run_record[f'mean_val_{metric}'] = np.nan
                run_record[f'std_val_{metric}'] = np.nan
                run_record[f'mean_train_{metric}'] = np.nan
                run_record[f'std_train_{metric}'] = np.nan
            # Classifying the combination as failed
            run_record['status'] = f'Failed: {str(e)}'
        # Appending the run_record to the results
        results.append(run_record)
        # Creating the dataframe and sorting by the primary metric
        df_result = pd.DataFrame(results).sort_values(
            f'mean_val_{primary_metric}', ascending=False, na_position='last')
        # Checking whether there is a directory to export results
        # and using pickle to export it if there is
        if results_file_dir is not None:
            df_result.to_pickle(results_file_dir)

    # Checking whether the best model is to be refitted on the full data
    # and grabbing the best parameter combination, applying it and
    # fitting the model if it is to be refitted.
    # Only do it if the best score comes from a successful run
    # In other words, if there is at least one succesfull run
    if refit and df_result.iloc[0]['status'] == 'Success':
        best_model = clone(model)
        best_model.set_params(**df_result.iloc[0]['params_config'])
        best_model.set_output(transform="pandas")
        best_model_fitted = best_model.fit(X, y)
        # Checking whether there is a directory to export the model
        # and using pickle to export it if there is
        if model_file_dir is not None:
            with open(model_file_dir, 'wb') as file:
                pickle.dump(best_model_fitted, file)
        return df_result, best_model_fitted
    else:
        return df_result
