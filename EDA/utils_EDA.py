import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def outlier_count_IQR(data: pd.DataFrame, variables: list,
                      outlier_type: str = 'normal') -> pd.DataFrame:
    """
    Evaluate the outliers of a dataset

    Returns a dataframe including the variable names and the
    correspoding number of outliers, according to the outlier_type
    parameter.

    Parameters:
        ----------
         - data (pd.DataFrame): The DataFrame containing the data.
         - variables (list): The column names of the variables to be evaluated.
         - type (string): The type of outliers to evaluate.
         Defaults to 'normal'.
    Returns:
        ----------
         - outlier_count_df (pd.DataFrame): Dataframe containing outlier counts
         by variable.
    """
    multiplier = 1.5 if outlier_type.lower() == 'normal' else 3
    Q1 = data[variables].quantile(0.25)
    Q3 = data[variables].quantile(0.75)
    IQR = Q3 - Q1
    lower_threshold = Q1 - multiplier * IQR
    upper_threshold = Q3 + multiplier * IQR
    outliers = (data[variables] < lower_threshold) | \
               (data[variables] > (upper_threshold))
    return outliers.sum().to_frame(name='N Outliers')


def outlier_filter_IQR(data: pd.DataFrame, variables: list,
                       outlier_type: str = 'normal',
                       return_dataframe: bool = False) -> None:
    """
    Evaluate the outliers of a dataset

    Prints the percentage of the dataset that is retained
    after excluding outliers and can optionally return a
    filtered DataFrame without outliers.

    Parameters:
        ----------
         - data (pd.DataFrame): The DataFrame containing the data.
         - variables (list): The column names of the variables to be evaluated.
         - type (string): The type of outliers to evaluate.
         Defaults to 'normal'.
         - return_dataframe(boolean): Whether to return the filtered dataset
         without outliers.

    Returns:
        ----------
        None, optionally a filtered Dataframe
    """
    multiplier = 1.5 if outlier_type.lower() == 'normal' else 3
    Q1 = data[variables].quantile(0.25)
    Q3 = data[variables].quantile(0.75)
    IQR = Q3 - Q1
    lower_threshold = Q1 - multiplier * IQR
    upper_threshold = Q3 + multiplier * IQR
    no_outlier_mask = ((data[variables] >= (lower_threshold)) &
                       (data[variables] <= (upper_threshold)))
    data_no_out = data[no_outlier_mask.fillna(True).all(axis=1)]
    perc_kept = round((len(data_no_out)/len(data))*100, 2)
    print(f"Excluding all {outlier_type} outliers, we are left "
          f"with {perc_kept}% of our dataset")
    if return_dataframe:
        return data_no_out
    else:
        return None


def bar_charts_categorical(data: pd.DataFrame, variables: list, target: str):
    """
    Plot side-by-side frequency and proportion stacked bar charts for
    categorical variables against a target.

    Parameters:
        ----------
         - data (pd.DataFrame): The DataFrame containing the data.
         - variables (list): The column names of the categorical variables
         to be plotted.
         - target (str): The column name of the binary target variable used
         for stacking.

    Returns:
        ----------
         None, but a plot is produced for each variable.
    """
    for var in variables:
        cont_tab = pd.crosstab(data[var], data[target],
                               margins=True)
        categories = cont_tab.index[:-1]

        _ = plt.figure(figsize=(15, 5))

        plt.subplot(121)
        p1 = plt.bar(categories, cont_tab.iloc[:-1, 0].values, 0.55,
                     color="gray")
        p2 = plt.bar(categories, cont_tab.iloc[:-1, 1].values, 0.55,
                     bottom=cont_tab.iloc[:-1, 0], color="yellowgreen")
        plt.legend((p2[0], p1[0]), ('$y_i=1$', '$y_i=0$'))
        plt.title("Frequency bar chart")
        plt.xlabel(var)
        plt.ylabel("$Frequency$")

        # auxiliary data for 122
        obs_pct = np.array([np.divide(cont_tab.iloc[:-1, 0].values,
                                      cont_tab.iloc[:-1, 2].values),
                            np.divide(cont_tab.iloc[:-1, 1].values,
                                      cont_tab.iloc[:-1, 2].values)])

        plt.subplot(122)
        p1 = plt.bar(categories, obs_pct[0], 0.55, color="gray")
        p2 = plt.bar(categories, obs_pct[1], 0.55, bottom=obs_pct[0],
                     color="yellowgreen")
        plt.legend((p2[0], p1[0]), ('$y_i=1$', '$y_i=0$'))
        plt.title("Proportion bar chart")
        plt.xlabel(var)
        plt.ylabel("$p$")

        plt.show()


def distribution_plot_grid(data: pd.DataFrame,
                           variables: list,
                           color: str = None,
                           edgecolor: str = 'black') -> None:
    """
    Plot a grid a histogram and a boxplot for each variable based
    on the data.

    Parameters:
        ----------
         - data (pd.DataFrame): The DataFrame containing the data.
         - variables (list): The column names of the variables to be plotted.
         - color (str, optional): Color for the bars. Defaults to None.
         - edgecolor (str, optional): Color for the bars edges.
         Defaults to 'black'.

    Returns:
        ----------
         None, but a plot is produced
    """
    outlier_count = outlier_count_IQR(data, variables, outlier_type='normal')
    for column in variables:
        _ = plt.figure(figsize=(15, 5))
        plt.subplot(121)
        sns.histplot(x=column, data=data, color=color,
                     edgecolor=edgecolor)
        plt.title(f'Column: {column} | Outliers: '
                  f'{outlier_count.loc[column, 'N Outliers']} '
                  'outliers')
        plt.subplot(122)
        sns.boxplot(x=column, data=data, color=color)
        plt.title(f'Column: {column} | Outliers: '
                  f'{outlier_count.loc[column, 'N Outliers']} '
                  'outliers')
        plt.show()


def cor_heatmap(cor: pd.DataFrame) -> None:
    '''
    Plot a correlation heatmap

    Function to plot a correlation heatmap from a dataframe of correlations.

    Arguments:
        ----------
         - cor(pd.DataFrame): DataFrame of correlations between variables

    Returns:
        ----------
         - None, although a heatmap is produced.
    '''
    mask = np.triu(np.ones_like(cor, dtype=bool))
    plt.figure(figsize=(20, 16))
    sns.heatmap(data=cor, annot=True,
                cmap=sns.color_palette("coolwarm", as_cmap=True),
                fmt='.2', mask=mask, vmin=-1, vmax=1)
    plt.show()


def chi2_TestIndependence(data, target, variables, alpha=0.05):
    '''
    This function will follow the steps of chi-square to check if an
    independent variable is an important predictor towards a dependent
    variable. It receives the full dataset, the name of the dependent
    variable and a list of predictors to test as well as the signifcance level
    to be used for the test. It returns a dataframe containing a Keep or
    Discard verdict for each feature

    Parameters:
        ----------
         - data (pd.DataFrame): The DataFrame containing the data.
         - target (str): The name of the dependent variable
         - variables (list): The column names of the variables to be evaluated.
         - alpha (float): The significance level to consider for the chi-square
         test

    Returns:
        ----------
        A dataframe containing the results of the test
    '''
    chi2_check = []
    # Get the X and y datasets for the test
    X_chi = data.drop(target, axis=1)
    y = data[target]
    for var in variables:
        # If p-value < alpha, reject H0 (similarity across groups)
        # and keep feature
        if chi2_contingency(pd.crosstab(y, X_chi[var]))[1] < alpha:
            chi2_check.append('Keep Feature')
        else:
            chi2_check.append('Discard Feature')

    res = pd.DataFrame(data=[variables, chi2_check]).T
    res.columns = ['Column', 'Suggestion']
    return res
