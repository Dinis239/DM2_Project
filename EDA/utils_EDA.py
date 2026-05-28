import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def outlier_count_IQR(data: pd.DataFrame,
                      variables: list,
                      outlier_type: str = 'normal') -> pd.DataFrame:
    """Evaluate the outliers of a dataset.

    Returns a dataframe including the variable names and the corresponding
    number of outliers, according to the `outlier_type` parameter (normal
    or extreme outliers).

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        variables (list of str): The column names of the variables to be
            evaluated.
        outlier_type (str, optional): The type of outliers to evaluate.
            Can be 'normal' or 'extreme'. Defaults to 'normal'.

    Returns:
        pd.DataFrame: A DataFrame containing outlier counts by variable,
            typically with columns for the variable name and its respective
            outlier count.
    """
    # Set the multiplier to be used for calculating outlier thresholds
    # based on the outlier_type parameter
    multiplier = 1.5 if outlier_type.lower() == 'normal' else 3
    # Obtaining Q1, Q3 and IQR values for all variables
    Q1 = data[variables].quantile(0.25)
    Q3 = data[variables].quantile(0.75)
    IQR = Q3 - Q1
    # Calculating the threshold for outlier identification
    lower_threshold = Q1 - multiplier * IQR
    upper_threshold = Q3 + multiplier * IQR
    # Obtaining and returning the dataframe of outlier counts
    # by filtering using the thresholds
    outliers = (data[variables] < lower_threshold) | \
               (data[variables] > (upper_threshold))
    return outliers.sum().to_frame(name='N Outliers')


def outlier_filter_IQR(data: pd.DataFrame, variables: list,
                       outlier_type: str = 'normal',
                       return_dataframe: bool = False) -> None:
    """Evaluate the outliers of a dataset.

    Prints the percentage of the dataset that is retained after excluding
    outliers and can optionally return a filtered DataFrame without outliers.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        variables (list of str): The column names of the variables to be
            evaluated.
        outlier_type (str, optional): The type of outliers to evaluate.
            Defaults to 'normal'.
        return_dataframe (bool, optional): Whether to return the filtered
            dataset without outliers. Defaults to False.

    Returns:
        pd.DataFrame or None: The filtered DataFrame without outliers if
            return_dataframe is True, otherwise None.
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


def bar_charts_categorical(data: pd.DataFrame,
                           variables: list,
                           target: str) -> None:
    """Plot side-by-side frequency and proportion stacked bar charts.

    Plots side-by-side frequency and proportion stacked bar charts for
    categorical variables against a binary target variable.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        variables (list of str): The column names of the categorical variables
            to be plotted.
        target (str): The column name of the binary target variable used
            for stacking.

    Returns:
        None: This function does not return a value, it directly displays
            the generated plots.
    """
    for var in variables:
        cont_tab = pd.crosstab(data[var], data[target],
                               margins=True, dropna=True)
        categories = cont_tab.index[:-1]

        _ = plt.figure(figsize=(15, 5))

        plt.subplot(121)
        p1 = plt.bar(categories, cont_tab.iloc[:-1, 0].values, 0.55,
                     color="gray")
        p2 = plt.bar(categories, cont_tab.iloc[:-1, 1].values, 0.55,
                     bottom=cont_tab.iloc[:-1, 0], color="yellowgreen")
        plt.xticks(rotation=45)
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
        plt.xticks(rotation=45)
        plt.legend((p2[0], p1[0]), ('$y_i=1$', '$y_i=0$'))
        plt.title("Proportion bar chart")
        plt.xlabel(var)
        plt.ylabel("$p$")

        plt.show()


def distribution_plot_grid(data: pd.DataFrame,
                           variables: list,
                           color: str = None,
                           edgecolor: str = 'black',
                           target: str = None,
                           target_hue_order: list = None,
                           target_palette: dict = None) -> None:
    """Plot a grid containing a histogram and a boxplot for each variable.

    Generates and displays a grid layout of histograms and boxplots for the
    specified variables.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        variables (list of str): The column names of the variables to be
            plotted.
        color (str, optional): Color for the bars. Defaults to None.
        edgecolor (str, optional): Color for the bar edges.
            Defaults to 'black'.
        target (str, optional): The name of the target variable to be used for
            plotting different bar segments in a classification analysis.
            Defaults to None.
        target_hue_order (list, optional): The order in which the bars for
            the segments representing each level of the target should appear.
            Defaults to None.
        target_palette (dict, optional): The palette with the colors to be
            plotted for each level of the target. Defaults to None.

    Returns:
        None: This function does not return a value, it directly renders
            the plot grid.
    """
    outlier_count = outlier_count_IQR(data, variables, outlier_type='normal')
    for column in variables:
        _ = plt.figure(figsize=(15, 5))
        plt.subplot(121)
        sns.histplot(x=column,
                     data=data,
                     color=color,
                     edgecolor=edgecolor,
                     hue=target,
                     multiple='stack',
                     hue_order=target_hue_order,
                     palette=target_palette)
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
    """Plot a correlation heatmap from a dataframe of correlations.

    Generates and displays a heatmap visualization representing the correlation
    coefficients between variables provided in the input DataFrame.

    Args:
        cor (pd.DataFrame): DataFrame of correlations between variables.

    Returns:
        None: This function does not return a value, it directly renders
        the heatmap plot.
    """
    mask = np.triu(np.ones_like(cor, dtype=bool))
    plt.figure(figsize=(20, 16))
    sns.heatmap(data=cor, annot=True,
                cmap=sns.color_palette("coolwarm", as_cmap=True),
                fmt='.2', mask=mask, vmin=-1, vmax=1)
    plt.show()
