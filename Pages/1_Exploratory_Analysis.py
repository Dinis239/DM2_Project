import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from Modeling.utils_modeling import DataCleaner

# Defining the column types for relevant plots and filter types

# Detailed demographic & historical variables
CORE_NUMERICAL = [
    'DONOR_AGE', 'INCOME_GROUP', 'WEALTH_RATING', 'MEDIAN_HOME_VALUE',
    'MEDIAN_HOUSEHOLD_INCOME', 'PER_CAPITA_INCOME', 'PCT_OWNER_OCCUPIED',
    'LAST_GIFT_AMT', 'LIFETIME_GIFT_AMOUNT', 'RECENT_AVG_GIFT_AMT',
    'RECENT_AVG_CARD_GIFT_AMT', 'LIFETIME_MAX_GIFT_AMT',
    'LIFETIME_MIN_GIFT_AMT'
]
# Neighborhood percentage attributes
PCT_NEIGHBORHOOD = [
    'PCT_ATTRIBUTE1', 'PCT_ATTRIBUTE2', 'PCT_ATTRIBUTE3', 'PCT_ATTRIBUTE4'
]
# Promotion and reaction counts
CORE_COUNT = [
    'CHILDREN', 'NUMBER_PROM_12', 'CARD_PROM_12', 'RECENT_RESPONSE_COUNT',
    'RECENT_CARD_RESPONSE_COUNT', 'LIFETIME_PROM', 'LIFETIME_CARD_PROM',
    'LIFETIME_GIFT_COUNT', 'MONTHS_SINCE_LAST_GIFT',
    'MONTHS_SINCE_FIRST_GIFT', 'MONTHS_SINCE_LAST_PROM_RESP'
]
# Ratios and rates
CORE_RATIOS = [
    'RECENT_RESPONSE_PROP', 'RECENT_CARD_RESPONSE_PROP', 'FILE_CARD_GIFT'
]
# Categorical variables (HOME_OWNER remains here for sidebar routing, but
# handles binary values)
CORE_CATEGORICAL = [
    'DONOR_GENDER', 'HOME_OWNER', 'RECENCY_STATUS_96NK',
    'FREQUENCY_STATUS_97NK', 'URBANICITY', 'SES'
]
# Flag indicators
CORE_FLAG = ['PEP_STAR', 'RECENT_STAR_STATUS']
# Consolidate all numeric features for distribution and scatter plotting
ALL_NUMERICAL_VARS = CORE_NUMERICAL + PCT_NEIGHBORHOOD + CORE_COUNT \
    + CORE_RATIOS

# Define the target and ID columns
TARGET_COL = 'TARGET_B'
ID_COL = 'CONTROL_NUMBER'

# Defining the map from original column mapping to display friendly
# equivalents, based strictly on the project guidelines
# Note: Updated 'HOME_OWNER' display name to match binary context
# it inherits from the data cleaner
COLUMN_MAP = {
    'TARGET_B': 'Donation Response Status',
    'CONTROL_NUMBER': 'Unique Donor ID',
    'DONOR_AGE': 'Donor Age',
    'URBANICITY': 'Urbanicity Classification',
    'SES': 'Socioeconomic Demographics Profile',
    'HOME_OWNER': 'Home Owner Status',
    'DONOR_GENDER': 'Donor Gender',
    'INCOME_GROUP': 'Income Group Rating',
    'WEALTH_RATING': 'Wealth Rating Group',
    'MEDIAN_HOME_VALUE': 'Median Home Value ($ in 100s)',
    'MEDIAN_HOUSEHOLD_INCOME': 'Median Household Income ($ in 100s)',
    'PCT_OWNER_OCCUPIED': 'Owner-Occupied Housing Near Donor (%)',
    'PER_CAPITA_INCOME': 'Neighborhood Per Capita Income ($)',
    'PCT_ATTRIBUTE1': 'Neighborhood Active Military Men (%)',
    'PCT_ATTRIBUTE2': 'Neighborhood Veterans Men (%)',
    'PCT_ATTRIBUTE3': 'Neighborhood Vietnam Veterans (%)',
    'PCT_ATTRIBUTE4': 'Neighborhood WW2 Veterans (%)',
    'PEP_STAR': 'PEP Star Donor Status',
    'RECENT_STAR_STATUS': 'Star Status Achieved Last 4 Yrs',
    'RECENCY_STATUS_96NK': 'Recency Status (96NK)',
    'FREQUENCY_STATUS_97NK': 'Frequency Status (97NK)',
    'RECENT_RESPONSE_PROP': 'Overall Solicitation Response Prop.',
    'RECENT_AVG_GIFT_AMT': 'Recent Avg. Gift Amt ($)',
    'RECENT_CARD_RESPONSE_PROP': 'Card Solicitation Response Count',
    'RECENT_AVG_CARD_GIFT_AMT': 'Recent Avg. Card Gift Amt ($)',
    'RECENT_RESPONSE_COUNT': 'Overall Response Count (4 Yrs)',
    'RECENT_CARD_RESPONSE_COUNT': 'Card Response Count (4 Yrs)',
    'MONTHS_SINCE_LAST_PROM_RESP': 'Months Since Last Promotion Resp',
    'LIFETIME_CARD_PROM': 'Lifetime Card Promotions Sent',
    'LIFETIME_PROM': 'Total Lifetime Promotions Sent',
    'LIFETIME_GIFT_AMOUNT': 'Total Lifetime Donation Amt ($)',
    'LIFETIME_GIFT_COUNT': 'Total Lifetime Donation Count',
    'LIFETIME_MAX_GIFT_AMT': 'Maximum Single Donation ($)',
    'LIFETIME_MIN_GIFT_AMT': 'Minimum Single Donation ($)',
    'LAST_GIFT_AMT': 'Most Recent Gift Amount ($)',
    'CARD_PROM_12': 'Card Promotions Sent (Last 12M)',
    'NUMBER_PROM_12': 'Total Promotions Sent (Last 12M)',
    'MONTHS_SINCE_LAST_GIFT': 'Months Since Most Recent Gift',
    'MONTHS_SINCE_FIRST_GIFT': 'Months Since First Gift',
    'FILE_CARD_GIFT': 'Lifetime Avg. Card Donation ($)',
    'CHILDREN': 'Number of Children'
}
# Also creating a reverse mapping between display names and
# original variable names.
DISPLAY_TO_VAR = {col_mapped: og_col for og_col, col_mapped in
                  COLUMN_MAP.items()}


# Creating the function to load the data
def load_and_clean_data(file_path, cleaner):
    """Loads raw data, applies data cleaner pipeline, and forces formatting."""
    try:
        raw_data = pd.read_csv(file_path)
        # Fail-safe just in case the cleaner path doesn't exist
        if cleaner is not None:
            cleaned_df = cleaner.transform(raw_data)
        else:
            st.warning("Data cleaner not found. Proceeding with"
                       "standard mapping.")
            cleaned_df = raw_data.copy()

        # Format metrics lists strictly for safe calculations and
        # visualizations
        for col in CORE_COUNT:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col],
                                                errors='coerce')\
                                                .fillna(0).astype(int)

        for col in CORE_RATIOS + CORE_NUMERICAL:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col],
                                                errors='coerce').fillna(0.0)

        # Map target classes to readable text labels
        if TARGET_COL in cleaned_df.columns:
            target_map = {1: "Donated", 0: "Didn't Donate"}
            cleaned_df['Target Status'] = cleaned_df[TARGET_COL]\
                .map(target_map)

        return cleaned_df
    # Fail-safe for a situation where the file doesn't exist
    except FileNotFoundError as e:
        st.error(f"Error: Missing resource file. Details: {e}")
        st.stop()


# Defining the paths of the data file and the data cleaner pickle file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\
    if '__file__' in locals() else os.getcwd()
FILE_PATH = os.path.join(BASE_DIR, 'Files', 'donors_train.csv')

# Creating a  DataCleaner instance
# Identical to our saved one, but
# trying to read the pickle file in
# this python script was failing
categorical_admissible_values = {
    "DONOR_GENDER": ["M", "F", "U"],
    "PEP_STAR": [0, 1],
    "RECENCY_STATUS_96NK": ["S", "A", "E", "F", "N", "L"],
    "RECENT_STAR_STATUS": [0, 1],
    "SES": np.arange(1, 6),
    "URBANICITY": ["S", "T", "U", "R", "C"],
    "INCOME_GROUP": np.arange(1, 8),
    "WEALTH_RATING": np.arange(0, 10)
}
data_cleaner = DataCleaner(
    categorical_cols_values=categorical_admissible_values)

# Running the data loading function
data_for_analysis = load_and_clean_data(FILE_PATH, data_cleaner)

# if the data is empty stop the dashboard
if data_for_analysis.empty:
    st.stop()

all_cols = data_for_analysis.columns.tolist()
all_numerical_cols = [col for col in all_cols if col in ALL_NUMERICAL_VARS]
DISPLAY_OPTIONS = [COLUMN_MAP[col] for col in all_cols if col in COLUMN_MAP
                   and col != ID_COL and col != TARGET_COL]


# Building the sidebar for filtering
st.sidebar.header("🔍 Data Filtering Engine")


def apply_filters(dataframe):
    df_filtered = dataframe.copy()

    st.sidebar.subheader("Demographics & Flags")
    cat_and_flag_cols = CORE_CATEGORICAL + CORE_FLAG

    # Building the sidebar filters for categorical
    # and flag columns
    for col_name in cat_and_flag_cols:
        if col_name in dataframe.columns:
            display_name = COLUMN_MAP[col_name]

            # Convert the binary HOME_OWNER variable options
            # to better display options
            if col_name == 'HOME_OWNER':
                binary_map = {
                    1: "Homeowner",
                    0: "Non-Homeowner",
                    "Unknown": "Unknown"
                }
                reverse_binary_map = {
                    "Homeowner": 1,
                    "Non-Homeowner": 0,
                    "Unknown": "Unknown"
                }

                unique_values = dataframe[col_name].fillna("Unknown")\
                    .unique().tolist()
                options_display = [binary_map.get(val, str(val)) for val in
                                   unique_values]
                # Creating the sidebar filter
                selected_display = st.sidebar.multiselect(
                    display_name,
                    options=options_display,
                    default=options_display
                )

                # Using the reverse map to filter the selected values
                selected_values = [reverse_binary_map.get(x, x) for x in
                                   selected_display]

                # Applying the filter
                df_filtered = df_filtered[
                    df_filtered[col_name].fillna("Unknown")
                    .isin(selected_values)
                ]

            # Convert the binary PEP_STAR variable options
            # to better display options
            elif col_name == 'PEP_STAR':
                pep_map = {
                    1: "Star Donor",
                    0: "Standard Donor",
                    "Unknown": "Unknown"
                }
                reverse_pep_map = {
                    "Star Donor": 1,
                    "Standard Donor": 0,
                    "Unknown": "Unknown"
                }

                unique_values = dataframe[col_name].fillna("Unknown")\
                    .unique().tolist()
                options_display = [pep_map.get(val, str(val)) for val in
                                   unique_values]

                # Creating the sidebar filter
                selected_display = st.sidebar.multiselect(
                    display_name,
                    options=options_display,
                    default=options_display
                )

                # Using the reverse map to filter the selected values
                selected_values = [reverse_pep_map.get(x, x) for x in
                                   selected_display]

                # Applying the filter
                df_filtered = df_filtered[
                    df_filtered[col_name].fillna("Unknown")
                    .isin(selected_values)
                ]

            # Convert the binary RECENT_STAR_STATUS variable options
            # to better display options
            elif col_name == 'RECENT_STAR_STATUS':
                star_map = {
                    1: "Achieved",
                    0: "Not Achieved",
                    "Unknown": "Unknown"
                }
                reverse_star_map = {
                    "Achieved": 1,
                    "Not Achieved": 0,
                    "Unknown": "Unknown"
                }

                unique_values = dataframe[col_name].fillna("Unknown")\
                    .unique().tolist()
                options_display = [star_map.get(val, str(val)) for val in
                                   unique_values]

                # Creating the sidebar filter
                selected_display = st.sidebar.multiselect(
                    display_name,
                    options=options_display,
                    default=options_display
                )

                # Using the reverse map to filter the selected values
                selected_values = [reverse_star_map.get(x, x) for x in
                                   selected_display]

                # Applying the filter
                df_filtered = df_filtered[
                    df_filtered[col_name].fillna("Unknown")
                    .isin(selected_values)
                ]

            else:
                # Creating standard drop-down menus for the other columns
                # since the variable values are already display friendly
                # and don't require conversion
                unique_values = dataframe[col_name].fillna("Unknown")\
                    .unique().tolist()
                selected_values = st.sidebar.multiselect(
                    display_name,
                    options=unique_values,
                    default=unique_values
                )

                # Applying the filter
                df_filtered = df_filtered[
                    df_filtered[col_name].fillna("Unknown")
                    .isin(selected_values)
                ]

    # Creating the Sliders for Numerical Attributes
    # We select 5 which we believe are more relevant
    st.sidebar.subheader("Numeric Baseline Filters")
    sidebar_num_filters = ['DONOR_AGE', 'INCOME_GROUP', 'WEALTH_RATING',
                           'LAST_GIFT_AMT', 'CHILDREN']

    for col_name in sidebar_num_filters:
        if col_name in dataframe.columns:
            # Grab the column display name
            display_name = COLUMN_MAP[col_name]

            # Get the range of values for filtering
            min_val = float(dataframe[col_name].min(numeric_only=True))
            max_val = float(dataframe[col_name].max(numeric_only=True))

            # Build the slide based on the range using custom steps between
            # values depending on the value
            val_range = st.sidebar.slider(
                display_name,
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                step=1.0 if col_name in CORE_COUNT or col_name in
                ['INCOME_GROUP', 'WEALTH_RATING'] else 5.0,
                format="%d"
            )

            # Apply the filter
            df_filtered = df_filtered[(df_filtered[col_name] >= val_range[0]) &
                                      (df_filtered[col_name] <= val_range[1])]

    return df_filtered


filtered_df = apply_filters(data_for_analysis)

# If the filtered data contains no observations stop the
# dashboard
if filtered_df.empty:
    st.error("No donor profiles match your current sidebar constraint"
             "configurations. Please widen bounds.")
    st.stop()


# Creating the dashboard header and cards
st.title("📊 CSA Donor Behavior Analysis Dashboard")
st.markdown(f"Database Total Universe: **{data_for_analysis.shape[0]}** "
            "baseline unique profiles.")
st.header("Quick Filtered Summary")

# Creating a 4 column wide layout
col1, col2, col3, col4 = st.columns(4)

# In the first column add a card with the number of records
# that are within the enabled filters
with col1:
    st.metric(label="Filtered Records Volume", value=filtered_df.shape[0])
# In the second column add a card with the average age of donors
# that are within the filters
with col2:
    avg_age = filtered_df['DONOR_AGE'].mean()
    st.metric(label="Avg. Donor Age", value=f"{round(avg_age, 1)} Yrs")
# In the third column add a card with the average of the last gift
# of all donors within the filters
with col3:
    avg_gift = filtered_df['LAST_GIFT_AMT'].mean()
    st.metric(label="Avg. Last Gift", value=f"${round(avg_gift, 2)}")
# In the fourth and final column add a card with the percentage of donations
# (TARGET_B) within the filtered donors
with col4:
    if filtered_df.shape[0] > 0:
        pct_donated = (filtered_df['TARGET_B'].sum() /
                       filtered_df.shape[0]) * 100
    else:
        pct_donated = 0.0
    st.metric(label="Donation Rate", value=f"{round(pct_donated, 1)}%")

st.divider()


# Creating the section of feature distributions
# Count plots for categorical features, histograms and boxplotss
# for numerical features
st.header("Feature Distribution Analyzer")


# Creating a selection box with the display names
# of the variables
feature_display_name = st.selectbox(
    "Select a variable to inspect distribution layouts on"
    "current FILTERED observations:",
    options=DISPLAY_OPTIONS
)

# Get the original variable name of the selected feature
feature_to_plot = DISPLAY_TO_VAR[feature_display_name]

# Creating the palette for the target
COLOR_MAP_TARGETS = {"Donated": "#22c55e", "Didn't Donate": "#ef4444"}


# Creating the plots for when the selected feature is numerical
if feature_to_plot in ALL_NUMERICAL_VARS:
    # Creating the selector for choosing between the histogram and the boxplot
    plot_type = st.radio("Select Visualization Matrix Framework:",
                         ('Histogram', 'Boxplot'), key='num_plot_type')

    # If histogram is selected create a histogram with 50 bins
    # containing stacked bars for the target
    if plot_type == 'Histogram':
        fig = px.histogram(
            filtered_df,
            x=feature_to_plot,
            color='Target Status',
            nbins=50,
            title=f'Histogram Distribution: {feature_display_name} by '
                  'Outcome Group',
            labels={feature_to_plot: feature_display_name,
                    'count': 'Record Frequency Count'},
            color_discrete_map=COLOR_MAP_TARGETS,
            barmode='stack'
        )
        fig.update_layout(bargap=0.05)

    # If boxplot is selected create a two boxplots
    # One for target 1 (Donated) and One for taregt 0 (Didn't donate)
    elif plot_type == 'Boxplot':
        fig = px.box(
            filtered_df,
            x='Target Status',
            y=feature_to_plot,
            color='Target Status',
            title=f'Statistical Dispersion Range: {feature_display_name}',
            labels={feature_to_plot: feature_display_name,
                    'Target Status': 'Campaign Performance Target'},
            color_discrete_map=COLOR_MAP_TARGETS
        )


# Creating the countplot for when the selected feature is categorical or a flag
# variable
elif feature_to_plot in CORE_CATEGORICAL or feature_to_plot in CORE_FLAG:
    # Make a temporary copy for readable categorical value mapping
    plot_df = filtered_df.copy()

    # Pre-emptively treat missing structural fields as "Unknown" strings
    # so they map consistently into the grouping breakdown visualizations
    plot_df[feature_to_plot] = plot_df[feature_to_plot].fillna("Unknown")

    # Changing flag variable values to more display friendly variants
    if feature_to_plot == 'HOME_OWNER':
        plot_df['HOME_OWNER'] = plot_df['HOME_OWNER']\
                                .map({1: "Homeowner", 0: "Non-Homeowner"})\
                                .fillna(plot_df['HOME_OWNER'])
    elif feature_to_plot == 'PEP_STAR':
        plot_df['PEP_STAR'] = plot_df['PEP_STAR']\
                            .map({1: "Star Donor", 0: "Standard Donor"})\
                            .fillna(plot_df['PEP_STAR'])
    elif feature_to_plot == 'RECENT_STAR_STATUS':
        plot_df['RECENT_STAR_STATUS'] = plot_df['RECENT_STAR_STATUS']\
                                    .map({1: "Achieved", 0: "Not Achieved"})\
                                    .fillna(plot_df['RECENT_STAR_STATUS'])

    # Grouping by the selected feature and the target to produce
    # bars for both
    df_counts = plot_df.groupby([feature_to_plot, 'Target Status']).size()\
        .reset_index(name='Count')

    # Plotting the countplot with 2 bars for each category
    # (1 for each target level)
    fig = px.bar(
        df_counts,
        x=feature_to_plot,
        y='Count',
        color='Target Status',
        title=f'Categorical Representation Breakdown: {feature_display_name}',
        labels={feature_to_plot: feature_display_name,
                'Count': 'Profiles Count'},
        color_discrete_map=COLOR_MAP_TARGETS,
        barmode='group'
    )

    # Rotating the x-axis labels 45 degrees
    fig.update_xaxes(tickangle=45)

st.plotly_chart(fig, use_container_width=True)
st.divider()


# Creating the section of scatter plots
st.header("Variable Relationship Cross-Explorer")

# Create the options list for selecting columns to plot
# obviously only numerical columns since it's a scatter plot
numerical_display_options = [COLUMN_MAP[col] for col in all_numerical_cols if
                             col in COLUMN_MAP]

# Raising an error if there are not multiple numerical options to select from
if len(numerical_display_options) < 2:
    st.error("Only one numerical feature is present in the data. "
             "A scatter plot can't be produced")
    st.stop()


# Create a side-by-side layout to add the selection boxes for the columns
col_x, col_y = st.columns(2)

# By default it selects age and median household income
with col_x:
    scatter_x_display = st.selectbox("Select **X-axis** Variable:",
                                     options=numerical_display_options,
                                     index=2)

with col_y:
    def_idx = 13 if len(numerical_display_options) > 1 else 0
    scatter_y_display = st.selectbox("Select **Y-axis** Variable:",
                                     options=numerical_display_options,
                                     index=def_idx)

# Convert the x and y variables from display names to original names
scatter_x = DISPLAY_TO_VAR.get(scatter_x_display)
scatter_y = DISPLAY_TO_VAR.get(scatter_y_display)


if scatter_x != scatter_y:
    # If the selected x and y variables are different produce
    # the scatter plot, with points colored by the target
    # and a trendline
    fig_scatter = px.scatter(
        filtered_df,
        x=scatter_x,
        y=scatter_y,
        trendline="ols",
        color='Target Status',
        title=f"Bivariate Correlation Map: {scatter_x_display} vs "
              f"{scatter_y_display}",
        labels={scatter_x: scatter_x_display, scatter_y: scatter_y_display},
        opacity=0.6,
        color_discrete_map=COLOR_MAP_TARGETS
    )

    st.plotly_chart(fig_scatter, use_container_width=True)
# Just in case the user selects the same variable twice
else:
    st.info("Please select 2 different variables.")
