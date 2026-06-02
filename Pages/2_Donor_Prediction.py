import sys
import os

# Get the absolute path of the directory 2 levels up (out of Pages/ into DM2_PROJECT)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Append it to the system path if it isn't already there
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Import your module using the directory structure path
from Modeling.utils_modeling import (OutlierClipper, CategoricalFeatureSelector, NumericalFeatureSelector, FeatureEngineer, DataCleaner)

# ─── THE CRITICAL FIX FOR PICKLE ──────────────────────────────────────
# Explicitly alias 'utils_modeling' in sys.modules so pickle can find it
import Modeling.utils_modeling
sys.modules['utils_modeling'] = Modeling.utils_modeling
# ──────────────────────────────────────────────────────────────────────

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================================
# 1. GLOBAL DATA DICTIONARY & MAPS (MATCHES MAIN EXPLORER ENGINE)
# =====================================================================
COLUMN_MAP = {
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

# =====================================================================
# 2. DYNAMIC PIPELINE SELECTION & FILE INTERCEPT ROUTINES
# =====================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else os.getcwd()
PIPELINES_DIR = os.path.join(BASE_DIR, 'Files', 'Pickle Files', 'Pipelines')

# 1. Scan directory for available .pkl models
available_pipelines = []
if os.path.exists(PIPELINES_DIR):
    available_pipelines = [f for f in os.listdir(PIPELINES_DIR) if f.endswith('.pkl')]
    available_pipelines.sort()

# 2. Add Sidebar selector for the pipelines
st.sidebar.header("⚙️ Model Configuration")
if available_pipelines:
    # Try to set 'KNN_GridSearch_Best_Pipeline.pkl' as default if it exists
    default_index = available_pipelines.index('KNN_GridSearch_Best_Pipeline.pkl') if 'KNN_GridSearch_Best_Pipeline.pkl' in available_pipelines else 0
    
    selected_pipeline_file = st.sidebar.selectbox(
        "Select Active Pipeline Model",
        options=available_pipelines,
        index=default_index
    )
    MODEL_PATH = os.path.join(PIPELINES_DIR, selected_pipeline_file)
else:
    st.sidebar.warning("No `.pkl` files found in the Pipelines directory.")
    MODEL_PATH = None


def load_serialized_artifacts(model_p):
    if model_p and os.path.exists(model_p):
        try:
            with open(model_p, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
            return None
    else:
        st.warning("Dashboard running in UI Preview mode.")
        return None


trained_model = load_serialized_artifacts(MODEL_PATH)

# Display active pipeline meta info in sidebar
if trained_model:
    st.sidebar.success(f"Loaded: `{selected_pipeline_file}`")
else:
    st.sidebar.info("Using Preview Mode (No Model Loaded)")

# =====================================================================
# 3. INTERACTIVE PAYLOAD INTAKE UI
# =====================================================================
st.title("🔮 CSA Real-Time Donor Response Predictor")
st.markdown("Enter individual demographic features and historical metric values to estimate conversion probability.")

st.info("**Project Requirement Reminder:** The Target Variable (`TARGET_B`) is strictly isolated and is not consumed as an input metric.")

st.header("Donor Profile Feature Construction Grid")

# Form separation structures for layout optimization
tab1, tab2, tab3 = st.tabs(["👤 Core Demographics", "📈 Recent Engagement Metrics", "📜 Lifetime Values & Background"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        donor_age = st.number_input(COLUMN_MAP['DONOR_AGE'], min_value=18, max_value=110, value=55, step=1)
        donor_gender = st.selectbox(COLUMN_MAP['DONOR_GENDER'], options=['F', 'M', 'U'], index=0)
        children = st.number_input(COLUMN_MAP['CHILDREN'], min_value=0, max_value=15, value=0, step=1)
    with col2:
        urbanicity = st.selectbox(COLUMN_MAP['URBANICITY'], options=['U', 'C', 'T', 'S', 'R'], index=3) # Default 'S'
        ses_profile = st.slider(COLUMN_MAP['SES'], min_value=1, max_value=5, value=2, step=1)
        home_owner = st.selectbox(COLUMN_MAP['HOME_OWNER'], options=["Homeowner", "Non-Homeowner"], index=0)
    with col3:
        income_grp = st.slider(COLUMN_MAP['INCOME_GROUP'], min_value=1, max_value=7, value=4, step=1)
        wealth_rat = st.slider(COLUMN_MAP['WEALTH_RATING'], min_value=0, max_value=9, value=5, step=1)

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        last_gift = st.number_input(COLUMN_MAP['LAST_GIFT_AMT'], min_value=0.0, max_value=5000.0, value=15.0, step=1.0)
        rec_avg_gift = st.number_input(COLUMN_MAP['RECENT_AVG_GIFT_AMT'], min_value=0.0, max_value=5000.0, value=14.5, step=1.0)
        rec_card_avg = st.number_input(COLUMN_MAP['RECENT_AVG_CARD_GIFT_AMT'], min_value=0.0, max_value=5000.0, value=12.0, step=1.0)
    with col2:
        rec_resp_prop = st.slider(COLUMN_MAP['RECENT_RESPONSE_PROP'], min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        rec_card_prop = st.slider(COLUMN_MAP['RECENT_CARD_RESPONSE_PROP'], min_value=0.0, max_value=1.0, value=0.20, step=0.01)
        rec_resp_cnt = st.number_input(COLUMN_MAP['RECENT_RESPONSE_COUNT'], min_value=0, max_value=100, value=3, step=1)
        rec_card_cnt = st.number_input(COLUMN_MAP['RECENT_CARD_RESPONSE_COUNT'], min_value=0, max_value=100, value=2, step=1)
    with col3:
        prom_12 = st.number_input(COLUMN_MAP['NUMBER_PROM_12'], min_value=0, max_value=200, value=12, step=1)
        card_prom_12 = st.number_input(COLUMN_MAP['CARD_PROM_12'], min_value=0, max_value=100, value=6, step=1)
        mos_since_prom = st.number_input(COLUMN_MAP['MONTHS_SINCE_LAST_PROM_RESP'], min_value=0, max_value=60, value=4, step=1)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Lifetime Campaign Totals")
        lt_prom = st.number_input(COLUMN_MAP['LIFETIME_PROM'], min_value=0, max_value=500, value=45, step=1)
        lt_card_prom = st.number_input(COLUMN_MAP['LIFETIME_CARD_PROM'], min_value=0, max_value=300, value=22, step=1)
        lt_gift_amt = st.number_input(COLUMN_MAP['LIFETIME_GIFT_AMOUNT'], min_value=0.0, max_value=50000.0, value=120.0, step=5.0)
        lt_gift_cnt = st.number_input(COLUMN_MAP['LIFETIME_GIFT_COUNT'], min_value=0, max_value=200, value=8, step=1)
        lt_max_gift = st.number_input(COLUMN_MAP['LIFETIME_MAX_GIFT_AMT'], min_value=0.0, max_value=10000.0, value=30.0, step=5.0)
        lt_min_gift = st.number_input(COLUMN_MAP['LIFETIME_MIN_GIFT_AMT'], min_value=0.0, max_value=1000.0, value=5.0, step=1.0)
        mos_last_gift = st.number_input(COLUMN_MAP['MONTHS_SINCE_LAST_GIFT'], min_value=0, max_value=120, value=6, step=1)
        mos_first_gift = st.number_input(COLUMN_MAP['MONTHS_SINCE_FIRST_GIFT'], min_value=0, max_value=300, value=48, step=1)
        file_card = st.number_input(COLUMN_MAP['FILE_CARD_GIFT'], min_value=0.0, max_value=5000.0, value=11.5, step=1.0)
    with col2:
        st.subheader("Status Codes & Neighborhood Data")
        pep_star = st.selectbox(COLUMN_MAP['PEP_STAR'], options=["Standard Donor", "Star Donor"], index=0)
        rec_star = st.selectbox(COLUMN_MAP['RECENT_STAR_STATUS'], options=["Not Achieved", "Achieved"], index=0)
        rec_status = st.selectbox(COLUMN_MAP['RECENCY_STATUS_96NK'], options=['A', 'E', 'F', 'L', 'N', 'S'], index=0)
        freq_status = st.number_input(COLUMN_MAP['FREQUENCY_STATUS_97NK'], min_value=1, max_value=10, value=1, step=1)
        
        med_home = st.number_input(COLUMN_MAP['MEDIAN_HOME_VALUE'], min_value=0, max_value=10000, value=850, step=50)
        med_inc = st.number_input(COLUMN_MAP['MEDIAN_HOUSEHOLD_INCOME'], min_value=0, max_value=5000, value=380, step=10)
        per_capita = st.number_input(COLUMN_MAP['PER_CAPITA_INCOME'], min_value=0, max_value=100000, value=14500, step=500)
        pct_owner = st.slider(COLUMN_MAP['PCT_OWNER_OCCUPIED'], min_value=0, max_value=100, value=72, step=1)
        
        pct_attr1 = st.slider(COLUMN_MAP['PCT_ATTRIBUTE1'], min_value=0, max_value=100, value=1, step=1)
        pct_attr2 = st.slider(COLUMN_MAP['PCT_ATTRIBUTE2'], min_value=0, max_value=100, value=12, step=1)
        pct_attr3 = st.slider(COLUMN_MAP['PCT_ATTRIBUTE3'], min_value=0, max_value=100, value=35, step=1)
        pct_attr4 = st.slider(COLUMN_MAP['PCT_ATTRIBUTE4'], min_value=0, max_value=100, value=20, step=1)

# =====================================================================
# 4. TRANSLATION TO ENGINE-READY DATAFRAME PAYLOAD
# =====================================================================

# Convert human labels back into raw representations for the pipeline
home_owner_raw = 1 if home_owner == "Homeowner" else 0
pep_star_raw = 1 if pep_star == "Star Donor" else 0
rec_star_raw = 1 if rec_star == "Achieved" else 0

payload_dict = {
    'CONTROL_NUMBER': [999999], # Dummy index configuration placeholder
    'DONOR_AGE': [donor_age], 'URBANICITY': [urbanicity], 'SES': [ses_profile],
    'HOME_OWNER': [home_owner_raw], 'DONOR_GENDER': [donor_gender], 'INCOME_GROUP': [income_grp],
    'WEALTH_RATING': [wealth_rat], 'MEDIAN_HOME_VALUE': [med_home], 'MEDIAN_HOUSEHOLD_INCOME': [med_inc],
    'PCT_OWNER_OCCUPIED': [pct_owner], 'PER_CAPITA_INCOME': [per_capita], 'PCT_ATTRIBUTE1': [pct_attr1],
    'PCT_ATTRIBUTE2': [pct_attr2], 'PCT_ATTRIBUTE3': [pct_attr3], 'PCT_ATTRIBUTE4': [pct_attr4],
    'PEP_STAR': [pep_star_raw], 'RECENT_STAR_STATUS': [rec_star_raw], 'RECENCY_STATUS_96NK': [rec_status],
    'FREQUENCY_STATUS_97NK': [freq_status], 'RECENT_RESPONSE_PROP': [rec_resp_prop], 'RECENT_AVG_GIFT_AMT': [rec_avg_gift],
    'RECENT_CARD_RESPONSE_PROP': [rec_card_prop], 'RECENT_AVG_CARD_GIFT_AMT': [rec_card_avg], 'RECENT_RESPONSE_COUNT': [rec_resp_cnt],
    'RECENT_CARD_RESPONSE_COUNT': [rec_card_cnt], 'MONTHS_SINCE_LAST_PROM_RESP': [mos_since_prom], 'LIFETIME_CARD_PROM': [lt_card_prom],
    'LIFETIME_PROM': [lt_prom], 'LIFETIME_GIFT_AMOUNT': [lt_gift_amt], 'LIFETIME_GIFT_COUNT': [lt_gift_cnt],
    'LIFETIME_MAX_GIFT_AMT': [lt_max_gift], 'LIFETIME_MIN_GIFT_AMT': [lt_min_gift], 'LAST_GIFT_AMT': [last_gift],
    'CARD_PROM_12': [card_prom_12], 'NUMBER_PROM_12': [prom_12], 'MONTHS_SINCE_LAST_GIFT': [mos_last_gift],
    'MONTHS_SINCE_FIRST_GIFT': [mos_first_gift], 'FILE_CARD_GIFT': [file_card], 'CHILDREN': [children]
}

raw_input_df = pd.DataFrame(payload_dict)

st.divider()


st.header("🎯 Predictive Output Strategy")

if st.button("Compute Donor Conversion Response Pipeline", type="primary"):
    
    if not trained_model:
        st.error("No pipeline is currently loaded. Please place a `.pkl` file in your Pipelines folder.")
    else:
        # KEEP CONTROL_NUMBER because your custom pipeline expects it in the raw feature space!
        processed_payload = raw_input_df.copy()
        
        if 'TARGET_B' in processed_payload.columns:
            processed_payload = processed_payload.drop(columns=['TARGET_B'])

        # Wrap execution in a spinner context manager
        with st.spinner(f"Executing pipeline ({selected_pipeline_file}) and computing donor probability..."):
            try:
                # Let the pipeline do ALL the heavy lifting (Cleaning -> Transforming -> Predicting)
                prediction = trained_model.predict(processed_payload)[0]
                probabilities = trained_model.predict_proba(processed_payload)[0]
                donor_prob = probabilities[1]
                
                # Extract the tuned threshold if available to show the user why a decision was made
                best_threshold = 0.5
                if isinstance(trained_model, dict) and 'model' in trained_model:
                    best_threshold = getattr(trained_model['model'], 'best_threshold_', 0.5)
                elif hasattr(trained_model, 'steps'):
                    best_threshold = getattr(trained_model.steps[-1][1], 'best_threshold_', 0.5)
                else:
                    best_threshold = getattr(trained_model, 'best_threshold_', 0.5)
                
                # Display localized KPIs
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    if prediction == 1:
                        st.success("### Prediction: 🟢 WILL DONATE")
                    else:
                        st.error("### Prediction: 🔴 WILL NOT DONATE")
                    
                    # Contextual sanity check for the user regarding the custom threshold
                    st.caption(f"Decision Threshold applied by Tuned Classifier: **{round(best_threshold, 3)}**")
                        
                with col_res2:
                    st.metric(label="Calculated Positive Conversion Probability", value=f"{round(donor_prob * 100, 2)}%")
                    st.progress(float(donor_prob))
                    
            except Exception as e:
                st.error(f"Model Execution Error: {e}. Check if feature spaces align with training outputs.")

# =====================================================================
# 5. OPTIONAL EXTRA SECTION: REMAINING PIPELINES BENCHMARK
# =====================================================================
st.write("")
with st.expander("📊 Compare with Remaining Pipelines (Optional Cross-Validation)", expanded=False):
    # Filter out the currently selected pipeline model to avoid duplication
    other_pipelines = [f for f in available_pipelines if f != selected_pipeline_file]
    
    if not other_pipelines:
        st.info("No alternative model pipelines found in directory to compare.")
    else:
        st.markdown("Click below to run the active donor payload profile through alternative discovery configurations simultaneously.")
        
        if st.button("Execute Cross-Model Benchmark Pipeline Run"):
            # Prepare an extraction dictionary for tracking results mapping
            comparison_records = []
            
            with st.spinner("Processing cross-model pipeline comparisons..."):
                for pipe_file in other_pipelines:
                    temp_path = os.path.join(PIPELINES_DIR, pipe_file)
                    try:
                        with open(temp_path, 'rb') as f:
                            alt_model = pickle.load(f)
                        
                        # Generate inference metrics safely
                        alt_pred = alt_model.predict(raw_input_df.copy())[0]
                        alt_prob_arr = alt_model.predict_proba(raw_input_df.copy())[0]
                        alt_prob = alt_prob_arr[1]
                        
                        # Fallback parsing strategy for decision threshold metrics
                        alt_threshold = 0.5
                        if isinstance(alt_model, dict) and 'model' in alt_model:
                            alt_threshold = getattr(alt_model['model'], 'best_threshold_', 0.5)
                        elif hasattr(alt_model, 'steps'):
                            alt_threshold = getattr(alt_model.steps[-1][1], 'best_threshold_', 0.5)
                        else:
                            alt_threshold = getattr(alt_model, 'best_threshold_', 0.5)
                        
                        comparison_records.append({
                            "Pipeline Artifact Model": pipe_file,
                            "Prediction Outcome": "🟢 WILL DONATE" if alt_pred == 1 else "🔴 WILL NOT DONATE",
                            "Conversion Probability": f"{round(alt_prob * 100, 2)}%",
                            "Tuned Decision Threshold": round(alt_threshold, 3)
                        })
                    except Exception as alt_err:
                        comparison_records.append({
                            "Pipeline Artifact Model": pipe_file,
                            "Prediction Outcome": "❌ Failed to Run",
                            "Conversion Probability": "N/A",
                            "Tuned Decision Threshold": f"Error: {str(alt_err)[:40]}..."
                        })
            
            # Construct summary frame matrix and render via optimized view container
            comparison_df = pd.DataFrame(comparison_records)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)