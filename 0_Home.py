import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="CSA Donor Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main Page Content ---

st.title("📊 CSA Donor Behavior & Conversion Dashboard")
st.markdown("""
Welcome to the **Charity/Donor Behavior Analysis Engine**. This interactive
            application provides a deep-dive exploratory interface and machine
            learning pipeline to understand donor demographics, tracking
            metrics, and predicting conversion yields for optimization
            campaign strategies.
""")

st.info("💡 **Navigation Tip:** Expand the sidebar on the left to seamlessly "
        "pivot between macro data exploration and micro predictive profiling.")

st.markdown("---")

# Section 1: Data Exploration
st.subheader("📈 Page 1: Data  Exploration Page")
st.markdown("""
This page focuses on auditing raw donor universes and breaking down
            demographic distributions to find foundational performance traits
            that can add value to the CSA.
Here you can:
* **Slice & Filter Data:** Dynamically narrow the cohort down using
            demographic factors (e.g., *Donor Age, Income Group,
            Wealth Rating*) and historical actions (e.g., *Last Gift Amount*).
* **Evaluate Distributions:** Run statistical inspection patterns
            using stacked histograms and boxplots split across performance
            outcomes (*Donated vs. Didn't Donate*).
* **Discover Bivariate Relationships:** Construct coordinate mapping metrics
            to identify underlying correlations between distinct features like
            *Median Household Income* vs. *Recent Average Gift Amount*.
""")

st.markdown("---")

# Section 2: Real-Time Donation Predictor
st.subheader("🔮 Page 2: Real-Time Donor Response Predictor")
st.markdown("""
This pages deploys our optimized predictive pipeline to compute transaction
            responses for individual, mock, or incoming prospective profiles.
Here you can:
* **Construct Target Mock Profiles:** Tweak individual parameters under
            structured tabs (*Core Demographics*, *Recent Engagement Metrics*,
            and *Lifetime Values*).
* **Isolate Target Data Requirements:** Test conversion behaviors in a
            controlled environment where the target variable (`TARGET_B`)
            remains fully isolated.
* **Run Production-Grade Inferences:** Push data payloads through custom
            cleaning wrappers to extract calculated **Positive Conversion
            Probabilities** and algorithmic strategy outcomes.
""")

st.markdown("---")
