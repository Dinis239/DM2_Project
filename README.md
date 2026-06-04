# Reach for Change - Predicting Donor Response to Optimize Outreach for Social Good

**Course Unit:** Data Mining II (2025/26)

**Group Number:** 12

**Members and Group Member Contribution:**
- Dinis Gaspar - 20221869 (40%)
    - Dataset Exploration
    - Additional Preprocessing
    - Modelling Methodology
    - Pipeline Development
    - Parameter Search
    - Interpretation of Results
    - Dashboard Development

- Margarida Cruz - 20221929 (30%)
    - Dataset Exploration
    - EDA Conclusions
    - Modelling Methodology
    - Pipeline Development
    - Interpretation of Results

- Beatriz Boura - 20250272 (30%)
    - Dataset Exploration
    - EDA Conclusions
    - Modelling Methodology
    - Interpretation of Results
    - Dashboard Development

# Project Overview

In recent years, nonprofit organizations have faced a growing challenge: while charitable causes have multiplied, public tolerance for repeated, generic solicitations has significantly decreased, often leading to donor fatigue and long-term disengagement. To address this issue, the Civic Support Alliance (CSA), rather than launching blanket campaigns across their entire database, the organization aims to transition to a highly targeted approach. The goal is to maximize operational efficiency and maintain donor respect by contacting fewer, but more receptive, individuals. 

In this project, we will tackle the task of opimtimizing the Civic Support Association's process for contacting portential donors, the goal is to develop machine learning pipelines that can help the Association target the correct potential donors to maximize the donations they bring in.

# Project Exe cution

If you wish to run the project locally, we recommend creating a virtual environment with Python version 3.12.4 as that was the version used during development, ipython and pip. Then using pip o install all required packages from the requirements.txt file. In the Anaconda prompt, this would be as follows:
- 1 - Run the following command: conda create --name DM2_Group12_2026env python=3.12.4 ipython pip - You may change the name
- 2 - Run the following command: conda activate DM2_Group12_2026env
- 3 - Within the anaconda prompt navigate to where the folder is, entering the folder. Copy the path of the delivery folder and put "cd" followed by the path in the prompt
- 4 - Run the following command : pip install -r requirements.txt


# Project Repository Structure

For clarity, this project has been split into multiple notebooks which work in a chain-like way where data is exported from one notebook to the next. For this reason, in theory, it is better to run all notebooks in order. Additionally, all notebooks are numbered to showcase the order in which they are to ben run.

For more practical ease of use all of the datasets and tools that are exported and required in future notebooks have been exported to the Files folder allowing for the option to run one notebook without having to run all of the previous ones.

Below is the directory tree for the project, outlining the organization of data, exploratory analysis, model tracking, and the Streamlit deployment files.

```text
DM2_PROJECT/
├── EDA/                                    # Exploratory Data Analysis Phase
│   ├── 00_EDA.ipynb                        # First Notebook: Data distribution, profiling & visualization
│   └── utils_EDA.py                        # Helper functions specifically for EDA visualizations
├── Files/                                  # Project Data, Artifacts & Serializations
│   ├── Images/                             # Plot images and assets exported for notebooks
│   ├── Pickle Files/                       # Serialized objects for reproducibility and caching
│   │   ├── Pipelines/                      # Fitted best pipelines from each model type hyperparameter search
│   │   ├── Results/                        # DataFrames containing metric logs from hyperparameter grid/random searches
│   │   ├── data_cleaner.pkl                # Fitted Custom DataCleaner object for consistent preprocessing
│   │   ├── Kaggle_scores.pkl               # Tracking DataFrame for internal validation vs. Kaggle leaderboard scores
│   │   ├── model_testing_skf.pkl           # StratifiedKFold cross-validation splitter instance
│   │   ├── X_train_preprocessed.pkl        # Preprocessed training feature matrix (Hold-out split)
│   │   ├── X_val_preprocessed.pkl          # Preprocessed validation feature matrix (Hold-out split)
│   │   ├── y_train.pkl                     # Target vector for training data (Hold-out split)
│   │   └── y_val.pkl                       # Target vector for validation data (Hold-out split)
│   ├── Submissions/                        # Exported predictions formatted for Kaggle evaluation
│   ├── donors_test.csv                     # Raw test dataset (features only)
│   ├── donors_train.csv                    # Raw training dataset (features and labels)
│   └── sample_submission.csv               # Baseline submission template provided by the competition
├── Modeling/                               # Model Experimentation & Evaluation Stage
│   ├── 01_Modeling_Tools.ipynb             # Second Notebook: Establishing evaluation frameworks & utility pipelines
│   ├── 02_Modeling_DT.ipynb                # Decision Tree experiments
│   ├── 02_Modeling_Ensemble.ipynb          # Ensemble methods
│   ├── 02_Modeling_KNN.ipynb               # K-Nearest Neighbors experiments
│   ├── 02_Modeling_Naive_Bayes.ipynb       # Naive Bayes testing
│   ├── 02_Modeling_NN.ipynb                # Multi-Layer Perceptron / Neural Network architectures
│   ├── 02_Modeling_Regression.ipynb        # Logistic Regression Testing
│   ├── 03_Final_Notebook.ipynb             # Final compilation
│   └── utils_modeling.py                   # Global auxiliary modeling tools and scoring metrics
├── pages/                                  # Multi-page Streamlit application views
│   ├── 1_Exploratory_Analysis.py           # Dashboard view for data insights
│   └── 2_Donor_Prediction.py               # Interactive interface for live inference or batch predictions
├── .gitattributes                          # Git attributes configuration (e.g., LFS tracking if needed)
├── .gitignore                              # Version control exclusions (ignores .pkl, __pycache__, cache/)
├── 0_Home.py                               # Main landing page for the Streamlit web application
├── Notas.txt                               # Scratchpad for internal project notes and reminders
├── Project_Guidelines_DM2.pdf              # Project Guidleines
├── README.md                               # Project documentation (this file)
└── requirements.txt                        # Pinpointed library dependencies for environment replication