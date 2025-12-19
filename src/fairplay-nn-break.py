# python fairplay.py preprocess_full.csv(coming from preprocessing_utils.py)
import cupy as cp
import cudf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
import sklearn.ensemble as ske
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import matplotlib.pyplot as plt
import os

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.explainers import MetricTextExplainer

from joblib import dump, load
from collections import OrderedDict
import nvtx
import argparse
import time
from datetime import datetime

# Neural Network Changes starts
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Neural Network model for Loan Approval Prediction (Binary Classification)
# Start timing for original NN execution
start_time_orig_nn = datetime.now()

# Main Function
def main(infile, trained_model=None, verbose=False):
    print("main enters")
    
    # Initialize a DataFrame to track the size changes
    size_tracker = pd.DataFrame(columns=['Function', 'Rows', 'Columns'])
    
    # Function A-Load and preprocess the data and filtering out based on activity year and computing descriptives
    preprocessed_df, columns_to_retain = load_and_preprocess_data(infile)
    size_tracker.loc[len(size_tracker)] = ['After load_and_preprocess_data', 
                                         preprocessed_df.shape[0], 
                                         preprocessed_df.shape[1]]
    print(f"Number of rows after load_and_preprocess_data function: {preprocessed_df.shape[0]}")
    print(f"Current DataFrame shape: {preprocessed_df.shape}")

    # Function B-Initialize the results dictionary
    results_dict = initialize_results_dict()
    size_tracker.loc[len(size_tracker)] = ['After initialize_results_dict', 
                                         preprocessed_df.shape[0], 
                                         preprocessed_df.shape[1]]
    print(f"Current DataFrame shape: {preprocessed_df.shape}")

    # 2024 changes-Stratified split based on bins
    # Function C-(inbuilt train_test_split function)
    df_train, df_test = train_test_split(preprocessed_df, test_size=0.25, stratify=preprocessed_df['bins'], random_state=42)
    size_tracker.loc[len(size_tracker)] = ['After train_test_split', 
                                         preprocessed_df.shape[0], 
                                         preprocessed_df.shape[1]]
    print(f"Applying train_test_split")
    print(f"df_train size: {df_train.shape[0]}")
    print(f"df_test size: {df_test.shape[0]}")
    print(f"After applying train_test_split, preprocessed_df size: {preprocessed_df.shape[0]}")
    print(f"Current DataFrame shape: {preprocessed_df.shape}")

    # Function D-Updating preprocessed dataset with adding denial reason logic and modeling denial reason
    (
        denial_conditions, X_train_denial, X_test_denial, y_train_denial_reason, y_test_denial_reason, denial_nn, 
        nn_denial_train_pred, nn_denial_test_pred, rf_denial_train_accuracy, rf_denial_test_accuracy,
        nn_denial_train_accuracy, nn_denial_test_accuracy, lr_denial_train_accuracy, lr_denial_test_accuracy
    ) = update_dataset_and_model_denial_reasons(df_train, df_test, preprocessed_df, columns_to_retain, PROTECTED_ATTRIBUTES)
    size_tracker.loc[len(size_tracker)] = ['After process_fairness_and_neural_network', 
                                         preprocessed_df.shape[0], 
                                         preprocessed_df.shape[1]]
    print(f"df_train size: {df_train.shape[0]}")
    print(f"df_test size: {df_test.shape[0]}")
    print(f"After applying process_fairness_and_neural_network, preprocessed_df size: {preprocessed_df.shape[0]}")
    print(f"Current DataFrame shape: {preprocessed_df.shape}")

    # Function E-Looping around protected attributes
    print("process_protected_attribute function starts")
    results_dict, execution_time_reweighing, execution_time_orig, execution_time_unbiased, energy_reweighing, energy_orig, energy_unbiased = process_protected_attribute(
    denial_conditions, trained_model, X_train_denial, X_test_denial, y_train_denial_reason, y_test_denial_reason, df_train, df_test,
    preprocessed_df, columns_to_retain, PROTECTED_ATTRIBUTES, results_dict, rf_denial_train_accuracy, rf_denial_test_accuracy,
    nn_denial_train_accuracy, nn_denial_test_accuracy, verbose)

    # Transpose the DataFrame so that metrics are rows and attributes are columns
    results_df = pd.DataFrame(results_dict, index=PROTECTED_ATTRIBUTES).T

    # 2024 changes-Save the performance results to CSV
    results_df.to_csv('performance_results_activity_year_2018.csv')
    print("Results saved to CSV.")

    fairness_results_file_path="performance_results_activity_year_2018.csv"

    # Function F-Visualization starts here
    generate_fairness_plots(fairness_results_file_path)
    size_tracker.loc[len(size_tracker)] = ['After generate_fairness_plots', 
                                         preprocessed_df.shape[0], 
                                         preprocessed_df.shape[1]]
    print(f"Current DataFrame shape: {preprocessed_df.shape}")
    
    # Save the size tracker to a CSV file
    size_tracker.to_csv('dataframe_size_tracker.csv', index=False)
    print("\nDataFrame Size Tracker:")
    print(size_tracker)
    print("Size tracking data saved to 'dataframe_size_tracker.csv'")

    print("Plots saved to folder.")

# Function A
def load_and_preprocess_data(infile):
    """
    Load and preprocess the data from the input file.
    
    Args:
        infile (str): Path to the input CSV file.
    
    Returns:
        preprocessed_df (pd.DataFrame): The preprocessed DataFrame.
    """
    print("load_and_preprocess_data function enters")
    chunks = []
    chunk_size = 50000

    # Load and filter data by activity year
    for chunk in pd.read_csv(infile, index_col=0, chunksize=chunk_size):
        filtered_chunk = chunk[chunk['activity_year_2018'] == 1]
        chunks.append(filtered_chunk)    

    preprocessed_df = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"Data size after filtering by activity_year_2018: {len(preprocessed_df)}")

    # Try to convert the entire DataFrame to float32
    for column in preprocessed_df.columns:
        try:
            preprocessed_df[column] = preprocessed_df[column].astype('float32')
        except ValueError:
            print(f"Could not convert column {column} to float. Skipping it.")
    preprocessed_df['approved'] = preprocessed_df['approved'].astype('int32')

    # Drop unnecessary columns
    preprocessed_df.drop(columns=['activity_year_2018', 'activity_year_2019', 'activity_year_2020', 'activity_year_2021'], axis=1, inplace=True)
    preprocessed_df.fillna(0, inplace=True)

    # Drop rows based on specific conditions
    conditions = {
        "loan_type_1": 1,
        "loan_purpose_1": 1,
        "lien_status_1": 1,
        "reverse_mortgage_2": 1,
        "open-end_line_of_credit_2": 1,
        "business_or_commercial_purpose_2": 1,
        "occupancy_type_1": 1,
        "submission_of_application_1": 1
    }

    for column, condition in conditions.items():
        if column in preprocessed_df.columns:
            if condition == 0:
                preprocessed_df = preprocessed_df[preprocessed_df[column] != 0]
            elif condition == 1:
                preprocessed_df = preprocessed_df[preprocessed_df[column] != 1]

    # Drop specific columns as predictors
    columns_to_drop = [
        "credit_score",
        "preapproval_1",
        "preapproval_2",
        "negative_amortization_1111",
        "negative_amortization_2",
        "interest_only_payment_1111",
        "interest_only_payment_2",
        "balloon_payment_1111",
        "balloon_payment_2",
        "other_nonamortizing_features_1111",
        "other_nonamortizing_features_2",
        "submission_of_application_1111",
        "submission_of_application_2",
        "submission_of_application_3",
        "total_units",
        "reverse_mortgage_1111",
        "open-end_line_of_credit_1111",
        "business_or_commercial_purpose_1",
        "business_or_commercial_purpose_1111",
        "construction_method_1",
        "occupancy_type_2",
        "occupancy_type_3",
        "manufactured_home_secured_property_type_1111",
        "manufactured_home_secured_property_type_3"
    ]

    for column in columns_to_drop:
        if column in preprocessed_df.columns:
            preprocessed_df.drop(columns=[column], axis=1, inplace=True)

    columns_to_retain = [
        "loan_amount",
        "loan_to_value_ratio",
        "property_value",
        "income",
        "debt_to_income_ratio",
        "loan_term"
    ]

    # Sub Function A1-Compute and save descriptives for 2018 conditions
    all_conditions_count = compute_descriptives(preprocessed_df)
    print("all_conditions_count:\n", all_conditions_count)

    print(f"Data size after applying conditions: {len(preprocessed_df)}")

    # Add binning logic for 'tract_minority_population_percent'
    preprocessed_df['bins'] = pd.cut(preprocessed_df['tract_minority_population_percent'], bins=10, labels=False)

    return preprocessed_df, columns_to_retain

# Sub Function A1-2024 changes-Compute descriptive statistics based on conditions and return as a DataFrame.
def compute_descriptives(preprocessed_df):
    conditions = {
        "loan_type_1": 1,
        "loan_purpose_1": 1,
        "lien_status_1": 1,
        "reverse_mortgage_2": 1,
        "open-end_line_of_credit_2": 1,
        "business_or_commercial_purpose_2": 1,
        "occupancy_type_1": 1,
        "submission_of_application_1": 1
    }

    # Create a dict to store the counts
    descriptives = {
        "Condition": [],
        "Count": []
    }

    # Count instances that satisfy each individual condition
    for column, value in conditions.items():
        count = preprocessed_df[preprocessed_df[column] == value].shape[0]
        descriptives["Condition"].append(f'{column} == {value}')
        descriptives["Count"].append(count)
    
    # Count instances that meet all the conditions
    combined_condition_df = preprocessed_df.copy()
    for column, value in conditions.items():
        combined_condition_df = combined_condition_df[combined_condition_df[column] == value]
    
    all_conditions_count = combined_condition_df.shape[0]
    descriptives["Condition"].append("All conditions met")
    descriptives["Count"].append(all_conditions_count)

    # Convert to DataFrame
    descriptives_df = pd.DataFrame(descriptives)

    # Save descriptives to a CSV file
    descriptives_df.to_csv('descriptives_2018.csv', index=False)
    print(f"Descriptives saved to 'descriptives_2018.csv'.")
    
    return all_conditions_count  # Return the count of instances meeting all conditions

# Function B
def initialize_results_dict():
    return {
        "#### Original training dataset": [],
        "RF Statistical parity difference (Original Train)": [],
        "RF Disparate impact (Original Train)": [],
        "RF Mean difference (Original Train)": [],
        "NN Statistical parity difference (Original Train)": [],
        "NN Disparate impact (Original Train)": [],
        "NN Mean difference (Original Train)": [],
        "LR Statistical parity difference (Original Train)": [],
        "LR Disparate impact (Original Train)": [],
        "LR Mean difference (Original Train)": [],
        "#### Reweighted training dataset": [],
        "RF Statistical parity difference (Reweighted Train)": [],
        "RF Disparate impact (Reweighted Train)": [],
        "RF Mean difference (Reweighted Train)": [],
        "NN Statistical parity difference (Reweighted Train)": [],
        "NN Disparate impact (Reweighted Train)": [],
        "NN Mean difference (Reweighted Train)": [],

        #Open item-1 After review changes starts
        "LR Statistical parity difference (Reweighted Train)": [],
        "LR Disparate impact (Reweighted Train)": [],
        "LR Mean difference (Reweighted Train)": [],
        #Open item-1 After review changes ends

        "#### Repaired (DIR) training dataset": [],
        "RF Statistical parity difference (Repaired Train)": [],
        "RF Disparate impact (Repaired Train)": [],
        "RF Mean difference (Repaired Train)": [],
        "NN Statistical parity difference (Repaired Train)": [],
        "NN Disparate impact (Repaired Train)": [],
        "NN Mean difference (Repaired Train)": [],
        "LR Statistical parity difference (Repaired Train)": [],
        "LR Disparate impact (Repaired Train)": [],
        "LR Mean difference (Repaired Train)": [],
        "#### Original testing dataset": [],
        "RF Statistical parity difference (Original Test)": [],
        "RF Disparate impact (Original Test)": [],
        "RF Mean difference (Original Test)": [],
        "NN Statistical parity difference (Original Test)": [],
        "NN Disparate impact (Original Test)": [],
        "NN Mean difference (Original Test)": [],
        "LR Statistical parity difference (Original Test)": [],
        "LR Disparate impact (Original Test)": [],
        "LR Mean difference (Original Test)": [],
        "#### Reweighted testing dataset": [],
        "RF Statistical parity difference (Reweighted Test)": [],
        "RF Disparate impact (Reweighted Test)": [],
        "RF Mean difference (Reweighted Test)": [],
        "NN Statistical parity difference (Reweighted Test)": [],
        "NN Disparate impact (Reweighted Test)": [],
        "NN Mean difference (Reweighted Test)": [],

        #Open item-1 After review changes starts
        "LR Statistical parity difference (Reweighted Test)": [],
        "LR Disparate impact (Reweighted Test)": [],
        "LR Mean difference (Reweighted Test)": [],
        #Open item-1 After review changes ends

        "#### Repaired (DIR) testing dataset": [],
        "RF Statistical parity difference (Repaired Test)": [],
        "RF Disparate impact (Repaired Test)": [],
        "RF Mean difference (Repaired Test)": [],
        "NN Statistical parity difference (Repaired Test)": [],
        "NN Disparate impact (Repaired Test)": [],
        "NN Mean difference (Repaired Test)": [],
        "LR Statistical parity difference (Repaired Test)": [],
        "LR Disparate impact (Repaired Test)": [],
        "LR Mean difference (Repaired Test)": [],
        "The Original Training Accuracy Score (RF)": [],
        "The Original Testing Accuracy Score (RF)": [],
        "The Original Training F1 Score (RF)": [],
        "The Original Testing F1 Score (RF)": [],
        "The Original Training Recall Score (RF)": [],
        "The Original Testing Recall Score (RF)": [],
        "The Original Training Precision Score (RF)": [],
        "The Original Testing Precision Score (RF)": [],
        "The Original Training Accuracy Score (NN)": [],
        "The Original Testing Accuracy Score (NN)": [],
        "The Original Training F1 Score (NN)": [],
        "The Original Testing F1 Score (NN)": [],
        "The Original Training Recall Score (NN)": [],
        "The Original Testing Recall Score (NN)": [],
        "The Original Training Precision Score (NN)": [],
        "The Original Testing Precision Score (NN)": [],
        "The Original Training Accuracy Score (LR)": [],
        "The Original Testing Accuracy Score (LR)": [],
        "The Original Training F1 Score (LR)": [],
        "The Original Testing F1 Score (LR)": [],
        "The Original Training Recall Score (LR)": [],
        "The Original Testing Recall Score (LR)": [],
        "The Original Training Precision Score (LR)": [],
        "The Original Testing Precision Score (LR)": [],
        "RF Training Accuracy (Repaired)": [],
        "RF Testing Accuracy (Repaired)": [],
        "NN Training Accuracy (Repaired)": [],
        "NN Testing Accuracy (Repaired)": [],
        "LR Training Accuracy (Repaired)": [],
        "LR Testing Accuracy (Repaired)": [],

        #Open item-3 After review changes starts
        "The Repaired Training F1 Score (RF)": [],
        "The Repaired Testing F1 Score (RF)": [],
        "The Repaired Training Recall Score (RF)": [],
        "The Repaired Testing Recall Score (RF)": [],
        "The Repaired Training Precision Score (RF)": [],
        "The Repaired Testing Precision Score (RF)": [],
        "The Repaired Training F1 Score (NN)": [],
        "The Repaired Testing F1 Score (NN)": [],
        "The Repaired Training Recall Score (NN)": [],
        "The Repaired Testing Recall Score (NN)": [],
        "The Repaired Training Precision Score (NN)": [],
        "The Repaired Testing Precision Score (NN)": [],
        "The Repaired Training F1 Score (LR)": [],
        "The Repaired Testing F1 Score (LR)": [],
        "The Repaired Training Recall Score (LR)": [],
        "The Repaired Testing Recall Score (LR)": [],
        "The Repaired Training Precision Score (LR)": [],
        "The Repaired Testing Precision Score (LR)": [],
        #Open item-3 After review changes ends


        "#### Original Training AIF360 Fairness Metrics:": [],
        "Average odds difference (Original Train) (RF)": [],
        "Equal opportunity difference (Original Train) (RF)": [],
        "Theil index (Original Train) (RF)": [],
        "Average odds difference (Original Train) (NN)": [],
        "Equal opportunity difference (Original Train) (NN)": [],
        "Theil index (Original Train) (NN)": [],
        "Average odds difference (Original Train) (LR)": [],
        "Equal opportunity difference (Original Train) (LR)": [],
        "Theil index (Original Train) (LR)": [],
        "#### Original Testing AIF360 Fairness Metrics:": [],
        "Average odds difference (Original Test) (RF)": [],
        "Equal opportunity difference (Original Test) (RF)": [],
        "Theil index (Original Test) (RF)": [],
        "Average odds difference (Original Test) (NN)": [],
        "Equal opportunity difference (Original Test) (NN)": [],
        "Theil index (Original Test) (NN)": [],
        "Average odds difference (Original Test) (LR)": [],
        "Equal opportunity difference (Original Test) (LR)": [],
        "Theil index (Original Test) (LR)": [],

        #Open item-2 After review changes starts
        "#### Repaired Training AIF360 Fairness Metrics:": [],
        "Average odds difference (Repaired Train) (RF)": [],
        "Equal opportunity difference (Repaired Train) (RF)": [],
        "Theil index (Repaired Train) (RF)": [],
        "Average odds difference (Repaired Train) (NN)": [],
        "Equal opportunity difference (Repaired Train) (NN)": [],
        "Theil index (Repaired Train) (NN)": [],
        "Average odds difference (Repaired Train) (LR)": [],
        "Equal opportunity difference (Repaired Train) (LR)": [],
        "Theil index (Repaired Train) (LR)": [],
        "#### Repaired Testing AIF360 Fairness Metrics:": [],
        "Average odds difference (Repaired Test) (RF)": [],
        "Equal opportunity difference (Repaired Test) (RF)": [],
        "Theil index (Repaired Test) (RF)": [],
        "Average odds difference (Repaired Test) (NN)": [],
        "Equal opportunity difference (Repaired Test) (NN)": [],
        "Theil index (Repaired Test) (NN)": [],
        "Average odds difference (Repaired Test) (LR)": [],
        "Equal opportunity difference (Repaired Test) (LR)": [],
        "Theil index (Repaired Test) (LR)": [],
        #Open item-2 After review changes ends

        #Unbiased is Reweighing
        "The Unbiased Training Accuracy Score (RF)": [],
        "The Unbiased Testing Accuracy Score (RF)": [],
        "The Unbiased Training F1 Score (RF)": [],
        "The Unbiased Testing F1 Score (RF)": [],
        "The Unbiased Training Recall Score (RF)": [],
        "The Unbiased Testing Recall Score (RF)": [],
        "The Unbiased Training Precision Score (RF)": [],
        "The Unbiased Testing Precision Score (RF)": [],
        "The Unbiased Training Accuracy Score (NN)": [],
        "The Unbiased Testing Accuracy Score (NN)": [],
        "The Unbiased Training F1 Score (NN)": [],
        "The Unbiased Testing F1 Score (NN)": [],
        "The Unbiased Training Recall Score (NN)": [],
        "The Unbiased Testing Recall Score (NN)": [],
        "The Unbiased Training Precision Score (NN)": [],
        "The Unbiased Testing Precision Score (NN)": [],
        "The Unbiased Training Accuracy Score (LR)": [],
        "The Unbiased Testing Accuracy Score (LR)": [],
        "The Unbiased Training F1 Score (LR)": [],
        "The Unbiased Testing F1 Score (LR)": [],
        "The Unbiased Training Recall Score (LR)": [],
        "The Unbiased Testing Recall Score (LR)": [],
        "The Unbiased Training Precision Score (LR)": [],
        "The Unbiased Testing Precision Score (LR)": [],
        "#### Unbiased Training AIF360 Fairness Metrics:": [],
        "Average odds difference (Unbiased Train) (RF)": [],
        "Equal opportunity difference (Unbiased Train) (RF)": [],
        "Theil index (Unbiased Train) (RF)": [],
        "Average odds difference (Unbiased Train) (NN)": [],
        "Equal opportunity difference (Unbiased Train) (NN)": [],
        "Theil index (Unbiased Train) (NN)": [],
        "Average odds difference (Unbiased Train) (LR)": [],
        "Equal opportunity difference (Unbiased Train) (LR)": [],
        "Theil index (Unbiased Train) (LR)": [],
        "#### Unbiased Testing AIF360 Fairness Metrics:": [],
        "Average odds difference (Unbiased Test) (RF)": [],
        "Equal opportunity difference (Unbiased Test) (RF)": [],
        "Theil index (Unbiased Test) (RF)": [],
        "Average odds difference (Unbiased Test) (NN)": [],
        "Equal opportunity difference (Unbiased Test) (NN)": [],
        "Theil index (Unbiased Test) (NN)": [],
        "Average odds difference (Unbiased Test) (LR)": [],
        "Equal opportunity difference (Unbiased Test) (LR)": [],
        "Theil index (Unbiased Test) (LR)": [],
        "Denial Reason Training Accuracy Score (RF)": [],
        "Denial Reason Testing Accuracy Score (RF)": [],
        "Denial Reason Training Accuracy Score (NN)": [],
        "Denial Reason Testing Accuracy Score (NN)": [],
        "Denial Reason Training Accuracy Score (LR)": [],
        "Denial Reason Testing Accuracy Score (LR)": [],
        "Execution_Time_unbiased (Hours)": [],
        "Execution_Time_orig (Hours)": [],
        "Energy_Unbiased (kWh)": [],
        "Energy_Orig (kWh)": [],
        "Execution_Time_unbiased_nn (Hours)": [],
        "Execution_Time_orig_nn (Hours)": [],
        "Energy_Unbiased_nn (kWh)": [],
        "Energy_Orig_nn (kWh)": [],

        #Open item-5 After review changes starts
        "Execution_Time_Orig_lr (Hours)":[],
        "Energy_Orig_lr (kWh)": [],
        "Execution_Time_Reweighted_lr (Hours)":[],
        "Energy_Reweighted_lr (kWh)": [],
        "Execution_Time_Orig_dir (Hours)":[],
        "Energy_Orig_dir (kWh)": [],
        "Execution_Time_Model4_9 (Hours)":[],
        "Energy_Model4_9 (kWh)": [],
        "Execution_Time_Model10_15 (Hours)":[],
        "Energy_Model10_15 (kWh)": [],
        "Execution_Time_Model16_21 (Hours)":[],
        "Energy_Model16_21 (kWh)": [],
        "Execution_Time_Model22_27 (Hours)":[],
        "Energy_Model22_27 (kWh)": [],
        "Execution_Time_Model28_33 (Hours)":[],
        "Energy_Model28_33 (kWh)": [],
        "Execution_Time_Model34_39 (Hours)":[],
        "Energy_Model34_39 (kWh)": []
        #Open item-5 After review changes ends
    }

# Function D
def update_dataset_and_model_denial_reasons(df_train, df_test, preprocessed_df, columns_to_retain, PROTECTED_ATTRIBUTES):
    """
    Process and prepare data for denial reasons using original dataset.
    
    Args:
        df_train (pd.DataFrame): Training dataset.
        df_test (pd.DataFrame): Testing dataset.
        preprocessed_df (pd.DataFrame): Preprocessed dataset.
        columns_to_retain (list): List of columns to retain for modeling.
        PROTECTED_ATTRIBUTES (list): List of protected attributes.
    
    Returns:
        tuple: Contains denial reason data (same as before but computed from original dataset)
    """
    print("update_dataset_and_model_denial_reasons function starts")
    
    # Identify non-numeric columns
    non_numeric_cols = df_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"Non-numeric columns found: {non_numeric_cols}")
        df_train = df_train.drop(columns=non_numeric_cols)
        df_test = df_test.drop(columns=non_numeric_cols)

    # Ensure all columns are numeric
    df_train = df_train.apply(pd.to_numeric, errors='coerce')
    df_test = df_test.apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values (if needed)
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # After stratified split
    print(f"Training set size: {len(df_train)}, Test set size: {len(df_test)}")

    # Remove 'bins' column after splitting
    df_train.drop(columns=['bins'], inplace=True)
    df_test.drop(columns=['bins'], inplace=True)

    # Check distribution in original data, train set, and test set
    original_dist = preprocessed_df['tract_minority_population_percent'].describe()
    train_dist = df_train['tract_minority_population_percent'].describe()
    test_dist = df_test['tract_minority_population_percent'].describe()

    print("Original Distribution:\n", original_dist)
    print("\nTrain Distribution:\n", train_dist)
    print("\nTest Distribution:\n", test_dist)

    # Define the predictors and protected attributes
    predictors = ["income", "loan_amount", "loan_to_value_ratio", "property_value", "debt_to_income_ratio", "loan_term"]
    protected_attributes = PROTECTED_ATTRIBUTES

    # Compute the new dataset with unbiased approvals (using original dataset)
    preprocessed_df = compute_unbiased_approvals(preprocessed_df, predictors, protected_attributes)
    preprocessed_df.to_csv("preprocess_full_added_denialreasons.csv", index=False)

    denial_conditions = {
        'debt_to_income_ratio': ['denial_reason-1_1', 'denial_reason-2_1.0', 'denial_reason-3_1.0', 'denial_reason-4_1.0'],
        'collateral': ['denial_reason-1_4', 'denial_reason-2_4.0', 'denial_reason-3_4.0', 'denial_reason-4_4.0'],
        'insufficient_cash': ['denial_reason-1_5', 'denial_reason-2_5.0', 'denial_reason-3_5.0', 'denial_reason-4_5.0']
    }

    # Prepare denial reason data using original dataset (not just denied loans)
    denial_columns = [
        'denial_reason-1_1', 'denial_reason-1_2', 'denial_reason-1_3', 'denial_reason-1_4', 'denial_reason-1_5',
        'denial_reason-1_6', 'denial_reason-1_7', 'denial_reason-1_8', 'denial_reason-1_9', 'denial_reason-1_10'
    ]

    # Get denial reasons from full dataset (not just denied loans)
    y_train_denial_reason = df_train[denial_columns].idxmax(axis=1).str.split('_').str[-1].astype(int)
    y_test_denial_reason = df_test[denial_columns].idxmax(axis=1).str.split('_').str[-1].astype(int)

    # Use all training data for features (not just denied loans)
    X_train_denial = df_train[columns_to_retain]
    X_test_denial = df_test[columns_to_retain]

    # Train Random Forest for denial reasons on full dataset
    clf_rf_denial = ske.RandomForestClassifier(n_estimators=120, random_state=np.random.seed(11850))
    clf_rf_denial.fit(X_train_denial, y_train_denial_reason)

    # Predict denial reasons using Random Forest on full dataset
    rf_denial_train_pred = clf_rf_denial.predict(X_train_denial)
    rf_denial_test_pred = clf_rf_denial.predict(X_test_denial)

    # Construct denial Reason NN
    num_classes = len(denial_columns)
    y_train_denial_reason_tensor = torch.clamp(torch.tensor(y_train_denial_reason.values), 0, num_classes-1).long().to(device)
    y_test_denial_reason_tensor = torch.clamp(torch.tensor(y_test_denial_reason.values), 0, num_classes-1).long().to(device)
    denial_nn = DenialReasonNN(input_dim=X_train_denial.shape[1], num_classes=num_classes).to(device)
    criterion_denial = nn.CrossEntropyLoss()
    optimizer_denial = optim.Adam(denial_nn.parameters(), lr=0.001)

    # Convert to DataLoader for Neural Network training (using full dataset)
    train_loader_denial = DataLoader(
        TensorDataset(
            torch.tensor(X_train_denial.values, dtype=torch.float32).to(device),
            y_train_denial_reason_tensor  # Use the clamped tensor directly
        ),
        batch_size=64, shuffle=True
    )

    test_loader_denial = DataLoader(
        TensorDataset(
            torch.tensor(X_test_denial.values, dtype=torch.float32).to(device),
            y_test_denial_reason_tensor  # Use the clamped tensor directly
        ),
        batch_size=64, shuffle=False
    )

    # Train Neural Network on full dataset
    train_neural_network(denial_nn, train_loader_denial, criterion_denial, optimizer_denial)

    # Evaluate the trained network on full dataset
    nn_denial_train_pred, nn_denial_train_labels = evaluate_neural_network(denial_nn, train_loader_denial, binary=False)
    nn_denial_test_pred, nn_denial_test_labels = evaluate_neural_network(denial_nn, test_loader_denial, binary=False)

    # Convert to numpy arrays if they are torch tensors for Logistic Regression
    if isinstance(X_train_denial, torch.Tensor):
        X_train_denial_lr = X_train_denial.cpu().numpy()
    else:
        X_train_denial_lr = X_train_denial.values if hasattr(X_train_denial, 'values') else X_train_denial
    
    if isinstance(y_train_denial_reason, torch.Tensor):
        y_train_denial_reason_lr = y_train_denial_reason.cpu().numpy()
    else:
        y_train_denial_reason_lr = y_train_denial_reason.values if hasattr(y_train_denial_reason, 'values') else y_train_denial_reason

    # Train Logistic Regression for denial reasons
    clf_lr_denial = train_logistic_regression_denial(X_train_denial_lr, y_train_denial_reason_lr)

    # Convert test data to numpy if needed
    if isinstance(X_test_denial, torch.Tensor):
        X_test_denial_lr = X_test_denial.cpu().numpy()
    else:
        X_test_denial_lr = X_test_denial.values if hasattr(X_test_denial, 'values') else X_test_denial
    
    if isinstance(y_test_denial_reason, torch.Tensor):
        y_test_denial_reason_lr = y_test_denial_reason.cpu().numpy()
    else:
        y_test_denial_reason_lr = y_test_denial_reason.values if hasattr(y_test_denial_reason, 'values') else y_test_denial_reason

    # Predict denial reasons using Logistic Regression
    lr_denial_train_pred = clf_lr_denial.predict(X_train_denial_lr)
    lr_denial_test_pred = clf_lr_denial.predict(X_test_denial_lr)

    # Denial reason accuracy scores (now computed on full dataset)
    rf_denial_train_accuracy = accuracy_score(y_train_denial_reason_lr, rf_denial_train_pred)
    rf_denial_test_accuracy = accuracy_score(y_test_denial_reason_lr, rf_denial_test_pred)
    
    # Convert PyTorch tensors to numpy arrays if needed
    if isinstance(nn_denial_train_pred, torch.Tensor):
        nn_denial_train_pred = nn_denial_train_pred.cpu().numpy()
    if isinstance(nn_denial_test_pred, torch.Tensor):
        nn_denial_test_pred = nn_denial_test_pred.cpu().numpy()
    
    nn_denial_train_accuracy = accuracy_score(y_train_denial_reason_lr, nn_denial_train_pred)
    nn_denial_test_accuracy = accuracy_score(y_test_denial_reason_lr, nn_denial_test_pred)

    # Calculate Accuracy Scores for Logistic Regression
    lr_denial_train_accuracy = accuracy_score(y_train_denial_reason_lr, lr_denial_train_pred)
    lr_denial_test_accuracy = accuracy_score(y_test_denial_reason_lr, lr_denial_test_pred)

    return (
        denial_conditions, X_train_denial, X_test_denial, y_train_denial_reason, y_test_denial_reason, denial_nn, 
        nn_denial_train_pred, nn_denial_test_pred, rf_denial_train_accuracy, rf_denial_test_accuracy,
        nn_denial_train_accuracy, nn_denial_test_accuracy, lr_denial_train_accuracy, lr_denial_test_accuracy
    )

# 2024 changes-Modifies the preprocessed dataset with all columns, predictors, original approval, and unbiased approval based on protected attributes.
# Sub Function D1-Updated compute_unbiased_approvals function with denial_reason_final logic
# The compute_unbiased_approvals is actually computing denial reason using original dataset
def compute_unbiased_approvals(df, predictors, protected_attributes):
    """
    Modifies the input DataFrame in-place by adding:
    - 'Original Approved' (copy of 'approved' column)
    - Protected attribute approval columns (e.g., 'tract_minority_percentage40_0_approved')
    - 'denial_reason_final' column with categorized denial reasons
    
    Args:
        df (pd.DataFrame): The input DataFrame to modify
        predictors (list): List of predictor column names (unused in this version)
        protected_attributes (list): List of protected attribute column names
        
    Returns:
        pd.DataFrame: The modified DataFrame (same as input df)
    """
    # Add 'Original Approved' column (direct copy of 'approved')
    df['Original Approved'] = df['approved']
    
    # Initialize list for denial reason categorization
    denial_reason_final = []
    
    for index, row in df.iterrows():
        # Categorize denial reasons
        if row['approved'] == 1:
            denial_reason_final.append(1)  # Approved
        elif any(row[col] == 1 for col in ['denial_reason-1_1', 'denial_reason-2_1.0', 
                                          'denial_reason-3_1.0', 'denial_reason-4_1.0']):
            denial_reason_final.append(2)  # Debt-to-income ratio
        elif any(row[col] == 1 for col in ['denial_reason-1_4', 'denial_reason-2_4.0',
                                          'denial_reason-3_4.0', 'denial_reason-4_4.0']):
            denial_reason_final.append(3)  # Collateral
        elif any(row[col] == 1 for col in ['denial_reason-1_5', 'denial_reason-2_5.0',
                                          'denial_reason-3_5.0', 'denial_reason-4_5.0']):
            denial_reason_final.append(4)  # Insufficient cash
        else:
            denial_reason_final.append(5)  # Other/unspecified
    
    # Add the denial reason column
    df['denial_reason_final'] = denial_reason_final
    
    # Add protected attribute approval columns
    for attribute in protected_attributes:
        df[f'{attribute}_approved'] = df[attribute]
    
    return df

# Sub Function D2-Neural Network for Multi-Class Denial Reason Classification
class DenialReasonNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DenialReasonNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.output(x)  # No activation; logits are used with CrossEntropyLoss
        return x

# Sub Function D3
def train_neural_network(model, train_loader, criterion, optimizer, num_epochs=30):
    model.to(device)  # Ensure the model is on the device
    model.train()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            # Move inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()  # Update the learning rate
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")    

    return model

# Sub Function D4
def evaluate_neural_network(model, test_loader, binary=True):
    """
    Evaluate a neural network model.

    Args:
        model (nn.Module): The neural network model.
        test_loader (DataLoader): The DataLoader for the test dataset.
        binary (bool): Whether the output is binary classification.

    Returns:
        predictions (numpy array): Predicted labels or values.
        true_labels (numpy array): True labels if provided, else an empty array.
    """
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if binary:
                    preds = (outputs.squeeze() >= 0.5).float()
                else:
                    preds = torch.argmax(outputs, dim=1)
                predictions.append(preds)
                true_labels.append(labels)
            else:
                inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
                inputs = inputs.to(device)
                outputs = model(inputs)
                if binary:
                    preds = (outputs.squeeze() >= 0.5).float()
                else:
                    preds = torch.argmax(outputs, dim=1)
                predictions.append(preds)
    
    # Convert all to numpy arrays before concatenation
    predictions = [p.cpu().numpy() for p in predictions]
    predictions = np.concatenate(predictions)
    
    if len(true_labels) > 0:
        true_labels = [l.cpu().numpy() for l in true_labels]
        true_labels = np.concatenate(true_labels)
        return predictions, true_labels
    
    return predictions, np.array([])

# Function E
def process_protected_attribute(denial_conditions, trained_model, X_train_denial, X_test_denial, y_train_denial_reason, y_test_denial_reason, df_train, 
    df_test, preprocessed_df, columns_to_retain, PROTECTED_ATTRIBUTES, results_dict, rf_denial_train_accuracy,  
    rf_denial_test_accuracy, nn_denial_train_accuracy, nn_denial_test_accuracy, verbose=False):
    """
    Process fairness metrics, reweighing, and timing for a given protected attribute.
    
    Args:
        attribute (str): The protected attribute to process.
        df_train (pd.DataFrame): Training dataset.
        df_test (pd.DataFrame): Testing dataset.
        preprocessed_df (pd.DataFrame): Preprocessed dataset.
        columns_to_retain (list): List of columns to retain for modeling.
        PROTECTED_ATTRIBUTES (list): List of protected attributes.
        results_dict (dict): Dictionary to store results.
        verbose (bool): Whether to print detailed logs.
    
    Returns:
        tuple: Contains updated results dictionary, execution times, and energy consumption.
    """
    for attribute in PROTECTED_ATTRIBUTES:
        print(f"Processing for protected attribute: {attribute}")

        if attribute not in preprocessed_df.columns:
            return results_dict, 0, 0, 0, 0, 0, 0

        # Start timing for this iteration
        start_time_reweighing = datetime.now()    

        unprivileged_groups = [{attribute: 0}]
        privileged_groups = [{attribute: 1}]

        categorical_columns = ['LEI', 'state_code']  # Add more as needed
        existing_columns_to_drop = [col for col in categorical_columns if col in df_train.columns]
        # Drop columns if they exist
        if existing_columns_to_drop:
            df_train = df_train.drop(columns=existing_columns_to_drop)
            df_test = df_test.drop(columns=existing_columns_to_drop)

        # Ensure necessary columns are present
        required_columns = PROTECTED_ATTRIBUTES + ['approved']
        missing_columns = [col for col in required_columns if col not in df_train.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in df_train: {missing_columns}")

        # Ensure labels and protected attributes are correctly formatted
        df_train['approved'] = df_train['approved'].astype(int)
        df_test['approved'] = df_test['approved'].astype(int)

        for col in PROTECTED_ATTRIBUTES:
            if df_train[col].dtype != int and df_train[col].dtype != float:
                df_train[col] = df_train[col].astype(float)
            if df_test[col].dtype != int and df_test[col].dtype != float:
                df_test[col] = df_test[col].astype(float)

        df_train_bld = BinaryLabelDataset(df=df_train,
                                          label_names=['approved'],
                                          protected_attribute_names=[attribute],
                                          favorable_label=1, unfavorable_label=0)

        df_test_bld = BinaryLabelDataset(df=df_test,
                                         label_names=['approved'],
                                         protected_attribute_names=[attribute],
                                         favorable_label=1, unfavorable_label=0)

        # Apply DisparateImpactRemover
        #Open item-5 After review changes starts
        start_time_orig_dir = datetime.now()
        dir_remover = DisparateImpactRemover(repair_level=1.0)
        df_train_dir = dir_remover.fit_transform(df_train_bld)
        df_test_dir = dir_remover.fit_transform(df_test_bld)

        # Convert back to pandas DataFrame if needed
        df_train_dir = df_train_dir.convert_to_dataframe()[0]
        df_test_dir = df_test_dir.convert_to_dataframe()[0]

        # Convert DIR-transformed datasets to BinaryLabelDataset
        df_train_dir_bld = BinaryLabelDataset(df=df_train_dir, label_names=['approved'], protected_attribute_names=[attribute], favorable_label=1, unfavorable_label=0)
        df_test_dir_bld = BinaryLabelDataset(df=df_test_dir, label_names=['approved'], protected_attribute_names=[attribute], favorable_label=1, unfavorable_label=0)

        # Sub Function E1-Apply Disparate Impact Remover
        df_train_unbiased_bld = apply_dir(df_train_bld, unprivileged_groups, privileged_groups, repair_level=1.0)
        df_test_unbiased_bld = apply_dir(df_test_bld, unprivileged_groups, privileged_groups, repair_level=1.0)

        # Sub Function E2-Apply Reweighing
        df_train_unbiased_bld, train_metrics = reweighing(df_train_bld, unprivileged_groups, privileged_groups)

        df_test_unbiased_bld, test_metrics = reweighing(bld=df_test_bld,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups,
                                                        training=False)

        # Compute fairness metrics after DIR
        train_metrics = {
        "Statistical parity difference": BinaryLabelDatasetMetric(df_train_unbiased_bld, unprivileged_groups, privileged_groups).statistical_parity_difference(),
        "Disparate impact": BinaryLabelDatasetMetric(df_train_unbiased_bld, unprivileged_groups, privileged_groups).disparate_impact(),
        }

        test_metrics = {
        "Statistical parity difference": BinaryLabelDatasetMetric(df_test_unbiased_bld, unprivileged_groups, privileged_groups).statistical_parity_difference(),
        "Disparate impact": BinaryLabelDatasetMetric(df_test_unbiased_bld, unprivileged_groups, privileged_groups).disparate_impact(),
        "Mean Difference": BinaryLabelDatasetMetric(df_test_unbiased_bld, unprivileged_groups, privileged_groups).mean_difference(),
        "Statistical parity difference (Reweighted)": train_metrics.get("Statistical parity difference (Reweighted)", 0),
        "Disparate impact (Reweighted)": train_metrics.get("Disparate impact (Reweighted)", 0),
        "Mean difference (Reweighted)": train_metrics.get("Mean difference (Reweighted)", 0),
        }
        end_time_orig_dir = datetime.now()
        execution_time_dir_orig = (end_time_orig_dir - start_time_orig_dir).total_seconds() / 3600  # T in hours
        energy_dir_orig = (((execution_time_dir_orig / 24) / 365) * 57754)/1000
        #Open item-5 After review changes ends

        # End timing for this iteration
        end_time_reweighing = datetime.now()
        execution_time_reweighing = (end_time_reweighing - start_time_reweighing).total_seconds() / 3600  # T in hours

        # Formula for computing Energy, Assuming T is in hours Assume 10 CPU cores and 1 GPU for A100
        energy_reweighing = (((execution_time_reweighing / 24) / 365) * 57754)/1000

        # Start timing for this iteration
        start_time_orig = datetime.now()

        scaler = StandardScaler()
        X_train_orig = df_train[columns_to_retain]
        scaler.fit(X_train_orig)  # Fit the scaler on training data
        y_train_orig = df_train['approved']

        X_test_orig = df_test[columns_to_retain]
        y_test_orig = df_test['approved']

        # Reweighed training/testing data
        train_unbiased_df = df_train_unbiased_bld.convert_to_dataframe()[0]
        X_train_unbiased = df_train_unbiased_bld.convert_to_dataframe()[0][columns_to_retain]
        y_train_unbiased = df_train_unbiased_bld.labels.ravel()

        test_unbiased_df = df_test_unbiased_bld.convert_to_dataframe()[0]
        X_test_unbiased = df_test_unbiased_bld.convert_to_dataframe()[0][columns_to_retain]
        y_test_unbiased = df_test_unbiased_bld.labels.ravel()

        print("Shape of df_train_unbiased_bld.features:", df_train_unbiased_bld.features.shape)

        input_dim = df_train_unbiased_bld.features.shape[1]
        print("Updated input_dim based on unbiased dataset:", input_dim)

        input_dim = len(columns_to_retain)

        # Update df_train_unbiased_bld and df_test_unbiased_bld to contain only the relevant columns
        df_train_unbiased_bld.features = df_train_unbiased_bld.features[:, :input_dim]
        df_test_unbiased_bld.features = df_test_unbiased_bld.features[:, :input_dim]

        print("Updated input_dim:", input_dim)
        print("Shape of df_train_unbiased_bld.features after selecting relevant columns:", df_train_unbiased_bld.features.shape)

        # Convert to DataLoader for unbiased data with correct input dimensions
        train_loader_unbiased_nn = DataLoader(
        TensorDataset(
        torch.tensor(df_train_unbiased_bld.features, dtype=torch.float32).to(device),
        torch.tensor(df_train_unbiased_bld.labels, dtype=torch.float32).to(device)
        ),
        batch_size=64, shuffle=True
        )

        test_loader_unbiased_nn = DataLoader(
        TensorDataset(
        torch.tensor(df_test_unbiased_bld.features, dtype=torch.float32).to(device),
        torch.tensor(df_test_unbiased_bld.labels, dtype=torch.float32).to(device)
        ),
        batch_size=64, shuffle=False
        )

        # Train and test the original model
        if verbose:
            # Slow: runs on CPU because GPU version doesn't have feature importance
            importance_rf = ske.RandomForestClassifier(n_estimators=120, random_state=np.random.seed(11850))
            importance_rf.fit(X_train_orig, pd.Series(y_train_orig))
            importances = importance_rf.feature_importances_
            sorted_idx = importances.argsort()

            print('\n===========================')
            print('Original Feature Importance')
            print('===========================')
            for feature, val in zip(X_train_orig.columns[sorted_idx], importances[sorted_idx]):
                print('Feature: {}, Score: {:.4f}%'.format(feature, val * 100))

            print('===========================\n')

        # Convert Pandas DataFrame to cuDF for GPU acceleration
        X_train_orig_gdf = cudf.DataFrame(X_train_orig)
        y_train_orig_gdf = cudf.Series(y_train_orig)

        X_test_orig_gdf = cudf.DataFrame(X_test_orig)

        # Initialize GPU-based RandomForest with best parameters
        rf_gpu = cuRF(n_estimators=100, max_depth=10, n_bins=16)

        # Train on GPU
        rf_gpu.fit(X_train_orig_gdf, y_train_orig_gdf)

        # Make Predictions
        pred_train_orig = rf_gpu.predict(X_train_orig_gdf)
        pred_test_orig = rf_gpu.predict(X_test_orig_gdf)

        # Convert predictions back to NumPy (needed for sklearn metrics)
        pred_train_orig_np = pred_train_orig.to_numpy()
        pred_test_orig_np = pred_test_orig.to_numpy()

        predictions1 = cp.asnumpy(pred_train_orig)
        predictions2 = cp.asnumpy(pred_test_orig)

        # Sub Function E3-Train and Evaluate Logistic Regression
        #Open item-5 After review changes starts
        start_time_orig_lr = datetime.now()
        clf_lr = train_logistic_regression(X_train_orig, y_train_orig)
        pred_train_lr = clf_lr.predict(X_train_orig)
        pred_test_lr = clf_lr.predict(X_test_orig)

        # Calculate Metrics for Logistic Regression
        print('The Original Training Accuracy Score (LR): ', accuracy_score(y_train_orig, pred_train_lr))
        print('The Original Testing Accuracy Score (LR): ', accuracy_score(y_test_orig, pred_test_lr))

        print('The Original Training F1 Score (LR): ', f1_score(y_train_orig, pred_train_lr))
        print('The Original Testing F1 Score (LR): ', f1_score(y_test_orig, pred_test_lr))

        print('The Original Training Recall Score (LR): ', recall_score(y_train_orig, pred_train_lr))
        print('The Original Testing Recall Score (LR): ', recall_score(y_test_orig, pred_test_lr))

        print('The Original Training Precision Score (LR): ', precision_score(y_train_orig, pred_train_lr))
        print('The Original Testing Precision Score (LR): ', precision_score(y_test_orig, pred_test_lr))

        del X_train_orig_gdf, y_train_orig_gdf, X_test_orig_gdf

        print('The Original Training Accuracy Score: ', accuracy_score(y_train_orig, predictions1))
        print('The Original Testing Accuracy Score: ', accuracy_score(y_test_orig, predictions2))

        print('The Original Training F1 Score: ', f1_score(y_train_orig, predictions1))
        print('The Original Testing F1 Score: ', f1_score(y_test_orig, predictions2))

        print('The Original Training Recall Score: ', recall_score(y_train_orig, predictions1))
        print('The Original Testing Recall Score: ', recall_score(y_test_orig, predictions2))

        print('The Original Training Precision Score: ', precision_score(y_train_orig, predictions1))
        print('The Original Testing Precision Score: ', precision_score(y_test_orig, predictions2))

        # Sub Function E4-Calculate fairness metrics for Logistic Regression
        lr_fairness_metrics = calculate_lr_fairness_metrics(df_train_bld, pred_train_lr, df_test_bld, pred_test_lr, unprivileged_groups, 
            privileged_groups, df_train_reweighted=df_train_unbiased_bld, df_test_reweighted=df_test_unbiased_bld)

        # Print Fairness Metrics for LR
        print("\n#### Logistic Regression Training Fairness Metrics:")
        for key, value in lr_fairness_metrics["train_metrics"].items():
            print(f"{key} (LR Train) = {value:.4f}")

        print("\n#### Logistic Regression Testing Fairness Metrics:")
        for key, value in lr_fairness_metrics["test_metrics"].items():
            print(f"{key} (LR Test) = {value:.4f}")
        end_time_orig_lr = datetime.now()
        execution_time_orig_lr = (end_time_orig_lr - start_time_orig_lr).total_seconds() / 3600  # T in hours
        energy_orig_lr = (((execution_time_orig_lr / 24) / 365) * 57754)/1000
        #Open item-5 After review changes ends

        df_train_bld_pred = df_train_bld.copy()
        df_train_bld_pred.labels = np.reshape(predictions1, (-1, 1))

        # Sub Function E5-Compute metrics
        metric_train_orig = compute_metrics(df_train_bld, df_train_bld_pred,
                                            unprivileged_groups, privileged_groups,
                                            disp=True)

        df_test_bld_pred = df_test_bld.copy()
        df_test_bld_pred.labels = np.reshape(predictions2, (-1, 1))

        metric_test_orig = compute_metrics(df_test_bld, df_test_bld_pred,
                                           unprivileged_groups, privileged_groups,
                                           disp=True)
        # End timing for this iteration
        end_time_orig = datetime.now()
        execution_time_orig = (end_time_orig - start_time_orig).total_seconds() / 3600  # T in hours

        # Formula for computing Energy, Assuming T is in hours Assume 10 CPU cores and 1 GPU for A100
        energy_orig = (((execution_time_orig / 24) / 365) * 57754)/1000


        # Start timing for this iteration
        start_time_unbiased = datetime.now()

        # Train and test the reweighted (unbiased) model
        X_train_unbiased_gdf = cudf.from_pandas(X_train_unbiased)
        y_train_unbiased_cp = cudf.Series(y_train_unbiased)

        if verbose:
            # Slow: runs on CPU because GPU version doesn't have feature importance
            importance_rf = ske.RandomForestClassifier(n_estimators=120, random_state=np.random.seed(11850))
            importance_rf.fit(X_train_unbiased, pd.Series(y_train_unbiased))
            importances = importance_rf.feature_importances_
            sorted_idx = importances.argsort()

            print('\n===========================')
            print('Unbiased Feature Importance')
            print('===========================')
            for feature, val in zip(X_train_unbiased.columns[sorted_idx], importances[sorted_idx]):
                print('Feature: {}, Score: {:.4f}%'.format(feature, val * 100))

            print('===========================\n')

        X_test_unbiased_gdf = cudf.from_pandas(X_test_unbiased)

        if trained_model is not None:
            print('Loading saved model {}...\n'.format(trained_model.name))
            best_rf_model_unbiased = load(trained_model.name)
        else:
            rf_param_grid = {
            'n_estimators': [100, 200],  # Reduced from [100, 200, 300] -> fewer models
            'max_depth': [10, 20],  # Removed `None`
            'min_samples_split': [2],  # Removed extra options
            'min_samples_leaf': [1, 2],  # Fewer choices
            'max_features': ['sqrt']  # Removed 'log2'
            }

            # Convert training data to Pandas for compatibility with GridSearchCV
            X_train_unbiased_pd = X_train_unbiased.copy()
            y_train_unbiased_pd = pd.Series(y_train_unbiased)

            # Initialize RandomForestClassifier and perform grid search
            rf_model = ske.RandomForestClassifier(random_state=42)
            grid_search_unbiased = RandomizedSearchCV(
            estimator=rf_model, param_distributions=rf_param_grid, 
            n_iter=10,  # Try only 10 random models instead of 108
            cv=3,  # Use 3-fold instead of 5-fold
            scoring='accuracy', n_jobs=-1, verbose=1, random_state=42
            )
            grid_search_unbiased.fit(X_train_unbiased_pd, y_train_unbiased_pd)

            # Retrieve the best model and hyperparameters
            best_rf_model_unbiased = grid_search_unbiased.best_estimator_
            print(f"Best Random Forest Parameters (Unbiased): {grid_search_unbiased.best_params_}")

            # Fit the best model on the GPU data (converted to Pandas for compatibility)
            best_rf_model_unbiased.fit(X_train_unbiased_gdf.to_pandas(), y_train_unbiased_cp.to_pandas())

        # Make predictions using the best model
        pred_train_unbiased = best_rf_model_unbiased.predict(X_train_unbiased_gdf.to_pandas())
        pred_test_unbiased = best_rf_model_unbiased.predict(X_test_unbiased_gdf.to_pandas())

        # Convert predictions to numpy arrays for further evaluation
        predictions1_unbiased = cp.asnumpy(pred_train_unbiased)
        predictions2_unbiased = cp.asnumpy(pred_test_unbiased)

        # End timing for this iteration
        end_time_unbiased = datetime.now()
        execution_time_unbiased = (end_time_unbiased - start_time_unbiased).total_seconds() / 3600  # T in hours

        # Formula for computing Energy, Assuming T is in hours Assume 10 CPU cores and 1 GPU for A100
        energy_unbiased = (((execution_time_unbiased / 24) / 365) * 57754)/1000

        total_energy_unbiased=energy_reweighing+energy_unbiased
        total_energy_orig=energy_orig
        total_execution_time_unbiased=execution_time_reweighing+execution_time_unbiased
        total_execution_time_orig=execution_time_orig

        # Compute fairness metrics for the original model
        df_train_bld_pred = df_train_bld.copy()
        df_train_bld_pred.labels = np.reshape(predictions1, (-1, 1))
        metric_train_orig = compute_metrics(df_train_bld, df_train_bld_pred, unprivileged_groups, privileged_groups, disp=False)

        df_test_bld_pred = df_test_bld.copy()
        df_test_bld_pred.labels = np.reshape(predictions2, (-1, 1))
        metric_test_orig = compute_metrics(df_test_bld, df_test_bld_pred, unprivileged_groups, privileged_groups, disp=False)

        # Compute fairness metrics for the reweighted (unbiased) model
        df_train_unbiased_bld_pred = df_train_unbiased_bld.copy()
        df_train_unbiased_bld_pred.labels = np.reshape(predictions1_unbiased, (-1, 1))
        metric_train_unbiased = compute_metrics(df_train_unbiased_bld, df_train_unbiased_bld_pred, unprivileged_groups, privileged_groups, disp=False)
        metric_train_dir = compute_metrics(df_train_unbiased_bld, df_train_unbiased_bld_pred, unprivileged_groups, privileged_groups, disp=False)

        df_test_unbiased_bld_pred = df_test_unbiased_bld.copy()
        df_test_unbiased_bld_pred.labels = np.reshape(predictions2_unbiased, (-1, 1))
        metric_test_unbiased = compute_metrics(df_test_unbiased_bld, df_test_unbiased_bld_pred, unprivileged_groups, privileged_groups, disp=False)
        metric_test_dir = compute_metrics(df_test_unbiased_bld, df_test_unbiased_bld_pred, unprivileged_groups, privileged_groups, disp=False)

        #Neural Network Changes starts
        # Convert to PyTorch Tensors for Neural Network
        train_data = torch.tensor(df_train[columns_to_retain].values, dtype=torch.float32).to(device)
        train_labels = torch.tensor(df_train['approved'].values, dtype=torch.float32).to(device)
        test_data = torch.tensor(df_test[columns_to_retain].values, dtype=torch.float32).to(device)
        test_labels = torch.tensor(df_test['approved'].values, dtype=torch.float32).to(device)

        # Sub Function E6-Train Neural Network on Original Data
        approval_nn = LoanApprovalNN(input_dim=train_data.shape[1])
        # Ensure neural network model is on GPU
        approval_nn = approval_nn.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(approval_nn.parameters(), lr=0.001)
        train_neural_network(approval_nn, DataLoader(TensorDataset(train_data, train_labels), batch_size=64), criterion, optimizer)

        # Predict and compute fairness metrics on original data
        nn_train_pred, nn_train_labels = evaluate_neural_network(approval_nn, DataLoader(TensorDataset(train_data, train_labels), batch_size=64))
        nn_test_pred, nn_test_labels = evaluate_neural_network(approval_nn, DataLoader(TensorDataset(test_data, test_labels), batch_size=64))

        # Create BinaryLabelDataset for predictions
        df_train_bld_pred_nn = df_train_bld.copy()
        df_test_bld_pred_nn = df_test_bld.copy()

        # Ensure NN predictions are properly thresholded and reshaped
        if isinstance(nn_train_pred, torch.Tensor):
            nn_train_pred = (nn_train_pred.cpu().numpy() >= 0.5).astype(int).reshape((-1, 1))
        else:
            nn_train_pred = (np.array(nn_train_pred) >= 0.5).astype(int).reshape((-1, 1))
        df_train_bld_pred_nn.labels = nn_train_pred
        metric_train_nn = compute_metrics(df_train_bld, df_train_bld_pred_nn, unprivileged_groups, privileged_groups, disp=False)

        # Similarly for test predictions
        if isinstance(nn_test_pred, torch.Tensor):
            nn_test_pred = (nn_test_pred.cpu().numpy() >= 0.5).astype(int).reshape((-1, 1))
        else:
            nn_test_pred = (np.array(nn_test_pred) >= 0.5).astype(int).reshape((-1, 1))
        df_test_bld_pred_nn.labels = nn_test_pred
        metric_test_nn = compute_metrics(df_test_bld, df_test_bld_pred_nn, unprivileged_groups, privileged_groups, disp=False)

        # Neural Network Predictions on Reweighted Data
        nn_train_pred_unbiased, nn_train_labels_unbiased = evaluate_neural_network(approval_nn, DataLoader(TensorDataset(torch.tensor(df_train_unbiased_bld.features, dtype=torch.float32), torch.tensor(df_train_unbiased_bld.labels, dtype=torch.float32)), batch_size=64))
        nn_test_pred_unbiased, nn_test_labels_unbiased = evaluate_neural_network(approval_nn, DataLoader(TensorDataset(torch.tensor(df_test_unbiased_bld.features, dtype=torch.float32), torch.tensor(df_test_unbiased_bld.labels, dtype=torch.float32)), batch_size=64))

        # End timing for unbiased NN execution
        end_time_unbiased_nn = datetime.now()
        execution_time_unbiased_nn = 1 # should be (end_time_unbiased_nn - start_time_unbiased_nn).total_seconds() / 3600  # in hours

        # Energy consumption calculation (kWh)
        execution_time_orig_nn = 1 # for temporary
        energy_orig_nn = (((execution_time_orig_nn / 24) / 365) * 57754) / 1000
        energy_unbiased_nn = (((execution_time_unbiased_nn / 24) / 365) * 57754) / 1000

        # Convert reweighted predictions to BinaryLabelDataset format for fairness metrics
        df_train_unbiased_bld_pred_nn = df_train_unbiased_bld.copy()
        df_train_unbiased_bld_pred_nn.labels = np.reshape(nn_train_pred_unbiased, (-1, 1))
        df_test_unbiased_bld_pred_nn = df_test_unbiased_bld.copy()
        df_test_unbiased_bld_pred_nn.labels = np.reshape(nn_test_pred_unbiased, (-1, 1))

        # Calculate fairness metrics on the reweighted training and testing data
        metric_train_unbiased_nn = compute_metrics(df_train_unbiased_bld, df_train_unbiased_bld_pred_nn, unprivileged_groups, privileged_groups, disp=False)
        metric_test_unbiased_nn = compute_metrics(df_test_unbiased_bld, df_test_unbiased_bld_pred_nn, unprivileged_groups, privileged_groups, disp=False)

        #Open item-1 After review changes starts
        # Calculate Unbiased Logistic Regression Metrics
        #Open item-5 After review changes starts
        start_time_reweighted_lr = datetime.now()
        clf_lr_unbiased = train_logistic_regression(X_train_unbiased, y_train_unbiased)
        pred_train_unbiased_lr = clf_lr_unbiased.predict(X_train_unbiased)
        pred_test_unbiased_lr = clf_lr_unbiased.predict(X_test_unbiased)

        # Convert unbiased predictions to BinaryLabelDataset format for AIF360
        df_train_unbiased_bld_pred_lr = df_train_unbiased_bld.copy()
        df_train_unbiased_bld_pred_lr.labels = pred_train_unbiased_lr.reshape(-1, 1)

        df_test_unbiased_bld_pred_lr = df_test_unbiased_bld.copy()
        df_test_unbiased_bld_pred_lr.labels = pred_test_unbiased_lr.reshape(-1, 1)

        # Calculate AIF360 fairness metrics on the unbiased training dataset
        metric_train_unbiased_lr = compute_metrics(df_train_unbiased_bld, df_train_unbiased_bld_pred_lr, unprivileged_groups, privileged_groups, disp=False)

        # Calculate AIF360 fairness metrics on the unbiased testing dataset
        metric_test_unbiased_lr = compute_metrics(df_test_unbiased_bld, df_test_unbiased_bld_pred_lr, unprivileged_groups, privileged_groups, disp=False)
        print("Available keys in metric_train_unbiased_lr:", metric_train_unbiased_lr.keys())
        print("Metric Train Unbiased LR Output:", metric_train_unbiased_lr)
        #Open item-1 After review changes ends
        end_time_reweighted_lr = datetime.now()
        execution_time_reweighted_lr = (end_time_reweighted_lr - start_time_reweighted_lr).total_seconds() / 3600  # T in hours
        energy_reweighted_lr = (((execution_time_reweighted_lr / 24) / 365) * 57754)/1000
        #Open item-5 After review changes ends

        # Sub Function E7-Train the Logistic Regression model on the training set (denied loans only)
        clf_lr_denial = train_logistic_regression_denial(X_train_denial, y_train_denial_reason)

        # Predict denial reasons using Logistic Regression
        lr_denial_train_pred = clf_lr_denial.predict(X_train_denial)
        lr_denial_test_pred = clf_lr_denial.predict(X_test_denial)

        # Calculate Accuracy Scores for Logistic Regression
        lr_denial_train_accuracy = accuracy_score(y_train_denial_reason, lr_denial_train_pred)
        lr_denial_test_accuracy = accuracy_score(y_test_denial_reason, lr_denial_test_pred)

        repaired_metrics = {
        "RF Training Accuracy (Repaired)": accuracy_score(y_train_unbiased, predictions1_unbiased),
        "RF Testing Accuracy (Repaired)": accuracy_score(y_test_unbiased, predictions2_unbiased),
        "NN Training Accuracy (Repaired)": accuracy_score(nn_train_labels_unbiased, nn_train_pred_unbiased),
        "NN Testing Accuracy (Repaired)": accuracy_score(nn_test_labels_unbiased, nn_test_pred_unbiased),
        "LR Training Accuracy (Repaired)": accuracy_score(y_train_unbiased, pred_train_unbiased_lr),
        "LR Testing Accuracy (Repaired)": accuracy_score(y_test_unbiased, pred_test_unbiased_lr),
        }

        #new-model starts
        # After reweighing and disparate impact processing
        # Generate predictions for reweighed and disparate impact datasets

        # For Neural Network (NN) with Reweighing
        #Open item-5 After review changes starts
        start_time_model4_9 = datetime.now()
        nn_reweighing_model = LoanApprovalNN(input_dim=X_train_unbiased.shape[1])
        nn_reweighing_model = train_neural_network(nn_reweighing_model, train_loader_unbiased_nn, criterion, optimizer)
        pred_test_nn_reweighing, _ = evaluate_neural_network(nn_reweighing_model, test_loader_unbiased_nn, binary=True)

        end_time_model4_9 = datetime.now()
        execution_time_model4_9 = (end_time_model4_9 - start_time_model4_9).total_seconds() / 3600  # T in hours
        energy_model4_9 = (((execution_time_model4_9 / 24) / 365) * 57754)/1000
        #Open item-5 After review changes ends

        # For Random Forest (RF) with Reweighing
        #Open item-5 After review changes starts
        start_time_model10_15 = datetime.now()
        rf_reweighing_model = ske.RandomForestClassifier(n_estimators=120, random_state=np.random.seed(11850))
        rf_reweighing_model.fit(X_train_unbiased, y_train_unbiased)
        pred_test_rf_reweighing = rf_reweighing_model.predict(X_test_unbiased)

        end_time_model10_15 = datetime.now()
        execution_time_model10_15 = (end_time_model10_15 - start_time_model10_15).total_seconds() / 3600  # T in hours
        energy_model10_15 = (((execution_time_model10_15 / 24) / 365) * 57754)/1000
        #Open item-5 After review changes ends

        # For Logistic Regression (LR) with Reweighing
        #Open item-5 After review changes starts
        start_time_model16_21 = datetime.now()
        lr_reweighing_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_reweighing_model.fit(X_train_unbiased, y_train_unbiased)
        pred_test_lr_reweighing = lr_reweighing_model.predict(X_test_unbiased)

        end_time_model16_21 = datetime.now()
        execution_time_model16_21 = (end_time_model16_21 - start_time_model16_21).total_seconds() / 3600  # T in hours
        energy_model16_21 = (((execution_time_model16_21 / 24) / 365) * 57754)/1000
        #Open item-5 After review changes ends

        # For Neural Network (NN) with Disparate Impact
        #Open item-5 After review changes starts
        start_time_model22_27 = datetime.now()
        nn_disparate_impact_model = LoanApprovalNN(input_dim=X_train_unbiased.shape[1])
        nn_disparate_impact_model = train_neural_network(nn_disparate_impact_model, train_loader_unbiased_nn, criterion, optimizer)
        pred_test_nn_disparate_impact, _ = evaluate_neural_network(nn_disparate_impact_model, test_loader_unbiased_nn, binary=True)

        end_time_model22_27 = datetime.now()
        execution_time_model22_27 = (end_time_model22_27 - start_time_model22_27).total_seconds() / 3600  # T in hours
        energy_model22_27 = (((execution_time_model22_27 / 24) / 365) * 57754)/1000
        #Open item-5 After review changes ends

        # For Random Forest (RF) with Disparate Impact
        #Open item-5 After review changes starts
        start_time_model28_33 = datetime.now()
        rf_disparate_impact_model = ske.RandomForestClassifier(n_estimators=120, random_state=np.random.seed(11850))
        rf_disparate_impact_model.fit(X_train_unbiased, y_train_unbiased)
        pred_test_rf_disparate_impact = rf_disparate_impact_model.predict(X_test_unbiased)

        end_time_model28_33 = datetime.now()
        execution_time_model28_33 = (end_time_model28_33 - start_time_model28_33).total_seconds() / 3600  # T in hours
        energy_model28_33 = (((execution_time_model28_33 / 24) / 365) * 57754)/1000
        #Open item-5 After review changes ends

        # For Logistic Regression (LR) with Disparate Impact
        #Open item-5 After review changes starts
        start_time_model34_39 = datetime.now()
        lr_disparate_impact_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_disparate_impact_model.fit(X_train_unbiased, y_train_unbiased)
        pred_test_lr_disparate_impact = lr_disparate_impact_model.predict(X_test_unbiased)

        end_time_model34_39 = datetime.now()
        execution_time_model34_39 = (end_time_model34_39 - start_time_model34_39).total_seconds() / 3600  # T in hours
        energy_model34_39 = (((execution_time_model34_39 / 24) / 365) * 57754)/1000
        #Open item-5 After review changes ends

        # Sub Function E8-Assign values to results_dict
        assign_results_dict(results_dict, train_metrics, metric_train_nn, lr_fairness_metrics, test_metrics, metric_test_nn, metric_train_dir, 
        metric_test_dir, metric_train_unbiased, metric_test_unbiased, metric_train_unbiased_nn, metric_test_unbiased_nn, metric_train_unbiased_lr, metric_test_unbiased_lr, y_train_orig, predictions1, 
        y_test_orig, predictions2, nn_train_labels, nn_train_pred, nn_test_labels, nn_test_pred, pred_train_lr, pred_test_lr, repaired_metrics, metric_train_orig, 
        metric_test_orig, y_train_unbiased, predictions1_unbiased, y_test_unbiased, predictions2_unbiased, nn_train_labels_unbiased, nn_train_pred_unbiased, 
        nn_test_labels_unbiased, nn_test_pred_unbiased, pred_train_unbiased_lr, pred_test_unbiased_lr, rf_denial_train_accuracy, rf_denial_test_accuracy, 
        nn_denial_train_accuracy, nn_denial_test_accuracy, lr_denial_train_accuracy, lr_denial_test_accuracy, total_execution_time_unbiased, total_execution_time_orig, 
        total_energy_unbiased, total_energy_orig, execution_time_unbiased_nn, execution_time_orig_nn, energy_unbiased_nn, energy_orig_nn, 
        #Open item-5 After review changes starts
        execution_time_orig_lr, energy_orig_lr, execution_time_reweighted_lr, energy_reweighted_lr, execution_time_dir_orig, energy_dir_orig,
        execution_time_model4_9, energy_model4_9, execution_time_model10_15, energy_model10_15, execution_time_model16_21, energy_model16_21,
        execution_time_model22_27, energy_model22_27, execution_time_model28_33, energy_model28_33, execution_time_model34_39, energy_model34_39)
        #Open item-5 After review changes ends

        # size_tracker.loc[len(size_tracker)] = [f'After process_protected_attribute: {attribute}', 
        #                                      preprocessed_df.shape[0], 
        #                                      preprocessed_df.shape[1]]
        print(f"df_train size: {df_train.shape[0]}")
        print(f"df_test size: {df_test.shape[0]}")
        print(f"After applying process_protected_attribute: {attribute}, preprocessed_df size: {preprocessed_df.shape[0]}")
        print(f"Current DataFrame shape: {preprocessed_df.shape}")

    # Sub Function E9-Call add_nn_results_to_csv after pred_test_lr is initialized
    add_nn_results_to_csv('preprocess_full_added_denialreasons.csv', approval_nn, scaler, columns_to_retain, denial_conditions, pred_test_lr, pred_test_nn_reweighing, 
            pred_test_rf_reweighing, pred_test_lr_reweighing, pred_test_nn_disparate_impact, pred_test_rf_disparate_impact, pred_test_lr_disparate_impact)

    return results_dict, execution_time_reweighing, execution_time_orig, execution_time_unbiased, energy_reweighing, energy_orig, energy_unbiased

#Sub Function E1
@nvtx.annotate("apply_dir", color="blue")
def apply_dir(dataset, unprivileged_groups, privileged_groups, repair_level=1.0):
    """
    Apply Disparate Impact Remover (DIR) to mitigate bias.
    
    Args:
        dataset (BinaryLabelDataset): The dataset to process.
        unprivileged_groups (list(dict)): Unprivileged groups for fairness.
        privileged_groups (list(dict)): Privileged groups for fairness.
        repair_level (float): Repair level for bias mitigation (0 to 1).
    
    Returns:
        BinaryLabelDataset: The dataset after applying DIR.
    """
    # Apply Disparate Impact Remover
    dir_remover = DisparateImpactRemover(repair_level=repair_level)
    processed_dataset = dir_remover.fit_transform(dataset)

    # Compute fairness metrics before and after applying DIR
    metric_before = BinaryLabelDatasetMetric(dataset, unprivileged_groups, privileged_groups)
    metric_after = BinaryLabelDatasetMetric(processed_dataset, unprivileged_groups, privileged_groups)
    print("\nFairness metrics BEFORE applying DIR:")
    print(f"Statistical Parity Difference: {metric_before.statistical_parity_difference():.4f}")
    print("\nFairness metrics AFTER applying DIR:")
    print(f"Statistical Parity Difference: {metric_after.statistical_parity_difference():.4f}")

    return processed_dataset

#Sub Function E2
@nvtx.annotate("reweighing", color="blue")
def reweighing(bld, unprivileged_groups, privileged_groups, training=True):
    """Reweighing is a preprocessing technique that weights the examples in each (group, label)
    combination differently to ensure fairness before classification.

    Args:
        bld (BinaryLabelDataset): Pandas Dataframe
        privileged_groups (list(dict)): Privileged groups. Format is a list
                of `dicts` where the keys are `protected_attribute_names` and
                the values are values in `protected_attributes`. Each `dict`
                element describes a single group. See examples for more details.
        unprivileged_groups (list(dict)): Unprivileged groups in the same
                format as `privileged_groups`.
        training (bool): True if df is the training data, otherwise False. True is default.
    Returns:
        BinaryLabelDataset: Reweighted dataset using the AIF360 reweighing pre-processing algorithm for
                            bias mitigation.
    """
    metric_orig_train = BinaryLabelDatasetMetric(bld,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

    text_explainer_train = MetricTextExplainer(metric_orig_train)

    if training:
        print("\n#### Original training dataset")
    else:
        print("\n#### Original testing dataset")

    # Difference in the probability of favorable outcomes between the unprivileged and
    # privileged groups. A value of 0 implies both groups have equal benefit, a value less
    # than 0 implies higher benefit for the privileged group, and a value greater than 0
    # implies higher benefit for the unprivileged group.
    print(text_explainer_train.statistical_parity_difference())
    spd = metric_orig_train.statistical_parity_difference()

    # The ratio in the probablity of favorable outcomes between the unprivileged adn privileged groups.
    # A value of 1 implies both groups have equal benefit, a value less than 1 implies higher benefit
    # for the privileged group, and a value greater than 1 implies higher benefit for the unprivileged group.
    print(text_explainer_train.disparate_impact())
    di = metric_orig_train.disparate_impact()

    # Mean difference (mean label value on unprivileged instances - mean label value on privileged instances)
    print(text_explainer_train.mean_difference())
    md = metric_orig_train.mean_difference()

    # Mitigate bias by transforming the original training dataset
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)

    df_reweighing_train = RW.fit_transform(bld)
    metric_reweighing_train = BinaryLabelDatasetMetric(df_reweighing_train,
                                                       unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups)

    spd_reweighted = metric_reweighing_train.statistical_parity_difference()
    di_reweighted = metric_reweighing_train.disparate_impact()
    md_reweighted = metric_reweighing_train.mean_difference()

    metrics = OrderedDict()
    metrics["Statistical parity difference"] = spd
    metrics["Disparate impact"] = di
    metrics["Mean difference"] = md
    metrics["Statistical parity difference (Reweighted)"] = spd_reweighted
    metrics["Disparate impact (Reweighted)"] = di_reweighted
    metrics["Mean difference (Reweighted)"] = md_reweighted

    return df_reweighing_train, metrics

# Sub Function E3-Train Logistic Regression for Loan Approval Prediction
def train_logistic_regression(X_train, y_train):
    clf_lr = LogisticRegression(max_iter=1000, random_state=42)
    clf_lr.fit(X_train, y_train)
    return clf_lr

# Sub Function E4
def calculate_lr_fairness_metrics(
    df_train_bld, pred_train_lr, df_test_bld, pred_test_lr, 
    unprivileged_groups, privileged_groups, 
    df_train_reweighted=None, df_test_reweighted=None
):
    """
    Calculate AIF360 fairness metrics for Logistic Regression model, including reweighed metrics.

    Args:
        df_train_bld (BinaryLabelDataset): Training dataset in AIF360 format.
        pred_train_lr (numpy array): Predictions from the Logistic Regression model on the training dataset.
        df_test_bld (BinaryLabelDataset): Testing dataset in AIF360 format.
        pred_test_lr (numpy array): Predictions from the Logistic Regression model on the testing dataset.
        unprivileged_groups (list): Unprivileged groups for fairness calculation.
        privileged_groups (list): Privileged groups for fairness calculation.
        df_train_reweighted (BinaryLabelDataset): Reweighted training dataset in AIF360 format (optional).
        df_test_reweighted (BinaryLabelDataset): Reweighted testing dataset in AIF360 format (optional).

    Returns:
        dict: Fairness metrics for original and reweighted train and test datasets.
    """
    # Initialize the result dictionary
    fairness_metrics = {
        "train_metrics": {},
        "test_metrics": {},
        "train_metrics_reweighted": {},
        "test_metrics_reweighted": {}
    }

    # Predictions in BinaryLabelDataset format for original data
    df_train_bld_pred_lr = df_train_bld.copy()
    df_train_bld_pred_lr.labels = pred_train_lr.reshape(-1, 1)

    df_test_bld_pred_lr = df_test_bld.copy()
    df_test_bld_pred_lr.labels = pred_test_lr.reshape(-1, 1)

    # Compute original fairness metrics for train and test
    fairness_metrics["train_metrics"] = compute_metrics(
        df_train_bld, df_train_bld_pred_lr, 
        unprivileged_groups, privileged_groups, disp=False
    )

    fairness_metrics["test_metrics"] = compute_metrics(
        df_test_bld, df_test_bld_pred_lr, 
        unprivileged_groups, privileged_groups, disp=False
    )

    try:
        fairness_metrics["train_metrics"]["Mean difference"] = BinaryLabelDatasetMetric(
            df_train_bld_pred_lr, unprivileged_groups, privileged_groups
        ).mean_difference()
    except KeyError:
        fairness_metrics["train_metrics"]["Mean difference"] = 0  # Default to 0 if missing

    try:
        fairness_metrics["test_metrics"]["Mean difference"] = BinaryLabelDatasetMetric(
            df_test_bld_pred_lr, unprivileged_groups, privileged_groups
        ).mean_difference()
    except KeyError:
        fairness_metrics["test_metrics"]["Mean difference"] = 0  # Default to 0 if missing

    # Compute metrics for reweighted data if provided
    if df_train_reweighted and df_test_reweighted:
        df_train_reweighted_pred = df_train_reweighted.copy()
        df_train_reweighted_pred.labels = pred_train_lr.reshape(-1, 1)

        df_test_reweighted_pred = df_test_reweighted.copy()
        df_test_reweighted_pred.labels = pred_test_lr.reshape(-1, 1)

        fairness_metrics["train_metrics_reweighted"] = compute_metrics(
            df_train_reweighted, df_train_reweighted_pred, 
            unprivileged_groups, privileged_groups, disp=False
        )

        fairness_metrics["test_metrics_reweighted"] = compute_metrics(
            df_test_reweighted, df_test_reweighted_pred, 
            unprivileged_groups, privileged_groups, disp=False
        )

        # Include mean difference for reweighted data
        fairness_metrics["train_metrics_reweighted"]["Mean difference"] = BinaryLabelDatasetMetric(
            df_train_reweighted_pred, unprivileged_groups, privileged_groups
        ).mean_difference()

        fairness_metrics["test_metrics_reweighted"]["Mean difference"] = BinaryLabelDatasetMetric(
            df_test_reweighted_pred, unprivileged_groups, privileged_groups
        ).mean_difference()

    return fairness_metrics

# Sub Function E5
@nvtx.annotate("compute_metrics", color="blue")
def compute_metrics(dataset_true, dataset_pred,
                    unprivileged_groups, privileged_groups,
                    disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                  dataset_pred,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)
    metrics = OrderedDict()

    # The average of difference in false positive rates and true positive rates between unprivileged and privileged groups.
    # A value of 0 implies both groups have equal benefit, a value less than 0 implies higher benefit for
    # the privileged group and a value greater than 0 implies higher benefit for the unprivileged group.
    # Needs to use input and output datasets to a classifier (ClassificationMetric)
    metrics["Average odds difference"] = classified_metric_pred.average_odds_difference()

    metrics["Statistical parity difference"] = classified_metric_pred.statistical_parity_difference()

    # The difference in true positive rates between unprivileged and privileged groups. A value of 0 implies both groups
    # have equal benefit, a value less than 0 implies higher benefit for the privileged group and a value greater than 0
    # implies higher benefit for the unprivileged group.
    # Needs to use input and output datasets to a classifier (ClassificationMetric)
    metrics["Equal opportunity difference"] = classified_metric_pred.equal_opportunity_difference()

    # generalized_entropy_index with alpha = 1.
    metrics["Theil index"] = classified_metric_pred.theil_index()

    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()

# Try to compute "Mean difference", handle errors gracefully
    try:
        metrics["Mean difference"] = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups, privileged_groups).mean_difference()
    except Exception as e:
        print(f"Warning: Failed to compute 'Mean difference' due to {e}")
        metrics["Mean difference"] = 0  # Default to 0 if missing
    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))

    return metrics

#Sub Function E6
class LoanApprovalNN(nn.Module):
    def __init__(self, input_dim):
        super(LoanApprovalNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Batch normalization
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.output(x))
        return x

# Sub Function E7-Train Logistic Regression for Denial Reason Prediction
def train_logistic_regression_denial(X_train, y_train_denial_reason):
    # Ensure inputs are on CPU and converted to numpy arrays
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.cpu().numpy()
    if isinstance(y_train_denial_reason, torch.Tensor):
        y_train_denial_reason = y_train_denial_reason.cpu().numpy()
    
    clf_lr_denial = LogisticRegression(max_iter=1000, random_state=42)
    clf_lr_denial.fit(X_train, y_train_denial_reason)
    return clf_lr_denial

# Sub Function E8
def add_nn_results_to_csv(unbiased_dataset_path, approval_nn, scaler, columns_to_retain, denial_conditions, 
                          pred_test_lr, pred_test_nn_reweighing, pred_test_rf_reweighing, pred_test_lr_reweighing, 
                          pred_test_nn_disparate_impact, pred_test_rf_disparate_impact, pred_test_lr_disparate_impact):
    """
    Updates the dataset with predictions from various models and saves it to CSV.
    """
    # Load the dataset
    df = pd.read_csv(unbiased_dataset_path)

    # Convert lists to arrays if necessary
    predictions = [
        pred_test_lr, pred_test_nn_reweighing, pred_test_rf_reweighing, pred_test_lr_reweighing,
        pred_test_nn_disparate_impact, pred_test_rf_disparate_impact, pred_test_lr_disparate_impact
    ]
    
    for i, pred in enumerate(predictions):
        if isinstance(pred, list):
            predictions[i] = np.array(pred)
    
    (pred_test_lr, pred_test_nn_reweighing, pred_test_rf_reweighing, pred_test_lr_reweighing,
     pred_test_nn_disparate_impact, pred_test_rf_disparate_impact, pred_test_lr_disparate_impact) = predictions

    # Align all predictions with the DataFrame length
    pred_test_lr = align_predictions(pred_test_lr, len(df))
    pred_test_nn_reweighing = align_predictions(pred_test_nn_reweighing, len(df))
    pred_test_rf_reweighing = align_predictions(pred_test_rf_reweighing, len(df))
    pred_test_lr_reweighing = align_predictions(pred_test_lr_reweighing, len(df))
    pred_test_nn_disparate_impact = align_predictions(pred_test_nn_disparate_impact, len(df))
    pred_test_rf_disparate_impact = align_predictions(pred_test_rf_disparate_impact, len(df))
    pred_test_lr_disparate_impact = align_predictions(pred_test_lr_disparate_impact, len(df))

    # Add predictions to DataFrame
    df['Original_Approved_lr'] = pred_test_lr

    # Compute `denial_reason_final_lr`
    df['denial_reason_final_lr'] = df.apply(lambda row: 1 if row['Original_Approved_lr'] == 1 else (
        2 if any(row[col] == 1 for col in denial_conditions['debt_to_income_ratio']) else (
            3 if any(row[col] == 1 for col in denial_conditions['collateral']) else (
                4 if any(row[col] == 1 for col in denial_conditions['insufficient_cash']) else 5
            )
        )
    ), axis=1)

    # Standardize features for NN prediction
    X_test = df[columns_to_retain]
    X_test_scaled = scaler.transform(X_test)

    # Convert to tensor and predict using Neural Network
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32)), batch_size=64, shuffle=False
    )
    
    nn_predictions, _ = evaluate_neural_network(approval_nn, test_loader, binary=True)
    
    # Ensure NN predictions are formatted correctly
    nn_predictions = np.array(nn_predictions).flatten()
    nn_predictions = (nn_predictions >= 0.5).astype(int)  # Convert to binary 0/1
    nn_predictions = align_predictions(nn_predictions, len(df))  # Align NN predictions
    df['Original_Approved_nn'] = nn_predictions  # Add NN predictions

    # Add columns for reweighing and disparate impact models
    df = add_model4_9_columns(df, pred_test_nn_reweighing)
    df = add_model10_15_columns(df, pred_test_rf_reweighing)
    df = add_model16_21_columns(df, pred_test_lr_reweighing)
    df = add_model22_27_columns(df, pred_test_nn_disparate_impact)
    df = add_model28_33_columns(df, pred_test_rf_disparate_impact)
    df = add_model34_39_columns(df, pred_test_lr_disparate_impact)

    # Remove specified columns
    columns_to_remove = [
        'tract_minority_percentage40_0_approved', 'tract_minority_percentage50_0_approved',
        'tract_minority_percentage60_0_approved', 'tract_minority_percentage70_0_approved',
        'tract_minority_percentage80_0_approved', 'tract_minority_percentage90_0_approved',
        'Original_Approved_nn_reweighing', 'Original_Approved_rf_reweighing', 'Original_Approved_lr_reweighing',
        'Original_Approved_nn_disparate_impact', 'Original_Approved_rf_disparate_impact', 'Original_Approved_lr_disparate_impact'
    ]

    df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)

    # Save the updated dataset
    updated_path = unbiased_dataset_path.replace(".csv", "_updated.csv")
    df.to_csv(updated_path, index=False)
    print(f"Updated dataset saved to {updated_path}")

# Sub Function E9
def assign_results_dict(results_dict, train_metrics, metric_train_nn, lr_fairness_metrics, test_metrics, metric_test_nn, metric_train_dir, metric_test_dir, metric_train_unbiased, metric_test_unbiased, 
    metric_train_unbiased_nn, metric_test_unbiased_nn, metric_train_unbiased_lr, metric_test_unbiased_lr, y_train_orig, predictions1, y_test_orig, predictions2, nn_train_labels, nn_train_pred, nn_test_labels, 
    nn_test_pred, pred_train_lr, pred_test_lr, repaired_metrics, metric_train_orig, metric_test_orig, y_train_unbiased, predictions1_unbiased, y_test_unbiased, predictions2_unbiased, 
    nn_train_labels_unbiased, nn_train_pred_unbiased, nn_test_labels_unbiased, nn_test_pred_unbiased, pred_train_unbiased_lr, pred_test_unbiased_lr, rf_denial_train_accuracy, 
    rf_denial_test_accuracy, nn_denial_train_accuracy, nn_denial_test_accuracy, lr_denial_train_accuracy, lr_denial_test_accuracy, total_execution_time_unbiased, 
    total_execution_time_orig, total_energy_unbiased, total_energy_orig, execution_time_unbiased_nn, execution_time_orig_nn, energy_unbiased_nn, energy_orig_nn,
    #Open item-5 After review changes starts
    execution_time_orig_lr, energy_orig_lr, execution_time_reweighted_lr, energy_reweighted_lr, execution_time_dir_orig, energy_dir_orig,
    execution_time_model4_9, energy_model4_9, execution_time_model10_15, energy_model10_15, execution_time_model16_21, energy_model16_21,
    execution_time_model22_27, energy_model22_27, execution_time_model28_33, energy_model28_33, execution_time_model34_39, energy_model34_39
    #Open item-5 After review changes ends
    ):
    results_dict["#### Original training dataset"].append("")
    results_dict["RF Statistical parity difference (Original Train)"].append(train_metrics["Statistical parity difference"])
    results_dict["RF Disparate impact (Original Train)"].append(train_metrics["Disparate impact"])
    results_dict["RF Mean difference (Original Train)"].append(train_metrics.get("Mean difference", 0))
    results_dict["NN Statistical parity difference (Original Train)"].append(metric_train_nn["Statistical parity difference"])
    results_dict["NN Disparate impact (Original Train)"].append(metric_train_nn["Disparate impact"])
    results_dict["NN Mean difference (Original Train)"].append(metric_train_nn["Theil index"])
    results_dict["LR Statistical parity difference (Original Train)"].append(lr_fairness_metrics["train_metrics"]["Statistical parity difference"])
    results_dict["LR Disparate impact (Original Train)"].append(lr_fairness_metrics["train_metrics"]["Disparate impact"])
    results_dict["LR Mean difference (Original Train)"].append(lr_fairness_metrics["train_metrics"]["Mean difference"])
    results_dict["#### Reweighted training dataset"].append("")
    results_dict["RF Statistical parity difference (Reweighted Train)"].append(train_metrics.get("Statistical parity difference", 0))
    results_dict["RF Disparate impact (Reweighted Train)"].append(train_metrics.get("Disparate impact", 0))
    results_dict["RF Mean difference (Reweighted Train)"].append(train_metrics.get("Mean difference", 0))
    results_dict["NN Statistical parity difference (Reweighted Train)"].append(metric_train_unbiased_nn["Statistical parity difference"])
    results_dict["NN Disparate impact (Reweighted Train)"].append(metric_train_unbiased_nn["Disparate impact"])
    results_dict["NN Mean difference (Reweighted Train)"].append(metric_train_unbiased_nn["Theil index"])

    #Open item-1 After review changes starts
    results_dict["LR Statistical parity difference (Reweighted Train)"].append(lr_fairness_metrics["train_metrics_reweighted"].get("Statistical parity difference", 0))
    results_dict["LR Disparate impact (Reweighted Train)"].append(lr_fairness_metrics["train_metrics_reweighted"].get("Disparate impact", 0))
    results_dict["LR Mean difference (Reweighted Train)"].append(lr_fairness_metrics["train_metrics_reweighted"].get("Mean difference", 0))
    #Open item-1 After review changes ends

    results_dict["#### Repaired (DIR) training dataset"].append("")
    results_dict["RF Statistical parity difference (Repaired Train)"].append(metric_train_dir["Statistical parity difference"])
    results_dict["RF Disparate impact (Repaired Train)"].append(metric_train_dir["Disparate impact"])
    results_dict["RF Mean difference (Repaired Train)"].append(metric_train_dir["Mean difference"])
    results_dict["NN Statistical parity difference (Repaired Train)"].append(metric_train_unbiased_nn["Statistical parity difference"])
    results_dict["NN Disparate impact (Repaired Train)"].append(metric_train_unbiased_nn["Disparate impact"])
    results_dict["NN Mean difference (Repaired Train)"].append(metric_train_unbiased_nn["Theil index"])
    results_dict["LR Statistical parity difference (Repaired Train)"].append(metric_train_unbiased_lr["Statistical parity difference"])
    results_dict["LR Disparate impact (Repaired Train)"].append(metric_train_unbiased_lr["Disparate impact"])
    results_dict["LR Mean difference (Repaired Train)"].append(metric_train_unbiased_lr["Mean difference"])
    results_dict["#### Original testing dataset"].append("")
    results_dict["RF Statistical parity difference (Original Test)"].append(test_metrics["Statistical parity difference"])
    results_dict["RF Disparate impact (Original Test)"].append(test_metrics["Disparate impact"])
    results_dict["RF Mean difference (Original Test)"].append(test_metrics.get("Mean difference",0))
    results_dict["NN Statistical parity difference (Original Test)"].append(metric_test_nn["Average odds difference"])
    results_dict["NN Disparate impact (Original Test)"].append(metric_test_nn["Average odds difference"])
    results_dict["NN Mean difference (Original Test)"].append(metric_test_nn["Average odds difference"])
    results_dict["LR Statistical parity difference (Original Test)"].append(lr_fairness_metrics["test_metrics"]["Statistical parity difference"])
    results_dict["LR Disparate impact (Original Test)"].append(lr_fairness_metrics["test_metrics"]["Disparate impact"])
    results_dict["LR Mean difference (Original Test)"].append(lr_fairness_metrics["test_metrics"]["Mean difference"])
    results_dict["#### Reweighted testing dataset"].append("")
    results_dict["RF Statistical parity difference (Reweighted Test)"].append(test_metrics.get("Statistical parity difference (Reweighted)",0))
    results_dict["RF Disparate impact (Reweighted Test)"].append(test_metrics["Disparate impact (Reweighted)"])
    results_dict["RF Mean difference (Reweighted Test)"].append(test_metrics["Mean difference (Reweighted)"])
    results_dict["NN Statistical parity difference (Reweighted Test)"].append(metric_test_unbiased_nn["Statistical parity difference"])
    results_dict["NN Disparate impact (Reweighted Test)"].append(metric_test_unbiased_nn["Equal opportunity difference"])
    results_dict["NN Mean difference (Reweighted Test)"].append(metric_test_unbiased_nn["Theil index"])

    #Open item-1 After review changes starts
    results_dict["LR Statistical parity difference (Reweighted Test)"].append(lr_fairness_metrics["test_metrics_reweighted"].get("Statistical parity difference", 0))
    results_dict["LR Disparate impact (Reweighted Test)"].append(lr_fairness_metrics["test_metrics_reweighted"].get("Disparate impact", 0))
    results_dict["LR Mean difference (Reweighted Test)"].append(lr_fairness_metrics["test_metrics_reweighted"].get("Mean difference", 0))
    #Open item-1 After review changes ends

    results_dict["#### Repaired (DIR) testing dataset"].append("")
    results_dict["RF Statistical parity difference (Repaired Test)"].append(metric_test_dir["Statistical parity difference"])
    results_dict["RF Disparate impact (Repaired Test)"].append(metric_test_dir["Disparate impact"])
    results_dict["RF Mean difference (Repaired Test)"].append(metric_test_dir["Mean difference"])
    results_dict["NN Statistical parity difference (Repaired Test)"].append(metric_test_unbiased_nn["Statistical parity difference"])
    results_dict["NN Disparate impact (Repaired Test)"].append(metric_test_unbiased_nn["Disparate impact"])
    results_dict["NN Mean difference (Repaired Test)"].append(metric_test_unbiased_nn["Theil index"])
    results_dict["LR Statistical parity difference (Repaired Test)"].append(metric_test_unbiased_lr["Statistical parity difference"])
    results_dict["LR Disparate impact (Repaired Test)"].append(metric_test_unbiased_lr["Disparate impact"])
    results_dict["LR Mean difference (Repaired Test)"].append(metric_test_unbiased_lr["Mean difference"])
    results_dict["The Original Training Accuracy Score (RF)"].append(accuracy_score(y_train_orig, predictions1))
    results_dict["The Original Testing Accuracy Score (RF)"].append(accuracy_score(y_test_orig, predictions2))
    results_dict["The Original Training F1 Score (RF)"].append(f1_score(y_train_orig, predictions1))
    results_dict["The Original Testing F1 Score (RF)"].append(f1_score(y_test_orig, predictions2))
    results_dict["The Original Training Recall Score (RF)"].append(recall_score(y_train_orig, predictions1))
    results_dict["The Original Testing Recall Score (RF)"].append(recall_score(y_test_orig, predictions2))
    results_dict["The Original Training Precision Score (RF)"].append(precision_score(y_train_orig, predictions1))
    results_dict["The Original Testing Precision Score (RF)"].append(precision_score(y_test_orig, predictions2))
    results_dict["The Original Training Accuracy Score (NN)"].append(accuracy_score(nn_train_labels, nn_train_pred))
    results_dict["The Original Testing Accuracy Score (NN)"].append(accuracy_score(nn_test_labels, nn_test_pred))
    results_dict["The Original Training F1 Score (NN)"].append(f1_score(nn_train_labels, nn_train_pred))
    results_dict["The Original Testing F1 Score (NN)"].append(f1_score(nn_test_labels, nn_test_pred))
    results_dict["The Original Training Recall Score (NN)"].append(recall_score(nn_train_labels, nn_train_pred))
    results_dict["The Original Testing Recall Score (NN)"].append(recall_score(nn_test_labels, nn_test_pred))
    results_dict["The Original Training Precision Score (NN)"].append(precision_score(nn_train_labels, nn_train_pred))
    results_dict["The Original Testing Precision Score (NN)"].append(precision_score(nn_test_labels, nn_test_pred))
    results_dict["The Original Training Accuracy Score (LR)"].append(accuracy_score(y_train_orig, pred_train_lr))
    results_dict["The Original Testing Accuracy Score (LR)"].append(accuracy_score(y_test_orig, pred_test_lr))
    results_dict["The Original Training F1 Score (LR)"].append(f1_score(y_train_orig, pred_train_lr))
    results_dict["The Original Testing F1 Score (LR)"].append(f1_score(y_test_orig, pred_test_lr))
    results_dict["The Original Training Recall Score (LR)"].append(recall_score(y_train_orig, pred_train_lr))
    results_dict["The Original Testing Recall Score (LR)"].append(recall_score(y_test_orig, pred_test_lr))
    results_dict["The Original Training Precision Score (LR)"].append(precision_score(y_train_orig, pred_train_lr))
    results_dict["The Original Testing Precision Score (LR)"].append(precision_score(y_test_orig, pred_test_lr))
    results_dict["RF Training Accuracy (Repaired)"].append(repaired_metrics["RF Training Accuracy (Repaired)"])
    results_dict["RF Testing Accuracy (Repaired)"].append(repaired_metrics["RF Testing Accuracy (Repaired)"])
    results_dict["NN Training Accuracy (Repaired)"].append(repaired_metrics["NN Training Accuracy (Repaired)"])
    results_dict["NN Testing Accuracy (Repaired)"].append(repaired_metrics["NN Testing Accuracy (Repaired)"])
    results_dict["LR Training Accuracy (Repaired)"].append(repaired_metrics["LR Training Accuracy (Repaired)"])
    results_dict["LR Testing Accuracy (Repaired)"].append(repaired_metrics["LR Testing Accuracy (Repaired)"])

    #Open item-3 After review changes starts
    results_dict["The Repaired Training F1 Score (RF)"].append(f1_score(y_train_unbiased, predictions1_unbiased))
    results_dict["The Repaired Testing F1 Score (RF)"].append(f1_score(y_test_unbiased, predictions2_unbiased))
    results_dict["The Repaired Training Recall Score (RF)"].append(recall_score(y_train_unbiased, predictions1_unbiased))
    results_dict["The Repaired Testing Recall Score (RF)"].append(recall_score(y_test_unbiased, predictions2_unbiased))
    results_dict["The Repaired Training Precision Score (RF)"].append(precision_score(y_train_unbiased, predictions1_unbiased))
    results_dict["The Repaired Testing Precision Score (RF)"].append(precision_score(y_test_unbiased, predictions2_unbiased))
    results_dict["The Repaired Training F1 Score (NN)"].append(f1_score(nn_train_labels_unbiased, nn_train_pred_unbiased))
    results_dict["The Repaired Testing F1 Score (NN)"].append(f1_score(nn_test_labels_unbiased, nn_test_pred_unbiased))
    results_dict["The Repaired Training Recall Score (NN)"].append(recall_score(nn_train_labels_unbiased, nn_train_pred_unbiased))
    results_dict["The Repaired Testing Recall Score (NN)"].append(recall_score(nn_test_labels_unbiased, nn_test_pred_unbiased))
    results_dict["The Repaired Training Precision Score (NN)"].append(precision_score(nn_train_labels_unbiased, nn_train_pred_unbiased))
    results_dict["The Repaired Testing Precision Score (NN)"].append(precision_score(nn_test_labels_unbiased, nn_test_pred_unbiased))
    results_dict["The Repaired Training F1 Score (LR)"].append(f1_score(y_train_unbiased, pred_train_unbiased_lr))
    results_dict["The Repaired Testing F1 Score (LR)"].append(f1_score(y_test_unbiased, pred_test_unbiased_lr))
    results_dict["The Repaired Training Recall Score (LR)"].append(recall_score(y_train_unbiased, pred_train_unbiased_lr))
    results_dict["The Repaired Testing Recall Score (LR)"].append(recall_score(y_test_unbiased, pred_test_unbiased_lr))
    results_dict["The Repaired Training Precision Score (LR)"].append(precision_score(y_train_unbiased, pred_train_unbiased_lr))
    results_dict["The Repaired Testing Precision Score (LR)"].append(precision_score(y_test_unbiased, pred_test_unbiased_lr))
    #Open item-3 After review changes ends

    results_dict["#### Original Training AIF360 Fairness Metrics:"].append("")
    results_dict["Average odds difference (Original Train) (RF)"].append(metric_train_orig["Average odds difference"])
    results_dict["Equal opportunity difference (Original Train) (RF)"].append(metric_train_orig["Equal opportunity difference"])
    results_dict["Theil index (Original Train) (RF)"].append(metric_train_orig["Theil index"])
    results_dict["Average odds difference (Original Train) (NN)"].append(metric_train_nn["Average odds difference"])
    results_dict["Equal opportunity difference (Original Train) (NN)"].append(metric_train_nn["Equal opportunity difference"])
    results_dict["Theil index (Original Train) (NN)"].append(metric_train_nn["Theil index"])
    results_dict["Average odds difference (Original Train) (LR)"].append(lr_fairness_metrics["train_metrics"]["Average odds difference"])
    results_dict["Equal opportunity difference (Original Train) (LR)"].append(lr_fairness_metrics["train_metrics"]["Equal opportunity difference"])
    results_dict["Theil index (Original Train) (LR)"].append(lr_fairness_metrics["train_metrics"]["Theil index"])
    results_dict["#### Original Testing AIF360 Fairness Metrics:"].append("")
    results_dict["Average odds difference (Original Test) (RF)"].append(metric_test_orig["Average odds difference"])
    results_dict["Equal opportunity difference (Original Test) (RF)"].append(metric_test_orig["Equal opportunity difference"])
    results_dict["Theil index (Original Test) (RF)"].append(metric_test_orig["Theil index"])
    results_dict["Average odds difference (Original Test) (NN)"].append(metric_test_nn["Average odds difference"])
    results_dict["Equal opportunity difference (Original Test) (NN)"].append(metric_test_nn["Equal opportunity difference"])
    results_dict["Theil index (Original Test) (NN)"].append(metric_test_nn["Theil index"])
    results_dict["Average odds difference (Original Test) (LR)"].append(lr_fairness_metrics["test_metrics"]["Average odds difference"])
    results_dict["Equal opportunity difference (Original Test) (LR)"].append(lr_fairness_metrics["test_metrics"]["Equal opportunity difference"])
    results_dict["Theil index (Original Test) (LR)"].append(lr_fairness_metrics["test_metrics"]["Theil index"])

    #Open item-2 After review changes starts
    results_dict["#### Repaired Training AIF360 Fairness Metrics:"].append("")
    results_dict["Average odds difference (Repaired Train) (RF)"].append(metric_train_dir["Average odds difference"])
    results_dict["Equal opportunity difference (Repaired Train) (RF)"].append(metric_train_dir["Equal opportunity difference"])
    results_dict["Theil index (Repaired Train) (RF)"].append(metric_train_dir["Theil index"])
    results_dict["Average odds difference (Repaired Train) (NN)"].append(metric_train_unbiased_nn["Average odds difference"])
    results_dict["Equal opportunity difference (Repaired Train) (NN)"].append(metric_train_unbiased_nn["Equal opportunity difference"])
    results_dict["Theil index (Repaired Train) (NN)"].append(metric_train_unbiased_nn["Theil index"])
    results_dict["Average odds difference (Repaired Train) (LR)"].append(metric_train_unbiased_lr["Average odds difference"])
    results_dict["Equal opportunity difference (Repaired Train) (LR)"].append(metric_train_unbiased_lr["Equal opportunity difference"])
    results_dict["Theil index (Repaired Train) (LR)"].append(metric_train_unbiased_lr["Theil index"])
    results_dict["#### Repaired Testing AIF360 Fairness Metrics:"].append("")
    results_dict["Average odds difference (Repaired Test) (RF)"].append(metric_test_dir["Average odds difference"])
    results_dict["Equal opportunity difference (Repaired Test) (RF)"].append(metric_test_dir["Equal opportunity difference"])
    results_dict["Theil index (Repaired Test) (RF)"].append(metric_test_dir["Theil index"])
    results_dict["Average odds difference (Repaired Test) (NN)"].append(metric_test_unbiased_nn["Average odds difference"])
    results_dict["Equal opportunity difference (Repaired Test) (NN)"].append(metric_test_unbiased_nn["Equal opportunity difference"])
    results_dict["Theil index (Repaired Test) (NN)"].append(metric_test_unbiased_nn["Theil index"])
    results_dict["Average odds difference (Repaired Test) (LR)"].append(metric_test_unbiased_lr["Average odds difference"])
    results_dict["Equal opportunity difference (Repaired Test) (LR)"].append(metric_test_unbiased_lr["Equal opportunity difference"])
    results_dict["Theil index (Repaired Test) (LR)"].append(metric_test_unbiased_lr["Theil index"])
    #Open item-2 After review changes ends

    results_dict["The Unbiased Training Accuracy Score (RF)"].append(accuracy_score(y_train_unbiased, predictions1_unbiased))
    results_dict["The Unbiased Testing Accuracy Score (RF)"].append(accuracy_score(y_test_unbiased, predictions2_unbiased))
    results_dict["The Unbiased Training F1 Score (RF)"].append(f1_score(y_train_unbiased, predictions1_unbiased))
    results_dict["The Unbiased Testing F1 Score (RF)"].append(f1_score(y_test_unbiased, predictions2_unbiased))
    results_dict["The Unbiased Training Recall Score (RF)"].append(recall_score(y_train_unbiased, predictions1_unbiased))
    results_dict["The Unbiased Testing Recall Score (RF)"].append(recall_score(y_test_unbiased, predictions2_unbiased))
    results_dict["The Unbiased Training Precision Score (RF)"].append(precision_score(y_train_unbiased, predictions1_unbiased))
    results_dict["The Unbiased Testing Precision Score (RF)"].append(precision_score(y_test_unbiased, predictions2_unbiased))
    results_dict["The Unbiased Training Accuracy Score (NN)"].append(accuracy_score(nn_train_labels_unbiased, nn_train_pred_unbiased))
    results_dict["The Unbiased Testing Accuracy Score (NN)"].append(accuracy_score(nn_test_labels_unbiased, nn_test_pred_unbiased))
    results_dict["The Unbiased Training F1 Score (NN)"].append(f1_score(nn_train_labels_unbiased, nn_train_pred_unbiased))
    results_dict["The Unbiased Testing F1 Score (NN)"].append(f1_score(nn_test_labels_unbiased, nn_test_pred_unbiased))
    results_dict["The Unbiased Training Recall Score (NN)"].append(recall_score(nn_train_labels_unbiased, nn_train_pred_unbiased))
    results_dict["The Unbiased Testing Recall Score (NN)"].append(recall_score(nn_test_labels_unbiased, nn_test_pred_unbiased))
    results_dict["The Unbiased Training Precision Score (NN)"].append(precision_score(nn_train_labels_unbiased, nn_train_pred_unbiased))
    results_dict["The Unbiased Testing Precision Score (NN)"].append(precision_score(nn_test_labels_unbiased, nn_test_pred_unbiased))
    results_dict["The Unbiased Training Accuracy Score (LR)"].append(accuracy_score(y_train_unbiased, pred_train_unbiased_lr))
    results_dict["The Unbiased Testing Accuracy Score (LR)"].append(accuracy_score(y_test_unbiased, pred_test_unbiased_lr))
    results_dict["The Unbiased Training F1 Score (LR)"].append(f1_score(y_train_unbiased, pred_train_unbiased_lr))
    results_dict["The Unbiased Testing F1 Score (LR)"].append(f1_score(y_test_unbiased, pred_test_unbiased_lr))
    results_dict["The Unbiased Training Recall Score (LR)"].append(recall_score(y_train_unbiased, pred_train_unbiased_lr))
    results_dict["The Unbiased Testing Recall Score (LR)"].append(recall_score(y_test_unbiased, pred_test_unbiased_lr))
    results_dict["The Unbiased Training Precision Score (LR)"].append(precision_score(y_train_unbiased, pred_train_unbiased_lr))
    results_dict["The Unbiased Testing Precision Score (LR)"].append(precision_score(y_test_unbiased, pred_test_unbiased_lr))
    results_dict["#### Unbiased Training AIF360 Fairness Metrics:"].append("")
    results_dict["Average odds difference (Unbiased Train) (RF)"].append(metric_train_unbiased["Average odds difference"])
    results_dict["Equal opportunity difference (Unbiased Train) (RF)"].append(metric_train_unbiased["Equal opportunity difference"])
    results_dict["Theil index (Unbiased Train) (RF)"].append(metric_train_unbiased["Theil index"])
    results_dict["Average odds difference (Unbiased Train) (NN)"].append(metric_train_unbiased_nn["Average odds difference"])
    results_dict["Equal opportunity difference (Unbiased Train) (NN)"].append(metric_train_unbiased_nn["Equal opportunity difference"])
    results_dict["Theil index (Unbiased Train) (NN)"].append(metric_train_unbiased_nn["Theil index"])
    results_dict["Average odds difference (Unbiased Train) (LR)"].append(metric_train_unbiased_lr["Average odds difference"])
    results_dict["Equal opportunity difference (Unbiased Train) (LR)"].append(metric_train_unbiased_lr["Equal opportunity difference"])
    results_dict["Theil index (Unbiased Train) (LR)"].append(metric_train_unbiased_lr["Theil index"])
    results_dict["#### Unbiased Testing AIF360 Fairness Metrics:"].append("")
    results_dict["Average odds difference (Unbiased Test) (RF)"].append(metric_test_unbiased["Average odds difference"])
    results_dict["Equal opportunity difference (Unbiased Test) (RF)"].append(metric_test_unbiased["Equal opportunity difference"])
    results_dict["Theil index (Unbiased Test) (RF)"].append(metric_test_unbiased["Theil index"])
    results_dict["Average odds difference (Unbiased Test) (NN)"].append(metric_test_unbiased_nn["Average odds difference"])
    results_dict["Equal opportunity difference (Unbiased Test) (NN)"].append(metric_test_unbiased_nn["Equal opportunity difference"])
    results_dict["Theil index (Unbiased Test) (NN)"].append(metric_test_unbiased_nn["Theil index"])
    results_dict["Average odds difference (Unbiased Test) (LR)"].append(metric_test_unbiased_lr["Average odds difference"])
    results_dict["Equal opportunity difference (Unbiased Test) (LR)"].append(metric_test_unbiased_lr["Equal opportunity difference"])
    results_dict["Theil index (Unbiased Test) (LR)"].append(metric_test_unbiased_lr["Theil index"])
    results_dict["Denial Reason Training Accuracy Score (RF)"] = rf_denial_train_accuracy
    results_dict["Denial Reason Testing Accuracy Score (RF)"] = rf_denial_test_accuracy
    results_dict["Denial Reason Training Accuracy Score (NN)"] = nn_denial_train_accuracy
    results_dict["Denial Reason Testing Accuracy Score (NN)"] = nn_denial_test_accuracy
    results_dict["Denial Reason Training Accuracy Score (LR)"] = lr_denial_train_accuracy
    results_dict["Denial Reason Testing Accuracy Score (LR)"] = lr_denial_test_accuracy
    results_dict["Execution_Time_unbiased (Hours)"] = total_execution_time_unbiased
    results_dict["Execution_Time_orig (Hours)"] = total_execution_time_orig
    results_dict["Energy_Unbiased (kWh)"] = total_energy_unbiased
    results_dict["Energy_Orig (kWh)"] = total_energy_orig
    results_dict["Execution_Time_unbiased_nn (Hours)"].append(execution_time_unbiased_nn)
    results_dict["Execution_Time_orig_nn (Hours)"].append(execution_time_orig_nn)
    results_dict["Energy_Unbiased_nn (kWh)"].append(energy_unbiased_nn)
    results_dict["Energy_Orig_nn (kWh)"].append(energy_orig_nn)

    #Open item-5 After review changes starts
    results_dict["Execution_Time_Orig_lr (Hours)"].append(execution_time_orig_lr)
    results_dict["Energy_Orig_lr (kWh)"].append(energy_orig_lr)
    results_dict["Execution_Time_Reweighted_lr (Hours)"].append(execution_time_reweighted_lr)
    results_dict["Energy_Reweighted_lr (kWh)"].append(energy_reweighted_lr)
    results_dict["Execution_Time_Orig_dir (Hours)"].append(execution_time_dir_orig)
    results_dict["Energy_Orig_dir (kWh)"].append(energy_dir_orig)
    results_dict["Execution_Time_Model4_9 (Hours)"].append(execution_time_model4_9)
    results_dict["Energy_Model4_9 (kWh)"].append(energy_model4_9)
    results_dict["Execution_Time_Model10_15 (Hours)"].append(execution_time_model10_15)
    results_dict["Energy_Model10_15 (kWh)"].append(energy_model10_15)
    results_dict["Execution_Time_Model16_21 (Hours)"].append(execution_time_model16_21)
    results_dict["Energy_Model16_21 (kWh)"].append(energy_model16_21)
    results_dict["Execution_Time_Model22_27 (Hours)"].append(execution_time_model22_27)
    results_dict["Energy_Model22_27 (kWh)"].append(energy_model22_27)
    results_dict["Execution_Time_Model28_33 (Hours)"].append(execution_time_model28_33)
    results_dict["Energy_Model28_33 (kWh)"].append(energy_model28_33)
    results_dict["Execution_Time_Model34_39 (Hours)"].append(execution_time_model34_39)
    results_dict["Energy_Model34_39 (kWh)"].append(energy_model34_39)
    #Open item-5 After review changes ends

def add_model3_columns(df, pred_test_lr):
    """
    Add columns for Model 3 (Logistic Regression) predictions.
    Ensures prediction length matches DataFrame length.
    """
    if isinstance(pred_test_lr, list):
        pred_test_lr = np.array(pred_test_lr)

    min_length = min(len(df), len(pred_test_lr))
    df = df.iloc[:min_length].copy()
    pred_test_lr = pred_test_lr[:min_length]

    df['Original_Approved_lr'] = pred_test_lr

    for attr in [
        'tract_minority_percentage40_0', 'tract_minority_percentage50_0', 'tract_minority_percentage60_0',
        'tract_minority_percentage70_0', 'tract_minority_percentage80_0', 'tract_minority_percentage90_0'
    ]:
        df[f'{attr}_approved_lr'] = pred_test_lr

    return df

def add_model4_9_columns(df, pred_test_nn_reweighing):
    """
    Add columns for Model 4-9 (NN new & Reweighing) predictions.
    Handles both 1D and 2D prediction arrays.
    """
    if isinstance(pred_test_nn_reweighing, list):
        pred_test_nn_reweighing = np.array(pred_test_nn_reweighing)

    # Align predictions with the DataFrame length
    pred_test_nn_reweighing = align_predictions(pred_test_nn_reweighing, len(df))

    df['Original_Approved_nn_reweighing'] = pred_test_nn_reweighing

    for i, attr in enumerate([
        'tract_minority_percentage40_0', 'tract_minority_percentage50_0', 'tract_minority_percentage60_0',
        'tract_minority_percentage70_0', 'tract_minority_percentage80_0', 'tract_minority_percentage90_0'
    ]):
        df[f'{attr}_approved_nn_reweighing'] = (
            pred_test_nn_reweighing if pred_test_nn_reweighing.ndim == 1 else pred_test_nn_reweighing[:, i]
        )

    return df

def add_model10_15_columns(df, pred_test_rf_reweighing):
    """
    Add columns for Model 10-15 (RF grid search & Reweighing) predictions.
    Handles both 1D and 2D prediction arrays.
    """
    if isinstance(pred_test_rf_reweighing, list):
        pred_test_rf_reweighing = np.array(pred_test_rf_reweighing)

    # Align predictions with the DataFrame length
    pred_test_rf_reweighing = align_predictions(pred_test_rf_reweighing, len(df))

    df['Original_Approved_rf_reweighing'] = pred_test_rf_reweighing

    for i, attr in enumerate([
        'tract_minority_percentage40_0', 'tract_minority_percentage50_0', 'tract_minority_percentage60_0',
        'tract_minority_percentage70_0', 'tract_minority_percentage80_0', 'tract_minority_percentage90_0'
    ]):
        df[f'{attr}_approved_rf_reweighing'] = (
            pred_test_rf_reweighing if pred_test_rf_reweighing.ndim == 1 else pred_test_rf_reweighing[:, i]
        )

    return df

def add_model16_21_columns(df, pred_test_lr_reweighing):
    """
    Add columns for Model 16-21 (LR & Reweighing) predictions.
    """
    if isinstance(pred_test_lr_reweighing, list):
        pred_test_lr_reweighing = np.array(pred_test_lr_reweighing)

    # Align predictions with the DataFrame length
    pred_test_lr_reweighing = align_predictions(pred_test_lr_reweighing, len(df))

    df['Original_Approved_lr_reweighing'] = pred_test_lr_reweighing

    for i, attr in enumerate([
        'tract_minority_percentage40_0', 'tract_minority_percentage50_0', 'tract_minority_percentage60_0',
        'tract_minority_percentage70_0', 'tract_minority_percentage80_0', 'tract_minority_percentage90_0'
    ]):
        df[f'{attr}_approved_lr_reweighing'] = (
            pred_test_lr_reweighing if pred_test_lr_reweighing.ndim == 1 else pred_test_lr_reweighing[:, i]
        )

    return df

def add_model22_27_columns(df, pred_test_nn_disparate_impact):
    """
    Add columns for Model 22-27 (NN new & Disparate impact) predictions.
    """
    if isinstance(pred_test_nn_disparate_impact, list):
        pred_test_nn_disparate_impact = np.array(pred_test_nn_disparate_impact)

    # Align predictions with the DataFrame length
    pred_test_nn_disparate_impact = align_predictions(pred_test_nn_disparate_impact, len(df))

    df['Original_Approved_nn_disparate_impact'] = pred_test_nn_disparate_impact

    for i, attr in enumerate([
        'tract_minority_percentage40_0', 'tract_minority_percentage50_0', 'tract_minority_percentage60_0',
        'tract_minority_percentage70_0', 'tract_minority_percentage80_0', 'tract_minority_percentage90_0'
    ]):
        df[f'{attr}_approved_nn_disparate_impact'] = (
            pred_test_nn_disparate_impact if pred_test_nn_disparate_impact.ndim == 1 else pred_test_nn_disparate_impact[:, i]
        )

    return df

def add_model28_33_columns(df, pred_test_rf_disparate_impact):
    """
    Add columns for Model 28-33 (RF grid search & Disparate impact) predictions.
    """
    if isinstance(pred_test_rf_disparate_impact, list):
        pred_test_rf_disparate_impact = np.array(pred_test_rf_disparate_impact)

    # Align predictions with the DataFrame length
    pred_test_rf_disparate_impact = align_predictions(pred_test_rf_disparate_impact, len(df))

    df['Original_Approved_rf_disparate_impact'] = pred_test_rf_disparate_impact

    for i, attr in enumerate([
        'tract_minority_percentage40_0', 'tract_minority_percentage50_0', 'tract_minority_percentage60_0',
        'tract_minority_percentage70_0', 'tract_minority_percentage80_0', 'tract_minority_percentage90_0'
    ]):
        df[f'{attr}_approved_rf_disparate_impact'] = (
            pred_test_rf_disparate_impact if pred_test_rf_disparate_impact.ndim == 1 else pred_test_rf_disparate_impact[:, i]
        )

    return df

def add_model34_39_columns(df, pred_test_lr_disparate_impact):
    """
    Add columns for Model 34-39 (LR & Disparate impact) predictions.
    """
    if isinstance(pred_test_lr_disparate_impact, list):
        pred_test_lr_disparate_impact = np.array(pred_test_lr_disparate_impact)

    # Align predictions with the DataFrame length
    pred_test_lr_disparate_impact = align_predictions(pred_test_lr_disparate_impact, len(df))

    df['Original_Approved_lr_disparate_impact'] = pred_test_lr_disparate_impact

    for i, attr in enumerate([
        'tract_minority_percentage40_0', 'tract_minority_percentage50_0', 'tract_minority_percentage60_0',
        'tract_minority_percentage70_0', 'tract_minority_percentage80_0', 'tract_minority_percentage90_0'
    ]):
        df[f'{attr}_approved_lr_disparate_impact'] = (
            pred_test_lr_disparate_impact if pred_test_lr_disparate_impact.ndim == 1 else pred_test_lr_disparate_impact[:, i]
        )

    return df

# Function F
def generate_fairness_plots(csv_file):
    # Get the directory where FairPlay file is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a directory for saving the plots within the same directory as the script
    output_dir = os.path.join(script_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # new changes for visulization starts

    # Load the 	unbiased_dataset_updated.csv file
    unbiased_df = pd.read_csv('	unbiased_dataset_updated.csv')

    # Extract the columns for visualization
    model_columns = [
        'tract_minority_percentage40_0_approved_nn_reweighing', 'tract_minority_percentage50_0_approved_nn_reweighing',
        'tract_minority_percentage60_0_approved_nn_reweighing', 'tract_minority_percentage70_0_approved_nn_reweighing',
        'tract_minority_percentage80_0_approved_nn_reweighing', 'tract_minority_percentage90_0_approved_nn_reweighing',
        'tract_minority_percentage40_0_approved_rf_reweighing', 'tract_minority_percentage50_0_approved_rf_reweighing',
        'tract_minority_percentage60_0_approved_rf_reweighing', 'tract_minority_percentage70_0_approved_rf_reweighing',
        'tract_minority_percentage80_0_approved_rf_reweighing', 'tract_minority_percentage90_0_approved_rf_reweighing',
        'tract_minority_percentage40_0_approved_lr_reweighing', 'tract_minority_percentage50_0_approved_lr_reweighing',
        'tract_minority_percentage60_0_approved_lr_reweighing', 'tract_minority_percentage70_0_approved_lr_reweighing',
        'tract_minority_percentage80_0_approved_lr_reweighing', 'tract_minority_percentage90_0_approved_lr_reweighing',
        'tract_minority_percentage40_0_approved_nn_disparate_impact', 'tract_minority_percentage50_0_approved_nn_disparate_impact',
        'tract_minority_percentage60_0_approved_nn_disparate_impact', 'tract_minority_percentage70_0_approved_nn_disparate_impact',
        'tract_minority_percentage80_0_approved_nn_disparate_impact', 'tract_minority_percentage90_0_approved_nn_disparate_impact',
        'tract_minority_percentage40_0_approved_rf_disparate_impact', 'tract_minority_percentage50_0_approved_rf_disparate_impact',
        'tract_minority_percentage60_0_approved_rf_disparate_impact', 'tract_minority_percentage70_0_approved_rf_disparate_impact',
        'tract_minority_percentage80_0_approved_rf_disparate_impact', 'tract_minority_percentage90_0_approved_rf_disparate_impact',
        'tract_minority_percentage40_0_approved_lr_disparate_impact', 'tract_minority_percentage50_0_approved_lr_disparate_impact',
        'tract_minority_percentage60_0_approved_lr_disparate_impact', 'tract_minority_percentage70_0_approved_lr_disparate_impact',
        'tract_minority_percentage80_0_approved_lr_disparate_impact', 'tract_minority_percentage90_0_approved_lr_disparate_impact'
    ]

    # Extract the relevant columns from the DataFrame
    model_data = unbiased_df[model_columns]

    # Create a figure for the visualization
    fig, ax = plt.subplots(figsize=(20, 10))

    # Define colors and line styles for different models
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    line_styles = ['-', '--', '-.', ':']

    # Plot each model's data
    for i, column in enumerate(model_columns):
        model_type = column.split('_')[-1]  # Extract model type (reweighing, disparate_impact)
        protected_attribute = column.split('_')[2]  # Extract protected attribute (40, 50, 60, etc.)
        model = column.split('_')[-2]  # Extract model (nn, rf, lr)

        # Determine color and line style based on model type and protected attribute
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]

        # Plot the data
        ax.plot(model_data.index, model_data[column], label=f'{model} {model_type} {protected_attribute}',
                color=color, linestyle=line_style, linewidth=2)

    # Customize the plot
    ax.set_xlabel('Data Points', fontsize=14)
    ax.set_ylabel('Approval Probability', fontsize=14)
    ax.set_title('Model 3-39: Approval Probability Across Protected Attributes', fontsize=16)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Save the plot
    output_file = os.path.join(output_dir, 'model_4_39_approval_probability.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    # Existing visualization code for fairness metrics, performance metrics, denial reason metrics, and energy/execution time comparison
    # (Keep the existing code as is, and add the new visualization code above)

    # new changes for visualization ends

    # Energy and Execution Time Visualization
    # Attributes to plot
    attributes = [
        'tract_minority_percentage40_0',
        'tract_minority_percentage50_0',
        'tract_minority_percentage60_0',
        'tract_minority_percentage70_0',
        'tract_minority_percentage80_0',
        'tract_minority_percentage90_0'
    ]

    # Original vs debiased visualization
    # Load the results file
    results_df = pd.read_csv(csv_file, index_col=0)

    # Fairness metrics to plot
    fairness_metrics_types = [
        "Statistical parity difference",
        "Disparate impact"
    ]

    # Sections for Original and Reweighted
    fairness_sections = {
        "Training": {
            "Original": [
                "RF Statistical parity difference (Original Train)",
                "RF Disparate impact (Original Train)",
                "NN Statistical parity difference (Original Train)",
                "NN Disparate impact (Original Train)",
                #Open item-6 After review changes starts
                "LR Statistical parity difference (Original Train)",
                "LR Disparate impact (Original Train)"
                #Open item-6 After review changes ends
            ],
            "Reweighted": [
                "RF Statistical parity difference (Reweighted Train)",
                "RF Disparate impact (Reweighted Train)",
                "NN Statistical parity difference (Reweighted Train)",
                "NN Disparate impact (Reweighted Train)",
                #Open item-6 After review changes starts
                "LR Statistical parity difference (Reweighted Train)",
                "LR Disparate impact (Reweighted Train)"
                #Open item-6 After review changes ends
            ],
            #Open item-6 After review changes starts
            "Repaired": [
                "RF Statistical parity difference (Repaired Train)",
                "RF Disparate impact (Repaired Train)",
                "NN Statistical parity difference (Repaired Train)",
                "NN Disparate impact (Repaired Train)",
                "LR Statistical parity difference (Repaired Train)",
                "LR Disparate impact (Repaired Train)"
            ],
            #Open item-6 After review changes ends
        },
        "Testing": {
            "Original": [
                "RF Statistical parity difference (Original Test)",
                "RF Disparate impact (Original Test)",
                "NN Statistical parity difference (Original Test)",
                "NN Disparate impact (Original Test)",
                #Open item-6 After review changes starts
                "LR Statistical parity difference (Original Test)",
                "LR Disparate impact (Original Test)"
                #Open item-6 After review changes ends
            ],
            "Reweighted": [
                "RF Statistical parity difference (Reweighted Test)",
                "RF Disparate impact (Reweighted Test)",
                "NN Statistical parity difference (Reweighted Test)",
                "NN Disparate impact (Reweighted Test)",
                #Open item-6 After review changes starts
                "LR Statistical parity difference (Reweighted Test)",
                "LR Disparate impact (Reweighted Test)"
                #Open item-6 After review changes ends
            ],
            #Open item-6 After review changes starts
            "Repaired": [
                "RF Statistical parity difference (Repaired Test)",
                "RF Disparate impact (Repaired Test)",
                "NN Statistical parity difference (Repaired Test)",
                "NN Disparate impact (Repaired Test)",
                "LR Statistical parity difference (Repaired Test)",
                "LR Disparate impact (Repaired Test)"
            ]
            #Open item-6 After review changes ends
        }
    }

    for section, metrics in fairness_sections.items():
        # Create a figure for each section
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax2 = ax1.twinx()  # Second y-axis for Disparate Impact

        # Colors for metrics
        colors = {
            "Statistical parity difference": "tab:red",
            "Disparate impact": "tab:blue"
        }

        # Loop through Original and Reweighted and Repaired metrics
        for category, category_metrics in metrics.items():
            #Open item-6 After review changes starts
            linestyle = "-" if category == "Original" else "--" if category == "Reweighted" else "-."
            #Open item-6 After review changes ends

            # Plot each metric
            for metric_name in category_metrics:
                # Determine metric type and model
                metric_type = "Statistical parity difference" if "Statistical parity difference" in metric_name else "Disparate impact"
                model = "RF" if "RF" in metric_name else ("NN" if "NN" in metric_name else "LR")

                # Extract values from results_df (or use placeholders for this demonstration)
                values = results_df.loc[metric_name, attributes].tolist() if metric_name in results_df.index else [0] * len(attributes)

                # Plot on the appropriate Y-axis
                if metric_type == "Statistical parity difference":
                    ax1.plot(
                        [40, 50, 60, 70, 80, 90], values,
                        marker='o', linestyle=linestyle,
                        label=f"{model} {category} {metric_type}",
                        color=colors[metric_type]
                    )
                elif metric_type == "Disparate impact":
                    ax2.plot(
                        [40, 50, 60, 70, 80, 90], values,
                        marker='x', linestyle=linestyle,
                        label=f"{model} {category} {metric_type}",
                        color=colors[metric_type]
                    )

        # Customize axes
        ax1.set_xlabel("Protected Attributes", fontsize=14)
        ax1.set_ylabel("Statistical Parity Difference", fontsize=14, color=colors["Statistical parity difference"])
        ax1.tick_params(axis="y", labelcolor=colors["Statistical parity difference"])
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2.set_ylabel("Disparate Impact", fontsize=14, color=colors["Disparate impact"])
        ax2.tick_params(axis="y", labelcolor=colors["Disparate impact"])

        # Add x-axis ticks
        ax1.set_xticks([40, 50, 60, 70, 80, 90])
        ax1.set_xticklabels([40, 50, 60, 70, 80, 90], fontsize=12)

        # Adjust legend position
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.15))

        # Set title and save the plot
        plt.title(f"Fairness Metrics ({section}): Statistical Parity Difference and Disparate Impact", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the legend better

        # Save the figure
        output_file = os.path.join(output_dir, f"{section}_Fairness_Metrics_Line_Graph.png")
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
        # print(f"Saved {section} fairness metrics plot to {output_file}")

    # Performance metrics to plot
    performance_metrics_types = ["Accuracy Score", "F1 Score", "Recall Score", "Precision Score"]
    sections = ["Training", "Testing"]  # Separate Training and Testing
    categories = ["Original", "Unbiased", "Repaired"]  # Original vs Unbiased vs Repaired
    models = ["RF", "NN", "LR"]  # Models: RF and NN and LR

    # Plot Performance Metrics for both RF and NN
    fig, ax = plt.subplots(figsize=(18, 12))

    # Colors and markers for distinction
    colors = [
        "tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple", 
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
        #Open item-6 After review changes starts
        "navy", "maroon", "darkgreen", "darkorange", "darkviolet"
        #Open item-6 After review changes ends
    ]
    #Open item-6 After review changes starts
    markers = ["o", "x", "s", "^", "D", "*", "P", "h", "v", "<", ">", "1", "2", "3", "4"]
    line_styles = {"Original": "-", "Unbiased": "--", "Repaired": "-."}  # Line styles for categories
    #Open item-6 After review changes ends

    # Plot metrics for each model (RF and NN) and Original/Unbiased
    for i, metric in enumerate(performance_metrics_types):
        for model in models:
            for category in categories:
                for section in sections:
                    # Construct metric name
                    #Open item-6 After review changes starts
                    if category == "Repaired":
                        metric_name = f"The {category} {section} {metric} ({model})"
                    else:
                        metric_name = f"The {category} {section} {metric} ({model})"
                    #Open item-6 After review changes ends

                    # Extract values from the dataframe (or use placeholders if not present)
                    values = results_df.loc[metric_name, attributes].tolist() if metric_name in results_df.index else [0] * len(attributes)

                    # Determine label for legend
                    label = f"{model} {category} {section} {metric}"

                    # Plot the line
                    ax.plot(
                        [40, 50, 60, 70, 80, 90],
                        values,
                        marker=markers[i % len(markers)],
                        linestyle=line_styles[category],
                        color=colors[(models.index(model) + len(models)*i) % len(colors)],
                        linewidth=1.5,
                        label=label
                    )

    # Customize the axes
    ax.set_xlabel("Protected Attributes", fontsize=14)
    ax.set_ylabel("Performance Metrics(Accuracy,F1,Recall,Precision)", fontsize=14)
    ax.set_xticks([40, 50, 60, 70, 80, 90])
    ax.set_xticklabels([40, 50, 60, 70, 80, 90], fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Adjust legend position
    ax.legend(
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.15),  # Move legend below the graph
        ncol=4,  # Arrange in 4 columns
        fontsize=8,
        frameon=False
    )

    # Set title and layout
    plt.title("Performance Metrics Comparison: RF, NN and LR Models (Original, Unbiased and Repaired)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to leave space for legend

    # Save the figure
    output_file = os.path.join(output_dir, "Performance_Metrics_Line_Graph.png")
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    # Denial Reason Metrics to plot
    denial_metrics_types = [
        "Denial Reason Training Accuracy Score (RF)",
        "Denial Reason Testing Accuracy Score (RF)",
        "Denial Reason Training Accuracy Score (NN)",
        "Denial Reason Testing Accuracy Score (NN)",
        #Open item-6 After review changes starts
        "Denial Reason Training Accuracy Score (LR)",
        "Denial Reason Testing Accuracy Score (LR)"
        #Open item-6 After review changes ends
    ]

    # Plot Denial Reason Performance Metrics
    fig, ax = plt.subplots(figsize=(16, 10))

    # Colors and markers for distinction
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:cyan"]
    markers = ["o", "x", "s", "^", "D", "*"]
    line_styles = {"Training": "-", "Testing": "--"}  # Line styles for Training and Testing

    # Plot metrics for RF and NN
    for i, metric_name in enumerate(denial_metrics_types):
        # Determine whether it's Training or Testing
        section = "Training" if "Training" in metric_name else "Testing"
        
        # Extract model type (RF or NN or LR)
        #Open item-6 After review changes starts
        model = "RF" if "(RF)" in metric_name else ("NN" if "(NN)" in metric_name else "LR")
        #Open item-6 After review changes ends

        # Extract values from the dataframe (or use placeholders if not present)
        values = results_df.loc[metric_name, attributes].tolist() if metric_name in results_df.index else [0] * len(attributes)

        # Determine label for legend
        label = f"{model} {section} Denial Reason Accuracy"

        # Plot the line
        ax.plot(
            [40, 50, 60, 70, 80, 90],
            values,
            marker=markers[i % len(markers)],
            linestyle=line_styles[section],
            color=colors[i % len(colors)],
            linewidth=2,
            label=label
        )

    # Customize the axes
    ax.set_xlabel("Protected Attributes", fontsize=14)
    ax.set_ylabel("Accuracy Score", fontsize=14)
    ax.set_xticks([40, 50, 60, 70, 80, 90])
    ax.set_xticklabels([40, 50, 60, 70, 80, 90], fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Adjust legend position
    ax.legend(
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.15),  # Move legend below the graph
        ncol=2,  # Arrange in 2 columns
        fontsize=10,
        frameon=False
    )

    # Set title and layout
    plt.title("Performance Denial Metrics: RF and NN and LR Models (Training and Testing)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to leave space for legend
    # Save the figure
    output_file = os.path.join(output_dir, "Performance_Denial_Metrics_Line_Graph.png")
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    # Execution Time and Energy Metrics
    #Open item-6 After review changes starts
    execution_metrics = [
        "Execution_Time_orig (Hours)",
        "Execution_Time_unbiased (Hours)",
        "Execution_Time_orig_nn (Hours)",
        "Execution_Time_unbiased_nn (Hours)",
        "Execution_Time_Orig_lr (Hours)",
        "Execution_Time_Reweighted_lr (Hours)",
        "Execution_Time_Orig_dir (Hours)",
        "Execution_Time_Model4_9 (Hours)",
        "Execution_Time_Model10_15 (Hours)",
        "Execution_Time_Model16_21 (Hours)",
        "Execution_Time_Model22_27 (Hours)",
        "Execution_Time_Model28_33 (Hours)",
        "Execution_Time_Model34_39 (Hours)"
    ]
    
    energy_metrics = [
        "Energy_Orig (kWh)",
        "Energy_Unbiased (kWh)",
        "Energy_Orig_nn (kWh)",
        "Energy_Unbiased_nn (kWh)",
        "Energy_Orig_lr (kWh)",
        "Energy_Reweighted_lr (kWh)",
        "Energy_Orig_dir (kWh)",
        "Energy_Model4_9 (kWh)",
        "Energy_Model10_15 (kWh)",
        "Energy_Model16_21 (kWh)",
        "Energy_Model22_27 (kWh)",
        "Energy_Model28_33 (kWh)",
        "Energy_Model34_39 (kWh)"
    ]
    #Open item-6 After review changes ends

    # Create execution time and energy comparison plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14))
    
    # Plot execution times
    for i, metric in enumerate(execution_metrics):
        values = results_df.loc[metric, attributes].tolist() if metric in results_df.index else [0] * len(attributes)
        ax1.plot([40, 50, 60, 70, 80, 90], values, 
                marker=markers[i % len(markers)],
                linestyle=line_styles[i % len(line_styles)],
                label=metric.replace("Execution_Time_", "").replace(" (Hours)", ""),
                linewidth=2)
    
    ax1.set_title("Execution Time Comparison Across All Models and Methods", fontsize=16)
    ax1.set_ylabel("Hours", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)
    
    # Plot energy consumption
    for i, metric in enumerate(energy_metrics):
        values = results_df.loc[metric, attributes].tolist() if metric in results_df.index else [0] * len(attributes)
        ax2.plot([40, 50, 60, 70, 80, 90], values,
                marker=markers[i % len(markers)],
                linestyle=line_styles[i % len(line_styles)],
                label=metric.replace("Energy_", "").replace(" (kWh)", ""),
                linewidth=2)
    
    ax2.set_title("Energy Consumption Comparison Across All Models and Methods", fontsize=16)
    ax2.set_xlabel("Protected Attributes", fontsize=14)
    ax2.set_ylabel("kWh", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "execution_energy_comparison_all_models.png"))
    plt.close()

    # AIF360 Fairness Metrics for Repaired data
    #Open item-6 After review changes starts
    aif360_metrics = {
        "Training": {
            "Average odds difference": [
                "Average odds difference (Repaired Train) (RF)",
                "Average odds difference (Repaired Train) (NN)",
                "Average odds difference (Repaired Train) (LR)"
            ],
            "Equal opportunity difference": [
                "Equal opportunity difference (Repaired Train) (RF)",
                "Equal opportunity difference (Repaired Train) (NN)",
                "Equal opportunity difference (Repaired Train) (LR)"
            ],
            "Theil index": [
                "Theil index (Repaired Train) (RF)",
                "Theil index (Repaired Train) (NN)",
                "Theil index (Repaired Train) (LR)"
            ]
        },
        "Testing": {
            "Average odds difference": [
                "Average odds difference (Repaired Test) (RF)",
                "Average odds difference (Repaired Test) (NN)",
                "Average odds difference (Repaired Test) (LR)"
            ],
            "Equal opportunity difference": [
                "Equal opportunity difference (Repaired Test) (RF)",
                "Equal opportunity difference (Repaired Test) (NN)",
                "Equal opportunity difference (Repaired Test) (LR)"
            ],
            "Theil index": [
                "Theil index (Repaired Test) (RF)",
                "Theil index (Repaired Test) (NN)",
                "Theil index (Repaired Test) (LR)"
            ]
        }
    }
    #Open item-6 After review changes ends

    for section, metrics in aif360_metrics.items():
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ["tab:blue", "tab:orange", "tab:green"]
        markers = ["o", "s", "^"]
        
        for i, (metric_type, metric_names) in enumerate(metrics.items()):
            for j, metric_name in enumerate(metric_names):
                model = "RF" if "(RF)" in metric_name else ("NN" if "(NN)" in metric_name else "LR")
                values = results_df.loc[metric_name, attributes].tolist() if metric_name in results_df.index else [0] * len(attributes)
                
                ax.plot(
                    [40, 50, 60, 70, 80, 90], values,
                    marker=markers[j], color=colors[i],
                    linestyle='-', linewidth=2,
                    label=f"{model} {metric_type}"
                )
        
        ax.set_xlabel("Protected Attributes", fontsize=14)
        ax.set_ylabel("Metric Value", fontsize=14)
        ax.set_title(f"AIF360 Fairness Metrics for Repaired Data ({section})", fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Repaired_AIF360_Metrics_{section}.png"))
        plt.close()

# Evaluate Logistic Regression Model
def evaluate_logistic_regression(clf, X_test):
    pred_test = clf.predict(X_test)
    return pred_test

# List of denial reason columns
DENIAL_REASON_COLUMNS = [
    'denial_reason-1_1', 'denial_reason-1_2', 'denial_reason-1_3', 'denial_reason-1_4', 'denial_reason-1_5',
    'denial_reason-1_6', 'denial_reason-1_7', 'denial_reason-1_8', 'denial_reason-1_9', 'denial_reason-1_10',
    'denial_reason-1_1111', 'denial_reason-2_1.0', 'denial_reason-2_2.0', 'denial_reason-2_3.0', 'denial_reason-2_4.0',
    'denial_reason-2_5.0', 'denial_reason-2_6.0', 'denial_reason-2_7.0', 'denial_reason-2_8.0', 'denial_reason-2_9.0',
    'denial_reason-3_1.0', 'denial_reason-3_2.0', 'denial_reason-3_3.0', 'denial_reason-3_4.0', 'denial_reason-3_5.0',
    'denial_reason-3_6.0', 'denial_reason-3_7.0', 'denial_reason-3_8.0', 'denial_reason-3_9.0', 'denial_reason-4_1.0',
    'denial_reason-4_2.0', 'denial_reason-4_3.0', 'denial_reason-4_4.0', 'denial_reason-4_5.0', 'denial_reason-4_6.0',
    'denial_reason-4_7.0', 'denial_reason-4_8.0', 'denial_reason-4_9.0'
]

# 2024 changes-adding protected attributes
PROTECTED_ATTRIBUTES = [
    'tract_minority_percentage40_0',
    'tract_minority_percentage50_0',
    'tract_minority_percentage60_0',
    'tract_minority_percentage70_0',
    'tract_minority_percentage80_0',
    'tract_minority_percentage90_0'
]

def align_predictions(predictions, df_length):
    """
    Align predictions with the DataFrame length by padding or trimming.
    """
    if len(predictions) < df_length:
        # Convert predictions to float to allow NaN padding
        predictions = predictions.astype(float)
        # Pad predictions with NaN to match the DataFrame length
        padding_length = df_length - len(predictions)
        padded_predictions = np.pad(predictions, (0, padding_length), mode='constant', constant_values=np.nan)
        return padded_predictions
    else:
        # Trim predictions to match the DataFrame length
        return predictions[:df_length]

# Evaluate Logistic Regression for Denial Reason Prediction
def evaluate_logistic_regression_denial(clf, X_test):
    pred_test_denial = clf.predict(X_test)
    return pred_test_denial

@nvtx.annotate("compute_denial_reason_counts", color="orange")
def compute_denial_reason_counts(infile):
    """
    Compute the count of each denial reason with value 1 in the provided file.
    Args:
        infile (str): Path to the input CSV file.
    Returns:
        denial_counts_df (pd.DataFrame): DataFrame with counts for each denial reason column where value is 1.
    """
    # Read only the specified columns related to denial reasons
    df = pd.read_csv(infile, usecols=DENIAL_REASON_COLUMNS)

    # Initialize a dictionary to hold the counts, summing values where each column has value 1
    denial_counts = {column: (df[column] == 1).sum() for column in DENIAL_REASON_COLUMNS}

    # Convert the dictionary to a DataFrame
    denial_counts_df = pd.DataFrame(list(denial_counts.items()), columns=["Denial Reason", "Count"])

    # Save counts to CSV
    denial_counts_df.to_csv("denial_reason_counts.csv", index=False)
    print("Denial reason counts saved to 'denial_reason_counts.csv'.")

    return denial_counts_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explore mortgage fairness and performance tradeoffs')
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('infile',
                        type=argparse.FileType('r'),
                        help='The preprocessed data file')
    parser.add_argument('model',
                        nargs='?',
                        default=None,
                        type=argparse.FileType('r'),
                        help='The trained model to use')

    args = parser.parse_args()
    # # Compute and print denial reason counts
    # denial_counts_df = compute_denial_reason_counts(args.infile.name)
    main(args.infile, args.model, args.verbose)
    # Compute and print denial reason counts
    denial_counts_df = compute_denial_reason_counts(args.infile.name)