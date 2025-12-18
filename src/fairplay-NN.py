import cupy as cp
import cudf
import cuml
from cuml.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import sklearn.ensemble as ske
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import matplotlib.pyplot as plt

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
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

# Neural Network model for Loan Approval Prediction (Binary Classification)
class LoanApprovalNN(nn.Module):
    def __init__(self, input_dim):
        super(LoanApprovalNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)  # Output layer for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))  # Sigmoid activation for binary output
        return x

# Neural Network for Multi-Class Denial Reason Classification
class DenialReasonNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DenialReasonNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)  # Output layer for multi-class classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.output(x)  # No activation; logits are used with CrossEntropyLoss
        return x

def train_neural_network(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
    return model

def evaluate_neural_network(model, test_loader, binary=True) -> tuple[list, list]:
    """
    Evaluate a neural network model.

    Args:
        model (nn.Module): The neural network model.
        test_loader (DataLoader): The DataLoader for the test dataset.
        binary (bool): Whether the output is binary classification.

    Returns:
        predictions (list): Predicted labels or values.
        true_labels (list): True labels if provided, else an empty list.
    """
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            # Check if labels are provided
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, labels = batch
                true_labels.extend(labels.numpy())
            else:
                inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
                true_labels = []  # No labels available

            outputs = model(inputs)
            if binary:
                preds = (outputs.squeeze() >= 0.5).float()  # Threshold at 0.5
            else:
                preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.numpy())
    return predictions, true_labels    
# Neural Network Changes ends

#@nvtx.annotate("balance", color="blue")
#def balance(dataset_orig):
#    if dataset_orig.empty:
#        return dataset_orig
#    print('imbalanced data:\n', dataset_orig['action_taken'].value_counts())
#    action_df = dataset_orig['action_taken'].value_counts()
#    maj_label = action_df.index[0]
#    print('maj_label',maj_label) #
#    min_label = action_df.index[-1]
#    print('min_label',min_label) #
#    df_majority = dataset_orig[dataset_orig.action_taken == maj_label]
#    df_minority = dataset_orig[dataset_orig.action_taken == min_label]

#    df_majority_downsampled = df_majority.sample(n=len(df_minority.index),
#                                                 replace=False,
#                                                 random_state=123)
    # Combine minority class with downsampled majority class
#    df_downsampled = cudf.concat([df_majority_downsampled, df_minority])

#    df_downsampled.reset_index(drop=True, inplace=True)

#    print('balanced data:\n', df_downsampled['action_taken'].value_counts())

#    print('processed data: ' + str(df_downsampled.shape))

#    return df_downsampled

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

    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))

    return metrics

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

# 2024 changes-Creates a dataset with all columns, predictors, original approval, and unbiased approval based on protected attributes.
# Updated compute_unbiased_approvals function with denial_reason_final logic

def compute_unbiased_approvals(df, predictors, protected_attributes):
    new_dataset = []
    denial_reason_final = []  # New column for denial reason based on conditions

    for index, row in df.iterrows():
        # Extract the predictor values and original approved value
        predictor_values = [row[p] for p in predictors]
        original_approved = row['approved']
        
        # Initialize the new row with predictor values and the original approved value
        new_row = predictor_values + [original_approved]
        
        # Determine 'denial_reason_final' based on conditions
        if original_approved == 1:
            denial_reason_final.append(1)
        # denial reason 2 is Debt-to-income ratio
        elif row['denial_reason-1_1'] == 1 or row['denial_reason-2_1.0'] == 1 or row['denial_reason-3_1.0'] == 1 or row['denial_reason-4_1.0'] == 1:
            denial_reason_final.append(2)
        # denial reason 3 is Collateral    
        elif row['denial_reason-1_4'] == 1 or row['denial_reason-2_4.0'] == 1 or row['denial_reason-3_4.0'] == 1 or row['denial_reason-4_4.0'] == 1:
            denial_reason_final.append(3)
        # denial reason 4 is Insufficient cash (downpayment, closing costs)    
        elif row['denial_reason-1_5'] == 1 or row['denial_reason-2_5.0'] == 1 or row['denial_reason-3_5.0'] == 1 or row['denial_reason-4_5.0'] == 1:
            denial_reason_final.append(4)
        # Denied but doesn't fall under other conditions
        else:
            denial_reason_final.append(5)

        # For each protected attribute, calculate and append the unbiased approval value
        for attribute in protected_attributes:
            unbiased_approved_value = row[attribute]
            new_row.append(unbiased_approved_value)

        # Add this row to the new dataset
        new_dataset.append(new_row)

    # Create a DataFrame from the new dataset with the proper columns
    columns = predictors + ['Original Approved'] + [f'{attr}_approved' for attr in protected_attributes]
    unbiased_df = pd.DataFrame(new_dataset, columns=columns)
    
    # Add 'denial_reason_final' to the final dataframe
    unbiased_df['denial_reason_final'] = denial_reason_final

    # Merge with the original dataframe (keeping all original columns)
    final_df = pd.concat([df.reset_index(drop=True), unbiased_df.reset_index(drop=True)], axis=1)

    return final_df

# 2024 changes-Compute descriptive statistics based on conditions and return as a DataFrame.
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
def add_nn_results_to_csv(unbiased_dataset_path, approval_nn, scaler, columns_to_retain, denial_conditions):
    # Load the existing dataset
    df = pd.read_csv(unbiased_dataset_path)

    # Extract features for NN predictions
    X = df[columns_to_retain]
    X_scaled = scaler.transform(X)

    # Prepare DataLoader for Neural Network predictions
    loader = DataLoader(TensorDataset(torch.tensor(X_scaled, dtype=torch.float32)), batch_size=64, shuffle=False)

    # Get predictions from the neural network
    nn_predictions, _ = evaluate_neural_network(approval_nn, loader, binary=True)
    nn_predictions = pd.Series(nn_predictions, name='Original_Approved_nn')

    # Add the `Original_Approved_nn` column to the DataFrame
    df['Original_Approved_nn'] = nn_predictions.values

    # Compute `denial_reason_final_nn` based on conditions
    denial_reason_final_nn = []
    for _, row in df.iterrows():
        if row['Original_Approved_nn'] == 1:
            denial_reason_final_nn.append(1)
        elif any(row[col] == 1 for col in denial_conditions['debt_to_income_ratio']):
            denial_reason_final_nn.append(2)
        elif any(row[col] == 1 for col in denial_conditions['collateral']):
            denial_reason_final_nn.append(3)
        elif any(row[col] == 1 for col in denial_conditions['insufficient_cash']):
            denial_reason_final_nn.append(4)
        else:
            denial_reason_final_nn.append(5)
    df['denial_reason_final_nn'] = denial_reason_final_nn

    # Add new `_nn` columns for protected attributes
    nn_approved_columns = [f"{col}_nn" for col in df.columns if col.endswith('_approved')]
    for col in nn_approved_columns:
        df[col] = nn_predictions.values

    # Save the updated dataset
    updated_path = unbiased_dataset_path.replace(".csv", "_updated.csv")
    df.to_csv(updated_path, index=False)
    print(f"Updated dataset saved to {updated_path}")
@nvtx.annotate("main", color="green")
def main(infile, trained_model=None, verbose=False):
    print("main enters")
    chunks = []
    chunk_size = 50000

    # Load and filter data by activity year
    for chunk in pd.read_csv(infile, index_col=0, chunksize=chunk_size):
        filtered_chunk = chunk[chunk['activity_year_2018'] == 1]
        chunks.append(filtered_chunk)    

    preprocessed_df = pd.concat(chunks, ignore_index=True)
    # After loading and filtering by activity_year_2018
    print(f"Data size after filtering by activity_year_2018: {len(preprocessed_df)}")

    preprocessed_df = preprocessed_df.astype('float32')
    preprocessed_df['approved'] = preprocessed_df['approved'].astype('int32')

    # Drop unnecessary columns
    preprocessed_df.drop(columns=['activity_year_2018', 'activity_year_2019', 'activity_year_2020', 'activity_year_2021'], axis=1, inplace=True)
    preprocessed_df.fillna(0, inplace=True)

    # 2024 changes
    # Drop rows based on specific conditions
    conditions = {
        "loan_type_1": 1,
        "loan_purpose_1": 1,
        "lien_status_1": 1,
        "reverse_mortgage_2": 1,
        "open-end_line_of_credit_2": 1,
        "business_or_commercial_purpose_2": 1,
        # "construction_method_1": 0,
        "occupancy_type_1": 1,
        # "manufactured_home_secured_property_type_3": 0,
        # "total_units": "not equals to 1",
        # "preapproval_2": 0,
        # "negative_amortization_2": 0,
        # "interest_only_payment_2": 0,
        # "balloon_payment_2": 0,
        # "other_nonamortizing_features_2": 0,
        "submission_of_application_1": 1
    }

    for column, condition in conditions.items():
        if column in preprocessed_df.columns:
            if condition == 0:
                preprocessed_df = preprocessed_df[preprocessed_df[column] != 0]
            elif condition == "not equals to 1":
                preprocessed_df = preprocessed_df[preprocessed_df[column] == 1]

    # 2024 changes
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
        # "submission_of_application_1",
        "submission_of_application_1111",
        "submission_of_application_2",
        "submission_of_application_3",
        "total_units",
        # "loan_type_1",
        # "loan_purpose_1",
        # "lien_status_1",
        "reverse_mortgage_1111",
        # "reverse_mortgage_2",
        "open-end_line_of_credit_1111",
        # "open-end_line_of_credit_2",
        "business_or_commercial_purpose_1",
        "business_or_commercial_purpose_1111",
        # "business_or_commercial_purpose_2",
        "construction_method_1",
        # "occupancy_type_1",
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
    # 2024 changes-Compute and save descriptives for 2018 conditions
    all_conditions_count = compute_descriptives(preprocessed_df)
    print("all_conditions_count:\n",all_conditions_count)

    # 2024 changes-After dropping rows based on specific conditions
    print(f"Data size after applying conditions: {len(preprocessed_df)}")

    predictors_df = preprocessed_df[columns_to_retain]
    outcome_df = preprocessed_df[['approved']]

    # 2024 changes-Add binning logic for 'tract_minority_population_percent'
    preprocessed_df['bins'] = pd.cut(preprocessed_df['tract_minority_population_percent'], bins=10, labels=False)

    # 2024 changes-Stratified split based on bins
    df_train, df_test = train_test_split(preprocessed_df, test_size=0.25, stratify=preprocessed_df['bins'], random_state=42)
    # After stratified split
    print(f"Training set size: {len(df_train)}, Test set size: {len(df_test)}")

    # 2024 changes-Remove 'bins' column after splitting
    df_train.drop(columns=['bins'], inplace=True)
    df_test.drop(columns=['bins'], inplace=True)

    # 2024 changes-Check distribution in original data, train set, and test set
    original_dist = preprocessed_df['tract_minority_population_percent'].describe()
    train_dist = df_train['tract_minority_population_percent'].describe()
    test_dist = df_test['tract_minority_population_percent'].describe()

    print("Original Distribution:\n", original_dist)
    print("\nTrain Distribution:\n", train_dist)
    print("\nTest Distribution:\n", test_dist)

    #Neural Network Changes starts

    # Standardize the data and prepare labels for both networks
    scaler = StandardScaler()
    X = preprocessed_df[columns_to_retain]
    X_scaled = scaler.fit_transform(X)

    # Prepare labels
    y_approval = preprocessed_df['approved']
    #y_denial_reason = preprocessed_df['denial_reason_final']

    # Split data and convert to PyTorch tensors
    X_train, X_test, y_train_approval, y_test_approval = train_test_split(X_scaled, y_approval, test_size=0.25, random_state=42)
    #X_train_denial, X_test_denial, y_train_denial, y_test_denial = train_test_split(X_scaled, y_denial_reason, test_size=0.25, random_state=42)

    # Convert to DataLoader format
    train_loader_approval = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_approval.values, dtype=torch.float32)), batch_size=64, shuffle=True)
    test_loader_approval = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test_approval.values, dtype=torch.float32)), batch_size=64, shuffle=False)

    # Initialize and train LoanApprovalNN for loan approval classification
    input_dim = X_train.shape[1]
    print("X_train shape:", X_train.shape)
    print("Expected input_dim:", input_dim)
    approval_nn = LoanApprovalNN(input_dim=input_dim)
    criterion_approval = nn.BCELoss()
    optimizer_approval = optim.Adam(approval_nn.parameters(), lr=0.001)

    # Train the neural network on the approval data
    train_neural_network(approval_nn, train_loader_approval, criterion_approval, optimizer_approval)

    # Evaluate the trained network on training and test data
    nn_train_pred, nn_train_labels = evaluate_neural_network(approval_nn, train_loader_approval)
    nn_train_pred = (np.array(nn_train_pred) >= 0.5).astype(int).reshape((-1, 1))
    nn_test_pred, nn_test_labels = evaluate_neural_network(approval_nn, test_loader_approval)

    # Prepare denial reason data for denied loans only
    # denied_indices_train = y_train_approval == 0
    # denied_indices_test = y_test_approval == 0
    # X_train_denied = X_train[denied_indices_train]
    # y_train_denied = y_train_denial.values[denied_indices_train]
    # X_test_denied = X_test[denied_indices_test]
    # y_test_denied = y_test_denial.values[denied_indices_test]

    # train_loader_denial = DataLoader(TensorDataset(torch.tensor(X_train_denied, dtype=torch.float32), torch.tensor(y_train_denied, dtype=torch.long)), batch_size=64, shuffle=True)
    # test_loader_denial = DataLoader(TensorDataset(torch.tensor(X_test_denied, dtype=torch.float32), torch.tensor(y_test_denied, dtype=torch.long)), batch_size=64, shuffle=False)

    # # Train and evaluate DenialReasonNN
    # denial_nn = DenialReasonNN(X_train.shape[1])
    # criterion_denial = nn.CrossEntropyLoss()
    # optimizer_denial = optim.Adam(denial_nn.parameters(), lr=0.001)
    # train_neural_network(denial_nn, train_loader_denial, criterion_denial, optimizer_denial)
    # nn_train_pred_denial, nn_train_labels_denial = evaluate_neural_network(denial_nn, train_loader_denial)
    # nn_test_pred_denial, nn_test_labels_denial = evaluate_neural_network(denial_nn, test_loader_denial)
    
    #Neural Network Changes ends

    # 2024 changes-Define the predictors and protected attributes
    predictors = ["income", "loan_amount", "loan_to_value_ratio", "property_value", "debt_to_income_ratio", "loan_term"]
    protected_attributes = PROTECTED_ATTRIBUTES

    # 2024 changes-Compute the new dataset with unbiased approvals
    unbiased_df = compute_unbiased_approvals(preprocessed_df, predictors, protected_attributes)

    # Save the new unbiased dataset to CSV
    unbiased_df.to_csv('unbiased_dataset.csv', index=False)
    print("Unbiased dataset saved to 'unbiased_dataset.csv'.")

    denial_conditions = {
    'debt_to_income_ratio': ['denial_reason-1_1', 'denial_reason-2_1.0', 'denial_reason-3_1.0', 'denial_reason-4_1.0'],
    'collateral': ['denial_reason-1_4', 'denial_reason-2_4.0', 'denial_reason-3_4.0', 'denial_reason-4_4.0'],
    'insufficient_cash': ['denial_reason-1_5', 'denial_reason-2_5.0', 'denial_reason-3_5.0', 'denial_reason-4_5.0']
    }
    add_nn_results_to_csv('unbiased_dataset.csv', approval_nn, scaler, columns_to_retain, denial_conditions)

    # # Generate unbiased predictions using Neural Network
    # print("Generating unbiased dataset using Neural Network...")
    # # Prepare data for neural network inference
    # scaler = StandardScaler()
    # X_scaled_full = scaler.fit_transform(preprocessed_df[columns_to_retain])

    # # Create a dummy tensor for labels since they are required by the DataLoader but not used in inference
    # dummy_labels = torch.zeros(X_scaled_full.shape[0], dtype=torch.float32)
    # full_loader = DataLoader(TensorDataset(torch.tensor(X_scaled_full, dtype=torch.float32), dummy_labels), batch_size=64, shuffle=False)

    # # Initialize the neural network model for loan approval classification
    # input_dim = X_scaled_full.shape[1]
    # approval_nn = LoanApprovalNN(input_dim=input_dim)
    # criterion = nn.BCELoss()
    # optimizer = optim.Adam(approval_nn.parameters(), lr=0.001)

    # # Train the model if not trained, or load pre-trained weights if available
    # train_neural_network(approval_nn, full_loader, criterion, optimizer)

    # # Perform inference to get unbiased predictions for the full dataset
    # nn_predictions_full, _ = evaluate_neural_network(approval_nn, full_loader)

    # # Ensure the predictions match the length of the original DataFrame
    # if len(nn_predictions_full) == len(preprocessed_df):
    #     # Add the predictions to the original DataFrame
    #     preprocessed_df['nn_unbiased_approval'] = nn_predictions_full
    # else:
    #     raise ValueError("Prediction length does not match dataset length.")

    # # Create the unbiased dataset using NN predictions
    # unbiased_nn_df = compute_unbiased_approvals(preprocessed_df, predictors, protected_attributes)

    # # Update denial_reason_final logic based on Neural Network predictions
    # denial_reason_final = []
    # for index, row in unbiased_nn_df.iterrows():
    #     if row['approved'] == 1:
    #             denial_reason_final.append(1)
    #     elif row['denial_reason-1_1'] == 1 or row['denial_reason-2_1.0'] == 1 or row['denial_reason-3_1.0'] == 1 or row['denial_reason-4_1.0'] == 1:
    #             denial_reason_final.append(2)
    #     elif row['denial_reason-1_4'] == 1 or row['denial_reason-2_4.0'] == 1 or row['denial_reason-3_4.0'] == 1 or row['denial_reason-4_4.0'] == 1:
    #             denial_reason_final.append(3)
    #     elif row['denial_reason-1_5'] == 1 or row['denial_reason-2_5.0'] == 1 or row['denial_reason-3_5.0'] == 1 or row['denial_reason-4_5.0'] == 1:
    #             denial_reason_final.append(4)
    #     else:
    #             denial_reason_final.append(5)

    # # Add 'denial_reason_final' to the final dataframe
    # unbiased_nn_df['denial_reason_final'] = denial_reason_final

    # # Save the unbiased dataset generated by Neural Network
    # unbiased_nn_df.to_csv('unbiased_dataset_nn.csv', index=False)
    # print("Unbiased dataset using Neural Network saved to 'unbiased_dataset_nn.csv'.")

    # File paths
    # rf_file = "unbiased_dataset.csv"
    # nn_file = "unbiased_dataset_nn.csv"
    # output_file = "merged_unbiased_dataset.csv"

    # # Define shared columns
    # shared_columns = [
    # "activity_year",
    # "denial_reason_final",
    # "tract_minority_percentage40_0",
    # "tract_minority_percentage50_0",
    # "tract_minority_percentage60_0",
    # "tract_minority_percentage70_0",
    # "tract_minority_percentage80_0",
    # "tract_minority_percentage90_0"
    # ]

    # chunk_size = 5000  # Smaller chunks for reduced memory usage
    # intermediate_files = []

    # # Process chunks iteratively
    # rf_reader = pd.read_csv(rf_file, chunksize=chunk_size)
    # nn_reader = pd.read_csv(nn_file, chunksize=chunk_size)

    # for i, (rf_chunk, nn_chunk) in enumerate(zip(rf_reader, nn_reader)):
    #     # Add suffixes
    #     rf_chunk = rf_chunk.add_suffix("_rf")
    #     nn_chunk = nn_chunk.add_suffix("_nn")
    
    #     # Retain shared columns without suffixes
    #     rf_chunk.rename(columns={f"{col}_rf": col for col in shared_columns}, inplace=True)
    #     nn_chunk.rename(columns={f"{col}_nn": col for col in shared_columns}, inplace=True)
    
    #     # Merge chunks
    #     merged_chunk = pd.merge(rf_chunk, nn_chunk, on=shared_columns, how="inner")
    
    #     # Save intermediate result
    #     intermediate_file = f"merged_intermediate_{i}.csv"
    #     merged_chunk.to_csv(intermediate_file, index=False)
    #     intermediate_files.append(intermediate_file)
    #     print(f"Processed chunk {i}, saved to {intermediate_file}")

    # # Combine intermediate files
    # with open(output_file, "w") as outfile:
    #     for i, file in enumerate(intermediate_files):
    #         with open(file, "r") as infile:
    #             if i != 0:
    #                 infile.readline()  # Skip header for subsequent chunks
    #             outfile.write(infile.read())
    #     print(f"Final merged dataset saved to {output_file}")

    # 2024 changes-Initialize the DataFrame for saving results in the specified format (from the original code)
    results_dict = {
        "#### Original training dataset": [],
        "RF Statistical parity difference (Original Train)": [],
        "RF Disparate impact (Original Train)": [],
        "RF Mean difference (Original Train)": [],
        "NN Statistical parity difference (Original Train)": [],
        "NN Disparate impact (Original Train)": [],
        "NN Mean difference (Original Train)": [],
        "#### Reweighted training dataset": [],
        "RF Statistical parity difference (Reweighted Train)": [],
        "RF Disparate impact (Reweighted Train)": [],
        "RF Mean difference (Reweighted Train)": [],
        "NN Statistical parity difference (Reweighted Train)": [],
        "NN Disparate impact (Reweighted Train)": [],
        "NN Mean difference (Reweighted Train)": [],
        "#### Original testing dataset": [],
        "RF Statistical parity difference (Original Test)": [],
        "RF Disparate impact (Original Test)": [],
        "RF Mean difference (Original Test)": [],
        "NN Statistical parity difference (Original Test)": [],
        "NN Disparate impact (Original Test)": [],
        "NN Mean difference (Original Test)": [],
        "#### Reweighted testing dataset": [],
        "RF Statistical parity difference (Reweighted Test)": [],
        "RF Disparate impact (Reweighted Test)": [],
        "RF Mean difference (Reweighted Test)": [],
        "NN Statistical parity difference (Reweighted Test)": [],
        "NN Disparate impact (Reweighted Test)": [],
        "NN Mean difference (Reweighted Test)": [],
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
        "#### Original Training AIF360 Fairness Metrics:": [],
        "Average odds difference (Original Train) (RF)": [],
        "Equal opportunity difference (Original Train) (RF)": [],
        "Theil index (Original Train) (RF)": [],
        "Average odds difference (Original Train) (NN)": [],
        "Equal opportunity difference (Original Train) (NN)": [],
        "Theil index (Original Train) (NN)": [],
        "#### Original Testing AIF360 Fairness Metrics:": [],
        "Average odds difference (Original Test) (RF)": [],
        "Equal opportunity difference (Original Test) (RF)": [],
        "Theil index (Original Test) (RF)": [],
        "Average odds difference (Original Test) (NN)": [],
        "Equal opportunity difference (Original Test) (NN)": [],
        "Theil index (Original Test) (NN)": [],
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
        "#### Unbiased Training AIF360 Fairness Metrics:": [],
        "Average odds difference (Unbiased Train) (RF)": [],
        "Equal opportunity difference (Unbiased Train) (RF)": [],
        "Theil index (Unbiased Train) (RF)": [],
        "Average odds difference (Unbiased Train) (NN)": [],
        "Equal opportunity difference (Unbiased Train) (NN)": [],
        "Theil index (Unbiased Train) (NN)": [],
        "#### Unbiased Testing AIF360 Fairness Metrics:": [],
        "Average odds difference (Unbiased Test) (RF)": [],
        "Equal opportunity difference (Unbiased Test) (RF)": [],
        "Theil index (Unbiased Test) (RF)": [],
        "Average odds difference (Unbiased Test) (NN)": [],
        "Equal opportunity difference (Unbiased Test) (NN)": [],
        "Theil index (Unbiased Test) (NN)": [],
        "Denial Reason Training Accuracy Score (RF)": [],
        "Denial Reason Testing Accuracy Score (RF)": [],
        "Denial Reason Training Accuracy Score (NN)": [],
        "Denial Reason Testing Accuracy Score (NN)": [],
        "Execution_Time_unbiased (Hours)": [],
        "Execution_Time_orig (Hours)": [],
        "Energy_Unbiased (kWh)": [],
        "Energy_Orig (kWh)": [],
    }
    # Prepare denial reason data for denied loans only
    denial_columns = [
    'denial_reason-1_1', 'denial_reason-1_2', 'denial_reason-1_3', 'denial_reason-1_4', 'denial_reason-1_5',
    'denial_reason-1_6', 'denial_reason-1_7', 'denial_reason-1_8', 'denial_reason-1_9', 'denial_reason-1_10'
    ]

    # Combine denial reason columns to create the true denial reason labels
    df_train_denial = df_train[df_train['approved'] == 0]
    df_test_denial = df_test[df_test['approved'] == 0]

    y_train_denial_reason = df_train_denial[denial_columns].idxmax(axis=1).str.split('_').str[-1].astype(int)
    y_test_denial_reason = df_test_denial[denial_columns].idxmax(axis=1).str.split('_').str[-1].astype(int)

    X_train_denial = df_train_denial[columns_to_retain]
    X_test_denial = df_test_denial[columns_to_retain]

    # Train Random Forest for denial reasons
    clf_rf_denial = ske.RandomForestClassifier(n_estimators=120, random_state=11850)
    clf_rf_denial.fit(X_train_denial, y_train_denial_reason)

    # Predict denial reasons using Random Forest
    rf_denial_train_pred = clf_rf_denial.predict(X_train_denial)
    rf_denial_test_pred = clf_rf_denial.predict(X_test_denial)

    # Train Neural Network for denial reasons
    num_classes = len(denial_columns)
    denial_nn = DenialReasonNN(input_dim=X_train_denial.shape[1], num_classes=num_classes)
    criterion_denial = nn.CrossEntropyLoss()
    optimizer_denial = optim.Adam(denial_nn.parameters(), lr=0.001)

    # Convert to DataLoader for Neural Network training
    train_loader_denial = DataLoader(
        TensorDataset(torch.tensor(X_train_denial.values, dtype=torch.float32), torch.tensor(y_train_denial_reason.values, dtype=torch.long)),
        batch_size=64, shuffle=True
    )
    test_loader_denial = DataLoader(
        TensorDataset(torch.tensor(X_test_denial.values, dtype=torch.float32), torch.tensor(y_test_denial_reason.values, dtype=torch.long)),
        batch_size=64, shuffle=False
    )

    # Train the neural network
    train_neural_network(denial_nn, train_loader_denial, criterion_denial, optimizer_denial)

    # Evaluate the trained network
    nn_denial_train_pred, nn_denial_train_labels = evaluate_neural_network(denial_nn, train_loader_denial, binary=False)
    nn_denial_test_pred, nn_denial_test_labels = evaluate_neural_network(denial_nn, test_loader_denial, binary=False)

    # Denial reason accuracy scores
    rf_denial_train_accuracy = accuracy_score(y_train_denial_reason, rf_denial_train_pred)
    rf_denial_test_accuracy = accuracy_score(y_test_denial_reason, rf_denial_test_pred)
    nn_denial_train_accuracy = accuracy_score(nn_denial_train_labels, nn_denial_train_pred)
    nn_denial_test_accuracy = accuracy_score(nn_denial_test_labels, nn_denial_test_pred)

    #2024 changes- Looping around protected attributes
    for attribute in PROTECTED_ATTRIBUTES:
        print(f"Processing for protected attribute: {attribute}")

        if attribute not in preprocessed_df.columns:
            continue

        # Start timing for this iteration
        start_time_reweighing = datetime.now()    

        unprivileged_groups = [{attribute: 0}]
        privileged_groups = [{attribute: 1}]

        df_train_bld = BinaryLabelDataset(df=df_train,
                                          label_names=['approved'],
                                          protected_attribute_names=[attribute],
                                          favorable_label=1, unfavorable_label=0)

        df_test_bld = BinaryLabelDataset(df=df_test,
                                         label_names=['approved'],
                                         protected_attribute_names=[attribute],
                                         favorable_label=1, unfavorable_label=0)

        df_train_unbiased_bld, train_metrics = reweighing(df_train_bld, unprivileged_groups, privileged_groups)

        df_test_unbiased_bld, test_metrics = reweighing(bld=df_test_bld,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups,
                                                        training=False)

        # End timing for this iteration
        end_time_reweighing = datetime.now()
        execution_time_reweighing = (end_time_reweighing - start_time_reweighing).total_seconds() / 3600  # T in hours

        # Formula for computing Energy, Assuming T is in hours Assume 10 CPU cores and 1 GPU for A100
        energy_reweighing = (((execution_time_reweighing / 24) / 365) * 57754)/1000

        # Start timing for this iteration
        start_time_orig = datetime.now()

        X_train_orig = df_train[columns_to_retain]
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
        train_loader_unbiased_nn = DataLoader(TensorDataset(torch.tensor(df_train_unbiased_bld.features, dtype=torch.float32),torch.tensor(df_train_unbiased_bld.labels, dtype=torch.float32)),batch_size=64,shuffle=True)
        test_loader_unbiased_nn = DataLoader(TensorDataset(torch.tensor(df_test_unbiased_bld.features, dtype=torch.float32),torch.tensor(df_test_unbiased_bld.labels, dtype=torch.float32)),batch_size=64,shuffle=False)

        # Train and test the original model
        if verbose:
            # Slow: runs on CPU because GPU version doesn't have feature importance
            importance_rf = ske.RandomForestClassifier(n_estimators=120, random_state=11850)
            importance_rf.fit(X_train_orig, pd.Series(y_train_orig))
            importances = importance_rf.feature_importances_
            sorted_idx = importances.argsort()

            print('\n===========================')
            print('Original Feature Importance')
            print('===========================')
            for feature, val in zip(X_train_orig.columns[sorted_idx], importances[sorted_idx]):
                print('Feature: {}, Score: {:.4f}%'.format(feature, val * 100))

            print('===========================\n')

        X_train_orig_gdf = cudf.from_pandas(X_train_orig)
        y_train_orig_cp = cudf.Series(y_train_orig)

        X_test_orig_gdf = cudf.from_pandas(X_test_orig)

        # Faster: runs on GPU
        clf_orig = RandomForestClassifier(n_estimators=120, random_state=11850)
        clf_orig.fit(X_train_orig_gdf, y_train_orig_cp)

        pred_train_orig = clf_orig.predict(X_train_orig_gdf)
        pred_test_orig = clf_orig.predict(X_test_orig_gdf)

        predictions1 = cp.asnumpy(pred_train_orig)
        predictions2 = cp.asnumpy(pred_test_orig)

        # Clear GPU memory
        del clf_orig
        del X_train_orig_gdf, y_train_orig_cp, X_test_orig_gdf

        print('The Original Training Accuracy Score: ', accuracy_score(y_train_orig, predictions1))
        print('The Original Testing Accuracy Score: ', accuracy_score(y_test_orig, predictions2))

        print('The Original Training F1 Score: ', f1_score(y_train_orig, predictions1))
        print('The Original Testing F1 Score: ', f1_score(y_test_orig, predictions2))

        print('The Original Training Recall Score: ', recall_score(y_train_orig, predictions1))
        print('The Original Testing Recall Score: ', recall_score(y_test_orig, predictions2))

        print('The Original Training Precision Score: ', precision_score(y_train_orig, predictions1))
        print('The Original Testing Precision Score: ', precision_score(y_test_orig, predictions2))

        print('\n#### Original Training AIF360 Fairness Metrics:')
        df_train_bld_pred = df_train_bld.copy()
        df_train_bld_pred.labels = np.reshape(predictions1, (-1, 1))

        metric_train_orig = compute_metrics(df_train_bld, df_train_bld_pred,
                                            unprivileged_groups, privileged_groups,
                                            disp=True)

        print('\n#### Original Testing AIF360 Fairness Metrics:')
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
            importance_rf = ske.RandomForestClassifier(n_estimators=120, random_state=11850)
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
            clf_unbiased = load(trained_model.name)
        else:
            # Faster: runs on GPU
            clf_unbiased = RandomForestClassifier(n_estimators=120, random_state=11850)
            clf_unbiased.fit(X_train_unbiased_gdf, y_train_unbiased_cp)

            # print("Saving trained model...\n")
            # dump(clf_unbiased, './trained_model_{}.model'.format(int(time.time())))

        pred_train_unbiased = clf_unbiased.predict(X_train_unbiased_gdf)
        pred_test_unbiased = clf_unbiased.predict(X_test_unbiased_gdf)

        predictions1_unbiased = cp.asnumpy(pred_train_unbiased)
        predictions2_unbiased = cp.asnumpy(pred_test_unbiased)

        # Clear GPU memory
        del clf_unbiased
        del X_train_unbiased_gdf, y_train_unbiased_cp, X_test_unbiased_gdf

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

        df_test_unbiased_bld_pred = df_test_unbiased_bld.copy()
        df_test_unbiased_bld_pred.labels = np.reshape(predictions2_unbiased, (-1, 1))
        metric_test_unbiased = compute_metrics(df_test_unbiased_bld, df_test_unbiased_bld_pred, unprivileged_groups, privileged_groups, disp=False)

        #Neural Network Changes starts
        # Convert to PyTorch Tensors for Neural Network
        train_data = torch.tensor(df_train[columns_to_retain].values, dtype=torch.float32)
        train_labels = torch.tensor(df_train['approved'].values, dtype=torch.float32)
        test_data = torch.tensor(df_test[columns_to_retain].values, dtype=torch.float32)
        test_labels = torch.tensor(df_test['approved'].values, dtype=torch.float32)

        # Train Neural Network on Original Data
        approval_nn = LoanApprovalNN(input_dim=train_data.shape[1])
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
        nn_train_pred = (np.array(nn_train_pred) >= 0.5).astype(int).reshape((-1, 1))
        df_train_bld_pred_nn.labels = nn_train_pred
        metric_train_nn = compute_metrics(df_train_bld, df_train_bld_pred_nn, unprivileged_groups, privileged_groups, disp=False)

        # Similarly for test predictions
        nn_test_pred = (np.array(nn_test_pred) >= 0.5).astype(int).reshape((-1, 1))
        df_test_bld_pred_nn.labels = nn_test_pred
        metric_test_nn = compute_metrics(df_test_bld, df_test_bld_pred_nn, unprivileged_groups, privileged_groups, disp=False)

        # Neural Network Predictions on Reweighted Data
        nn_train_pred_unbiased, nn_train_labels_unbiased = evaluate_neural_network(approval_nn, DataLoader(TensorDataset(torch.tensor(df_train_unbiased_bld.features, dtype=torch.float32), torch.tensor(df_train_unbiased_bld.labels, dtype=torch.float32)), batch_size=64))
        nn_test_pred_unbiased, nn_test_labels_unbiased = evaluate_neural_network(approval_nn, DataLoader(TensorDataset(torch.tensor(df_test_unbiased_bld.features, dtype=torch.float32), torch.tensor(df_test_unbiased_bld.labels, dtype=torch.float32)), batch_size=64))

        # Convert reweighted predictions to BinaryLabelDataset format for fairness metrics
        df_train_unbiased_bld_pred_nn = df_train_unbiased_bld.copy()
        df_train_unbiased_bld_pred_nn.labels = np.reshape(nn_train_pred_unbiased, (-1, 1))
        df_test_unbiased_bld_pred_nn = df_test_unbiased_bld.copy()
        df_test_unbiased_bld_pred_nn.labels = np.reshape(nn_test_pred_unbiased, (-1, 1))

        # Calculate fairness metrics on the reweighted training and testing data
        metric_train_unbiased_nn = compute_metrics(df_train_unbiased_bld, df_train_unbiased_bld_pred_nn, unprivileged_groups, privileged_groups, disp=False)
        metric_test_unbiased_nn = compute_metrics(df_test_unbiased_bld, df_test_unbiased_bld_pred_nn, unprivileged_groups, privileged_groups, disp=False)

        # 2024 changes-Append metrics to the results dictionary
        results_dict["#### Original training dataset"].append("")
        results_dict["RF Statistical parity difference (Original Train)"].append(train_metrics["Statistical parity difference"])
        results_dict["RF Disparate impact (Original Train)"].append(train_metrics["Disparate impact"])
        results_dict["RF Mean difference (Original Train)"].append(train_metrics["Mean difference"])
        results_dict["NN Statistical parity difference (Original Train)"].append(metric_train_nn["Statistical parity difference"])
        results_dict["NN Disparate impact (Original Train)"].append(metric_train_nn["Disparate impact"])
        results_dict["NN Mean difference (Original Train)"].append(metric_train_nn["Mean difference"])
        results_dict["#### Reweighted training dataset"].append("")
        results_dict["RF Statistical parity difference (Reweighted Train)"].append(train_metrics["Statistical parity difference (Reweighted)"])
        results_dict["RF Disparate impact (Reweighted Train)"].append(train_metrics["Disparate impact (Reweighted)"])
        results_dict["RF Mean difference (Reweighted Train)"].append(train_metrics["Mean difference (Reweighted)"])
        results_dict["NN Statistical parity difference (Reweighted Train)"].append(metric_train_unbiased_nn["Statistical parity difference"])
        results_dict["NN Disparate impact (Reweighted Train)"].append(metric_train_unbiased_nn["Disparate impact"])
        results_dict["NN Mean difference (Reweighted Train)"].append(metric_train_unbiased_nn["Theil index"])
        results_dict["#### Original testing dataset"].append("")
        results_dict["RF Statistical parity difference (Original Test)"].append(test_metrics["Statistical parity difference"])
        results_dict["RF Disparate impact (Original Test)"].append(test_metrics["Disparate impact"])
        results_dict["RF Mean difference (Original Test)"].append(test_metrics["Mean difference"])
        results_dict["NN Statistical parity difference (Original Test)"].append(metric_test_nn["Statistical parity difference"])
        results_dict["NN Disparate impact (Original Test)"].append(metric_test_nn["Disparate impact"])
        results_dict["NN Mean difference (Original Test)"].append(metric_test_nn["Mean difference"])
        results_dict["#### Reweighted testing dataset"].append("")
        results_dict["RF Statistical parity difference (Reweighted Test)"].append(test_metrics["Statistical parity difference (Reweighted)"])
        results_dict["RF Disparate impact (Reweighted Test)"].append(test_metrics["Disparate impact (Reweighted)"])
        results_dict["RF Mean difference (Reweighted Test)"].append(test_metrics["Mean difference (Reweighted)"])
        results_dict["NN Statistical parity difference (Reweighted Test)"].append(metric_test_unbiased_nn["Statistical parity difference"])
        results_dict["NN Disparate impact (Reweighted Test)"].append(metric_test_unbiased_nn["Equal opportunity difference"])
        results_dict["NN Mean difference (Reweighted Test)"].append(metric_test_unbiased_nn["Theil index"])
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
        results_dict["#### Original Training AIF360 Fairness Metrics:"].append("")
        results_dict["Average odds difference (Original Train) (RF)"].append(metric_train_orig["Average odds difference"])
        results_dict["Equal opportunity difference (Original Train) (RF)"].append(metric_train_orig["Equal opportunity difference"])
        results_dict["Theil index (Original Train) (RF)"].append(metric_train_orig["Theil index"])
        results_dict["Average odds difference (Original Train) (NN)"].append(metric_train_nn["Average odds difference"])
        results_dict["Equal opportunity difference (Original Train) (NN)"].append(metric_train_nn["Equal opportunity difference"])
        results_dict["Theil index (Original Train) (NN)"].append(metric_train_nn["Theil index"])
        results_dict["#### Original Testing AIF360 Fairness Metrics:"].append("")
        results_dict["Average odds difference (Original Test) (RF)"].append(metric_test_orig["Average odds difference"])
        results_dict["Equal opportunity difference (Original Test) (RF)"].append(metric_test_orig["Equal opportunity difference"])
        results_dict["Theil index (Original Test) (RF)"].append(metric_test_orig["Theil index"])
        results_dict["Average odds difference (Original Test) (NN)"].append(metric_test_nn["Average odds difference"])
        results_dict["Equal opportunity difference (Original Test) (NN)"].append(metric_test_nn["Equal opportunity difference"])
        results_dict["Theil index (Original Test) (NN)"].append(metric_test_nn["Theil index"])
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
        results_dict["#### Unbiased Training AIF360 Fairness Metrics:"].append("")
        results_dict["Average odds difference (Unbiased Train) (RF)"].append(metric_train_unbiased["Average odds difference"])
        results_dict["Equal opportunity difference (Unbiased Train) (RF)"].append(metric_train_unbiased["Equal opportunity difference"])
        results_dict["Theil index (Unbiased Train) (RF)"].append(metric_train_unbiased["Theil index"])
        results_dict["Average odds difference (Unbiased Train) (NN)"].append(metric_train_unbiased_nn["Average odds difference"])
        results_dict["Equal opportunity difference (Unbiased Train) (NN)"].append(metric_train_unbiased_nn["Equal opportunity difference"])
        results_dict["Theil index (Unbiased Train) (NN)"].append(metric_train_unbiased_nn["Theil index"])
        results_dict["#### Unbiased Testing AIF360 Fairness Metrics:"].append("")
        results_dict["Average odds difference (Unbiased Test) (RF)"].append(metric_test_unbiased["Average odds difference"])
        results_dict["Equal opportunity difference (Unbiased Test) (RF)"].append(metric_test_unbiased["Equal opportunity difference"])
        results_dict["Theil index (Unbiased Test) (RF)"].append(metric_test_unbiased["Theil index"])
        results_dict["Average odds difference (Unbiased Test) (NN)"].append(metric_test_unbiased_nn["Average odds difference"])
        results_dict["Equal opportunity difference (Unbiased Test) (NN)"].append(metric_test_unbiased_nn["Equal opportunity difference"])
        results_dict["Theil index (Unbiased Test) (NN)"].append(metric_test_unbiased_nn["Theil index"])
        results_dict["Denial Reason Training Accuracy Score (RF)"].append(rf_denial_train_accuracy)
        results_dict["Denial Reason Testing Accuracy Score (RF)"].append(rf_denial_test_accuracy)
        results_dict["Denial Reason Training Accuracy Score (NN)"].append(nn_denial_train_accuracy)
        results_dict["Denial Reason Testing Accuracy Score (NN)"].append(nn_denial_test_accuracy)
        # Append T and Energy to results_dict
        results_dict["Execution_Time_unbiased (Hours)"].append(total_execution_time_unbiased)
        results_dict["Execution_Time_orig (Hours)"].append(total_execution_time_orig)
        results_dict["Energy_Unbiased (kWh)"].append(total_energy_unbiased)
        results_dict["Energy_Orig (kWh)"].append(total_energy_orig)

    # Transpose the DataFrame so that metrics are rows and attributes are columns
    results_df = pd.DataFrame(results_dict, index=PROTECTED_ATTRIBUTES).T

    # 2024 changes-Save the performance results to CSV
    results_df.to_csv('performance_results_activity_year_2018.csv')

    print("Results saved to CSV.")

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