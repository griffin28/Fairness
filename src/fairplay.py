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

    # The difference in true positive rates between unprivileged and privileged groups. A value of 0 implies both groups
    # have equal benefit, a value less than 0 implies higher benefit for the privileged group and a value greater than 0
    # implies higher benefit for the unprivileged group.
    # Needs to use input and output datasets to a classifier (ClassificationMetric)
    metrics["Equal opportunity difference"] = classified_metric_pred.equal_opportunity_difference()

    # generalized_entropy_index with alpha = 1.
    metrics["Theil index"] = classified_metric_pred.theil_index()

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

    # 2024 changes-Define the predictors and protected attributes
    predictors = ["income", "loan_amount", "loan_to_value_ratio", "property_value", "debt_to_income_ratio", "loan_term"]
    protected_attributes = PROTECTED_ATTRIBUTES

    # 2024 changes-Compute the new dataset with unbiased approvals
    unbiased_df = compute_unbiased_approvals(preprocessed_df, predictors, protected_attributes)

    # Save the new unbiased dataset to CSV
    unbiased_df.to_csv('unbiased_dataset.csv', index=False)
    print("Unbiased dataset saved to 'unbiased_dataset.csv'.")

    # 2024 changes-Initialize the DataFrame for saving results in the specified format (from the original code)
    results_dict = {
        "#### Original training dataset": [],
        "Statistical parity difference (Original Train)": [],
        "Disparate impact (Original Train)": [],
        "Mean difference (Original Train)": [],
        "#### Reweighted training dataset": [],
        "Statistical parity difference (Reweighted Train)": [],
        "Disparate impact (Reweighted Train)": [],
        "Mean difference (Reweighted Train)": [],
        "#### Original testing dataset": [],
        "Statistical parity difference (Original Test)": [],
        "Disparate impact (Original Test)": [],
        "Mean difference (Original Test)": [],
        "#### Reweighted testing dataset": [],
        "Statistical parity difference (Reweighted Test)": [],
        "Disparate impact (Reweighted Test)": [],
        "Mean difference (Reweighted Test)": [],
        "The Original Training Accuracy Score": [],
        "The Original Testing Accuracy Score": [],
        "The Original Training F1 Score": [],
        "The Original Testing F1 Score": [],
        "The Original Training Recall Score": [],
        "The Original Testing Recall Score": [],
        "The Original Training Precision Score": [],
        "The Original Testing Precision Score": [],
        "#### Original Training AIF360 Fairness Metrics:": [],
        "Average odds difference (Original Train)": [],
        "Equal opportunity difference (Original Train)": [],
        "Theil index (Original Train)": [],
        "#### Original Testing AIF360 Fairness Metrics:": [],
        "Average odds difference (Original Test)": [],
        "Equal opportunity difference (Original Test)": [],
        "Theil index (Original Test)": [],
        "The Unbiased Training Accuracy Score": [],
        "The Unbiased Testing Accuracy Score": [],
        "The Unbiased Training F1 Score": [],
        "The Unbiased Testing F1 Score": [],
        "The Unbiased Training Recall Score": [],
        "The Unbiased Testing Recall Score": [],
        "The Unbiased Training Precision Score": [],
        "The Unbiased Testing Precision Score": [],
        "#### Unbiased Training AIF360 Fairness Metrics:": [],
        "Average odds difference (Unbiased Train)": [],
        "Equal opportunity difference (Unbiased Train)": [],
        "Theil index (Unbiased Train)": [],
        "#### Unbiased Testing AIF360 Fairness Metrics:": [],
        "Average odds difference (Unbiased Test)": [],
        "Equal opportunity difference (Unbiased Test)": [],
        "Theil index (Unbiased Test)": [],
        "Execution_Time_unbiased (Hours)": [],
        "Execution_Time_orig (Hours)": [],
        "Energy_Unbiased (kWh)": [],
        "Energy_Orig (kWh)": []
    }
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

        # Get reweighted datasets and metrics
        df_train_unbiased_bld, train_metrics = reweighing(bld=df_train_bld,
                                                          unprivileged_groups=unprivileged_groups,
                                                          privileged_groups=privileged_groups)

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
        X_train_unbiased = train_unbiased_df[columns_to_retain]
        y_train_unbiased = df_train_unbiased_bld.labels.ravel()

        test_unbiased_df = df_test_unbiased_bld.convert_to_dataframe()[0]
        X_test_unbiased = test_unbiased_df[columns_to_retain]
        y_test_unbiased = df_test_unbiased_bld.labels.ravel()

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

        X_train_orig_gdf = cudf.from_pandas(X_train_orig)
        y_train_orig_cp = cudf.Series(y_train_orig)

        X_test_orig_gdf = cudf.from_pandas(X_test_orig)

        # Faster: runs on GPU
        clf_orig = RandomForestClassifier(n_estimators=120, random_state=np.random.seed(11850))
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
            clf_unbiased = load(trained_model.name)
        else:
            # Faster: runs on GPU
            clf_unbiased = RandomForestClassifier(n_estimators=120, random_state=np.random.seed(11850))
            clf_unbiased.fit(X_train_unbiased_gdf, y_train_unbiased_cp)

            print("Saving trained model...\n")
            dump(clf_unbiased, './trained_model_{}.model'.format(int(time.time())))

        pred_train_unbiased = clf_unbiased.predict(X_train_unbiased_gdf)
        pred_test_unbiased = clf_unbiased.predict(X_test_unbiased_gdf)

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

        df_test_unbiased_bld_pred = df_test_unbiased_bld.copy()
        df_test_unbiased_bld_pred.labels = np.reshape(predictions2_unbiased, (-1, 1))
        metric_test_unbiased = compute_metrics(df_test_unbiased_bld, df_test_unbiased_bld_pred, unprivileged_groups, privileged_groups, disp=False)

        # 2024 changes-Append metrics to the results dictionary
        results_dict["#### Original training dataset"].append("")
        results_dict["Statistical parity difference (Original Train)"].append(train_metrics["Statistical parity difference"])
        results_dict["Disparate impact (Original Train)"].append(train_metrics["Disparate impact"])
        results_dict["Mean difference (Original Train)"].append(train_metrics["Mean difference"])
        results_dict["#### Reweighted training dataset"].append("")
        results_dict["Statistical parity difference (Reweighted Train)"].append(train_metrics["Statistical parity difference (Reweighted)"])
        results_dict["Disparate impact (Reweighted Train)"].append(train_metrics["Disparate impact (Reweighted)"])
        results_dict["Mean difference (Reweighted Train)"].append(train_metrics["Mean difference (Reweighted)"])
        results_dict["#### Original testing dataset"].append("")
        results_dict["Statistical parity difference (Original Test)"].append(test_metrics["Statistical parity difference"])
        results_dict["Disparate impact (Original Test)"].append(test_metrics["Disparate impact"])
        results_dict["Mean difference (Original Test)"].append(test_metrics["Mean difference"])
        results_dict["#### Reweighted testing dataset"].append("")
        results_dict["Statistical parity difference (Reweighted Test)"].append(test_metrics["Statistical parity difference (Reweighted)"])
        results_dict["Disparate impact (Reweighted Test)"].append(test_metrics["Disparate impact (Reweighted)"])
        results_dict["Mean difference (Reweighted Test)"].append(test_metrics["Mean difference (Reweighted)"])
        results_dict["The Original Training Accuracy Score"].append(accuracy_score(y_train_orig, predictions1))
        results_dict["The Original Testing Accuracy Score"].append(accuracy_score(y_test_orig, predictions2))
        results_dict["The Original Training F1 Score"].append(f1_score(y_train_orig, predictions1))
        results_dict["The Original Testing F1 Score"].append(f1_score(y_test_orig, predictions2))
        results_dict["The Original Training Recall Score"].append(recall_score(y_train_orig, predictions1))
        results_dict["The Original Testing Recall Score"].append(recall_score(y_test_orig, predictions2))
        results_dict["The Original Training Precision Score"].append(precision_score(y_train_orig, predictions1))
        results_dict["The Original Testing Precision Score"].append(precision_score(y_test_orig, predictions2))
        results_dict["#### Original Training AIF360 Fairness Metrics:"].append("")
        results_dict["Average odds difference (Original Train)"].append(metric_train_orig["Average odds difference"])
        results_dict["Equal opportunity difference (Original Train)"].append(metric_train_orig["Equal opportunity difference"])
        results_dict["Theil index (Original Train)"].append(metric_train_orig["Theil index"])
        results_dict["#### Original Testing AIF360 Fairness Metrics:"].append("")
        results_dict["Average odds difference (Original Test)"].append(metric_test_orig["Average odds difference"])
        results_dict["Equal opportunity difference (Original Test)"].append(metric_test_orig["Equal opportunity difference"])
        results_dict["Theil index (Original Test)"].append(metric_test_orig["Theil index"])
        results_dict["The Unbiased Training Accuracy Score"].append(accuracy_score(y_train_unbiased, predictions1_unbiased))
        results_dict["The Unbiased Testing Accuracy Score"].append(accuracy_score(y_test_unbiased, predictions2_unbiased))
        results_dict["The Unbiased Training F1 Score"].append(f1_score(y_train_unbiased, predictions1_unbiased))
        results_dict["The Unbiased Testing F1 Score"].append(f1_score(y_test_unbiased, predictions2_unbiased))
        results_dict["The Unbiased Training Recall Score"].append(recall_score(y_train_unbiased, predictions1_unbiased))
        results_dict["The Unbiased Testing Recall Score"].append(recall_score(y_test_unbiased, predictions2_unbiased))
        results_dict["The Unbiased Training Precision Score"].append(precision_score(y_train_unbiased, predictions1_unbiased))
        results_dict["The Unbiased Testing Precision Score"].append(precision_score(y_test_unbiased, predictions2_unbiased))
        results_dict["#### Unbiased Training AIF360 Fairness Metrics:"].append("")
        results_dict["Average odds difference (Unbiased Train)"].append(metric_train_unbiased["Average odds difference"])
        results_dict["Equal opportunity difference (Unbiased Train)"].append(metric_train_unbiased["Equal opportunity difference"])
        results_dict["Theil index (Unbiased Train)"].append(metric_train_unbiased["Theil index"])
        results_dict["#### Unbiased Testing AIF360 Fairness Metrics:"].append("")
        results_dict["Average odds difference (Unbiased Test)"].append(metric_test_unbiased["Average odds difference"])
        results_dict["Equal opportunity difference (Unbiased Test)"].append(metric_test_unbiased["Equal opportunity difference"])
        results_dict["Theil index (Unbiased Test)"].append(metric_test_unbiased["Theil index"])
        # Append T and Energy to results_dict
        results_dict["Execution_Time_unbiased (Hours)"] = total_execution_time_unbiased
        results_dict["Execution_Time_orig (Hours)"] = total_execution_time_orig
        results_dict["Energy_Unbiased (kWh)"] = total_energy_unbiased
        results_dict["Energy_Orig (kWh)"] = total_energy_orig


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
