# Overview

This repository contains two Python files, preprocessing_utils.py and fairplay.py, which work together to preprocess, analyze, and evaluate fairness in mortgage data. These scripts are designed to handle large datasets efficiently using GPU-accelerated libraries, providing preprocessing, scaling, and fairness evaluation capabilities for data scientists working with mortgage-related datasets.

# Files Description

**1. preprocessing_utils.py**
This file is responsible for the initial data loading and transformation. It performs several key preprocessing steps on the dataset to prepare it for analysis:

* Reading and Parsing Data: Reads the dataset, specifying relevant columns and their data types for efficient memory usage and performance.
*  Data Transformation: Converts specific columns to categorical data types to optimize processing and storage.
* Feature Engineering: Generates new columns representing the activity year (e.g., activity_year_2018) and computes other features such as minority population percentages.
* Handling Missing Values: Removes rows with missing values for key predictors to maintain dataset integrity.
* Data Scaling: Applies a Min-Max scaler to normalize numeric columns for further analysis.
* Descriptive Statistics Generation: Computes and saves descriptive statistics for different years and overall data.
* Data Encoding: One-hot encodes categorical variables to make them suitable for model training.

**2. fairplay.py**
This script builds upon the preprocessed data to assess and mitigate fairness-related issues in the dataset. It includes modules for model training and fairness evaluation using the AIF360 library:

* Fairness Metrics Computation: Computes various fairness metrics such as statistical parity, disparate impact, and mean difference, highlighting biases in the dataset.
*  Reweighing Algorithm: Utilizes a bias mitigation technique called "Reweighing" to adjust the dataset, ensuring fairer outcomes for protected groups.
* Model Training: Trains a RandomForestClassifier on the data to predict loan approvals, and evaluates performance using accuracy, F1, recall, and precision scores.
* Unbiased Approval Computation: Creates a dataset that includes unbiased loan approvals based on protected attributes.
* Results Logging: Saves performance and fairness metrics for different conditions and groups, providing insights into the model's behavior and bias levels.

**3. fairplay-NN.py**
This script builds upon the preprocessed data to assess and mitigate fairness-related issues in the dataset using advanced neural network models. It includes modules for model training, fairness evaluation, and bias mitigation leveraging the AIF360 library and PyTorch:

* Fairness Metrics Computation: Computes various fairness metrics using the AIF360 library, such as Statistical Parity Difference, Disparate Impact, Equal Opportunity Difference, Theil Index. Highlights biases in the dataset across protected attributes.
* Reweighing Algorithm: Implements the "Reweighing" bias mitigation technique to adjust dataset weights for fairer outcomes. Ensures balanced treatment of privileged and unprivileged groups.
* Neural Network Model Training: 
     * Loan Approval Neural Network (Binary Classification)-> Predicts loan approvals with features such as income, loan amount, and debt-to-income ratio. Trains using PyTorch with BCE loss and Adam optimizer.
     * Denial Reason Neural Network (Multi-Class Classification)-> Classifies denial reasons for rejected loans (e.g., debt-to-income ratio, collateral issues). Utilizes CrossEntropyLoss and outputs multi-class 
       probabilities.
* Unbiased Approval Computation: Generates an unbiased dataset by applying fairness adjustments. Updates denial reasons (denial_reason_final) for improved interpretability.
* Results Logging: Saves performance and fairness metrics for different conditions and groups, providing insights into the model's behavior and bias levels.
* Energy and Efficiency Metrics: Calculates energy consumption and execution times for model training and fairness evaluation. Tracks efficiency improvements for reweighted datasets.
* Denial Reason Analysis: Provides a count of denial reasons in the dataset for deeper insights into loan application rejections. Exports denial reason statistics to CSV for transparency.

# Dependencies

Both scripts require the following libraries:

cudf: GPU-accelerated data frame library

cuml: GPU-accelerated machine learning library

aif360: Fairness library for bias detection and mitigation

scikit-learn: Standard machine learning library

NVTX (NVIDIA Tools Extension): Annotation library for performance profiling

cupy: NumPy-like library for GPU arrays

PyTorch: Used for building and training neural networks. Provides modules like torch.nn, torch.optim, and utilities for data handling (DataLoader, TensorDataset).

sklearn.preprocessing.StandardScaler: Provides scaling and normalization of input data to improve neural network training.

torch.utils.data: Provides utilities like DataLoader and TensorDataset for creating data pipelines for training and evaluation.

Ensure that these libraries are properly installed and configured for GPU acceleration before running the scripts.

# Usage

**preprocessing_utils.py**

Run this script to preprocess and transform the input dataset:
```bash
python preprocessing_utils.py HDMA_predCredScore_XGBmsa_noHDMA-FFmatches.csv preprocess_full.csv any_file_name
```

* <input_file>: The input CSV file containing the raw data.
* <output_file>: The output file where the transformed data will be saved.
* <descriptive_prefix>: Prefix for naming the descriptive statistics file.

**fairplay.py**

Run this script to train the model, evaluate fairness, and compute results:
```bash
python fairplay.py preprocess_full.csv [model_file] -v
```

* <preprocessed_file>: The preprocessed data file generated by preprocessing_utils.py.
* \[model_file\]: (Optional) A trained model file to use for evaluation.
* -v: Verbose flag to display detailed outputs.


**fairplay-NN.py**

Run this script to train the model, evaluate fairness, and compute results:
```bash
python fairplay-NN.py preprocess_full.csv [model_file] -v
```

* <preprocessed_file>: The preprocessed data file generated by preprocessing_utils.py.
* \[model_file\]: (Optional) A trained model file to use for evaluation.
* -v: Verbose flag to display detailed outputs.
