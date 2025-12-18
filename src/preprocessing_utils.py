import cudf
from cuml.preprocessing import MinMaxScaler
import nvtx
import argparse
import traceback

@nvtx.annotate("read_hmda_data", color="blue")
def read_hmda_data(infile):
    # 2024 changes-adding datatype to respective columns
    gdf = cudf.read_csv(infile, usecols=[
        'activity_year', 'debt_to_income_ratio', 'derived_ethnicity', 'derived_race', 'derived_sex',
        'preapproval', 'loan_type', 'loan_purpose', 'lien_status', 'reverse_mortgage', 'open-end_line_of_credit', 
        'business_or_commercial_purpose', 'loan_amount', 'loan_to_value_ratio','loan_term', 'negative_amortization', 
        'interest_only_payment', 'balloon_payment', 'other_nonamortizing_features', 'property_value', 'construction_method', 
        'occupancy_type', 'manufactured_home_secured_property_type', 'total_units', 'income', 'applicant_ethnicity-1', 
        'co-applicant_ethnicity-1', 'applicant_ethnicity_observed', 'co-applicant_ethnicity_observed', 'applicant_race-1', 
        'co-applicant_race-1', 'applicant_race_observed', 'co-applicant_race_observed', 'applicant_sex', 'co-applicant_sex', 
        'applicant_sex_observed', 'co-applicant_sex_observed', 'applicant_age', 'co-applicant_age', 'applicant_age_above_62', 
        'co-applicant_age_above_62', 'submission_of_application', 'credit_score', 'action_taken', 'aus-1', 'aus-2', 'aus-3', 
        'aus-4', 'aus-5', 'denial_reason-1', 'denial_reason-2', 'denial_reason-3', 'denial_reason-4', 'tract_minority_population_percent'
    ], dtype={
        'activity_year':'int32', 'debt_to_income_ratio':'float32', 'derived_ethnicity':'str', 'derived_race':'str', 'derived_sex':'str',
        'preapproval':'str', 'loan_type':'str', 'loan_purpose':'str', 'lien_status':'str', 'reverse_mortgage':'str', 'open-end_line_of_credit':'str',
        'business_or_commercial_purpose':'str', 'loan_amount':'float32', 'loan_to_value_ratio':'float32', 'loan_term':'float32', 'negative_amortization':'str',
        'interest_only_payment':'str', 'balloon_payment':'str', 'other_nonamortizing_features':'str', 'property_value':'float32', 'construction_method':'str',
        'occupancy_type':'str', 'manufactured_home_secured_property_type':'str', 'total_units':'int32', 'income':'float32',
        'applicant_ethnicity-1':'str', 'co-applicant_ethnicity-1':'str', 'applicant_ethnicity_observed':'str', 'co-applicant_ethnicity_observed':'str',
        'applicant_race-1':'str', 'co-applicant_race-1':'str', 'applicant_race_observed':'str', 'co-applicant_race_observed':'str',
        'applicant_sex':'str', 'co-applicant_sex':'str', 'applicant_sex_observed':'str', 'co-applicant_sex_observed':'str',
        'applicant_age':'str', 'co-applicant_age':'str', 'applicant_age_above_62':'str', 'co-applicant_age_above_62':'str',
        'submission_of_application':'str', 'credit_score':'float32', 'action_taken':'int32',
        'aus-1':'str', 'aus-2':'str', 'aus-3':'str', 'aus-4':'str', 'aus-5':'str',
        'denial_reason-1':'str', 'denial_reason-2':'str', 'denial_reason-3':'str', 'denial_reason-4':'str',
        'tract_minority_population_percent':'float64'
    })

    # 2024 changes-Convert appropriate columns to categorical
    categorical_columns = ['activity_year', 'derived_ethnicity', 'derived_race', 'derived_sex', 'preapproval', 'loan_type',
                           'loan_purpose', 'lien_status', 'reverse_mortgage', 'open-end_line_of_credit', 'business_or_commercial_purpose',
                           'negative_amortization', 'interest_only_payment', 'balloon_payment', 'other_nonamortizing_features',
                           'construction_method', 'occupancy_type', 'submission_of_application', 'manufactured_home_secured_property_type',
                           'applicant_ethnicity-1', 'co-applicant_ethnicity-1', 'applicant_ethnicity_observed', 'co-applicant_ethnicity_observed',
                           'applicant_race-1', 'co-applicant_race-1', 'applicant_race_observed', 'co-applicant_race_observed',
                           'applicant_sex', 'co-applicant_sex', 'applicant_sex_observed', 'co-applicant_sex_observed', 'applicant_age',
                           'co-applicant_age', 'applicant_age_above_62', 'co-applicant_age_above_62',
                           'aus-1', 'aus-2', 'aus-3', 'aus-4', 'aus-5',
                           'denial_reason-1', 'denial_reason-2', 'denial_reason-3', 'denial_reason-4']

    for col in categorical_columns:
        gdf[col] = gdf[col].astype('category')

    return gdf

# 2024 changes-generating year wise columns(splitting the activity_year into different years)
@nvtx.annotate("generate_year_columns", color="purple")
def generate_year_columns(gdf):
    unique_years = gdf['activity_year'].unique().to_arrow().to_pylist()
    for year in unique_years:
        gdf[f'activity_year_{year}'] = (gdf['activity_year'] == year).astype('int32')
    return gdf
    # Create new columns for each year
    # gdf['activity_year_2018'] = (gdf['activity_year'] == 2018).astype('int32')
    # gdf['activity_year_2019'] = (gdf['activity_year'] == 2019).astype('int32')
    # gdf['activity_year_2020'] = (gdf['activity_year'] == 2020).astype('int32')
    # gdf['activity_year_2021'] = (gdf['activity_year'] == 2021).astype('int32')
    

# 2024 changes-dropping missing values
@nvtx.annotate("drop_missing_values", color="purple")
def drop_missing_values(gdf, predictors):
    # Drop instances with missing values for the specified protected attributes
    return gdf.dropna(subset=predictors)

# 2024 changes-generating descriptives with respect to year
@nvtx.annotate("generate_descriptives_by_year", color="orange")
def generate_descriptives_by_year(gdf, attributes, outfile_prefix):
    years = [2018, 2019, 2020, 2021]
    descriptives = cudf.DataFrame()

    all_descriptives = []
    for year in years:
        year_df = gdf[gdf['activity_year'] == year]
        year_desc = year_df[attributes].describe().T

        # Rename columns based on actual output of describe()
        column_mapping = {
            'count': f'activity_year_{year}_count',
            'mean': f'activity_year_{year}_mean',
            'std': f'activity_year_{year}_std',
            'min': f'activity_year_{year}_min',
            '25%': f'activity_year_{year}_25%',
            '50%': f'activity_year_{year}_50%',
            '75%': f'activity_year_{year}_75%',
            'max': f'activity_year_{year}_max',
        }
        year_desc = year_desc.rename(columns=column_mapping)
        all_descriptives.append(year_desc)
        
    descriptives = cudf.concat(all_descriptives, axis=1)

    overall_desc = gdf[attributes].describe().T
    overall_desc.columns = ['All_years_count', 'All_years_mean', 'All_years_std', 'All_years_min',
                            'All_years_25%', 'All_years_50%', 'All_years_75%', 'All_years_max']

    final_descriptives = cudf.concat([descriptives, overall_desc], axis=1)
    final_descriptives.to_csv(f"{outfile_prefix}_descriptives.csv")

# 2024 changes-adding tract_minority columns based on different thresholds
@nvtx.annotate("read_process_save", color="green")
def read_process_save(infile, outfile, descriptives_prefix):
    try: 
        preprocessed_df = read_hmda_data(infile)
        print(f"Data read successfully: {preprocessed_df.shape}")
    except Exception as e:
        print(f"Error reading data: {e}")
        traceback.print_exc()
        return

    # print(preprocessed_df.shape)

    # Generate year-based columns
    preprocessed_df = generate_year_columns(preprocessed_df)

    # Drop instances with missing values for the predictors
    predictors = ['loan_amount', 'loan_to_value_ratio', 'property_value', 'income', 'debt_to_income_ratio', 'loan_term']
    preprocessed_df = drop_missing_values(preprocessed_df, predictors)
    print(f"Shape after dropping missing values: {preprocessed_df.shape}")

    # Generate descriptives for the specified attributes by year
    generate_descriptives_by_year(preprocessed_df, predictors, descriptives_prefix)

    valid_actions = [1, 2, 3]
    preprocessed_df = preprocessed_df[preprocessed_df['action_taken'].isin(valid_actions)]

    replacement_values = [1,1,0]
    preprocessed_df['approved'] = preprocessed_df['action_taken'].replace(valid_actions, replacement_values)
    print(preprocessed_df.shape)

    # Add new columns for 'tract_minority' based on different thresholds
    preprocessed_df['tract_minority_percentage40'] = (preprocessed_df['tract_minority_population_percent'] >= 40).astype('int32')
    preprocessed_df['tract_minority_percentage50'] = (preprocessed_df['tract_minority_population_percent'] >= 50).astype('int32')
    preprocessed_df['tract_minority_percentage60'] = (preprocessed_df['tract_minority_population_percent'] >= 60).astype('int32')
    preprocessed_df['tract_minority_percentage70'] = (preprocessed_df['tract_minority_population_percent'] >= 70).astype('int32')
    preprocessed_df['tract_minority_percentage80'] = (preprocessed_df['tract_minority_population_percent'] >= 80).astype('int32')
    preprocessed_df['tract_minority_percentage90'] = (preprocessed_df['tract_minority_population_percent'] >= 90).astype('int32')

    # Scale values between (0,1)
    scaler = MinMaxScaler()

    columns_to_scale = ['loan_amount','debt_to_income_ratio', 'loan_to_value_ratio',  'loan_term',
                        'property_value', 'total_units', 'income', 'credit_score', 'tract_minority_population_percent']
    preprocessed_df[columns_to_scale] = scaler.fit_transform(preprocessed_df[columns_to_scale])

    # 2024 changes-adding more columns for column_to_encode
    # One-hot encode categorical data
    columns_to_encode = ['derived_ethnicity', 'derived_race', 'derived_sex',
                         'preapproval', 'loan_type', 'loan_purpose', 'lien_status',
                         'reverse_mortgage', 'open-end_line_of_credit', 'business_or_commercial_purpose',
                         'negative_amortization', 'interest_only_payment',
                         'balloon_payment', 'other_nonamortizing_features', 'construction_method', 'occupancy_type',
                         'submission_of_application',
                         'manufactured_home_secured_property_type',
                         'applicant_ethnicity-1', 'co-applicant_ethnicity-1', 'applicant_ethnicity_observed',
                         'co-applicant_ethnicity_observed', 'applicant_race-1', 'co-applicant_race-1', 'applicant_race_observed',
                         'co-applicant_race_observed', 'applicant_sex', 'co-applicant_sex', 'applicant_sex_observed',
                         'co-applicant_sex_observed', 'applicant_age', 'co-applicant_age', 'applicant_age_above_62',
                         'co-applicant_age_above_62',
                         'aus-1', 'aus-2', 'aus-3', 'aus-4', 'aus-5',
                         'denial_reason-1', 'denial_reason-2', 'denial_reason-3', 'denial_reason-4']

    transformed_df = cudf.get_dummies(preprocessed_df[columns_to_encode], columns=columns_to_encode)
    preprocessed_df = preprocessed_df.drop(columns_to_encode, axis=1)
    preprocessed_df[transformed_df.columns] = transformed_df
    preprocessed_df = preprocessed_df.fillna(0)

    preprocessed_df.reset_index(drop=True, inplace=True)
    preprocessed_df.to_csv(outfile, chunksize=1000)

# 2024 changes-adding argument for the descriptives to be generated
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocesses the merged HDMA-FannieFreddie dataset.')
    parser.add_argument('infile', type=argparse.FileType('r'), help='The merged data file')
    parser.add_argument('outfile', type=argparse.FileType('w'), help='The pre-processed output file')
    parser.add_argument('descriptives_prefix', type=str, help='The prefix for the descriptives output file')

    args = parser.parse_args()
    read_process_save(args.infile.name, args.outfile.name, args.descriptives_prefix)