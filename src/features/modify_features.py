import logging 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))
from logger import create_log_path , CustomLogger

target_column = 'trip_duration'
plot_path = Path('reports/figures/target_distribution.png')

log_file_path = create_log_path('modify_features')
modify_logger = CustomLogger(logger_name='modify_features', log_filename=log_file_path)

modify_logger.set_log_level(level=logging.INFO)


# target column is converted into minutes
def convert_target_to_minutes(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    dataframe[target_column] = dataframe[target_column]/60
    modify_logger.save_logs(msg='Target columns converted into minutes from seconds')
    return dataframe


# drop all the column where trip_duration is greater than 200 minutes
def drop_above_two_hundred_minutes(dataframe:pd.DataFrame, target_column:str)->pd.DataFrame:
    new_dataframe = dataframe[dataframe[target_column]<=200].copy()
    # max value of target column
    max_val = new_dataframe[target_column].max()
    modify_logger.save_logs(msg='The maximum value in target column after transformation is {max_value} and the state of transformation is {max_value<=200}')
    print(max_val)
    if max_val<=200:
        return new_dataframe
    else:
        raise ValueError('Oulier target value are not removed from the data')


# plot the distribution of data through non parametric test
def plot_target(dataframe:pd.DataFrame, target_col: str, save_path:str):
    sns.kdeplot(data=dataframe, x=target_col)
    plt.title(f'Distribution of {target_column}')

    # save the plot at the destination path
    plt.savefig(save_path)
    modify_logger.save_logs(msg='The distribution of target column is saved at destination')

# drop columns
def drop_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    modify_logger.save_logs(f'Columns in data before removal are {list(dataframe.columns)}')
    # drop columns from train and val data
    if 'dropoff_datetime' in dataframe.columns:
        columns_to_drop = ['id','dropoff_datetime','store_and_fwd_flag']
        # dropping the columns from dataframe
        dataframe_after_removal = dataframe.drop(columns=columns_to_drop)
        list_of_columns_after_removal = list(dataframe_after_removal.columns)
        modify_logger.save_logs(f'Columns in data after removal are {list_of_columns_after_removal}')
        # verifying if columns dropped
        modify_logger.save_logs(msg=f"Columns {', '.join(columns_to_drop)} dropped from data  verify={columns_to_drop not in list_of_columns_after_removal}")
        return dataframe_after_removal
    # drop columns from the test data
    else:
        columns_to_drop = ['id','store_and_fwd_flag']
        # dropping the columns from dataframe
        dataframe_after_removal = dataframe.drop(columns=columns_to_drop)
        list_of_columns_after_removal = list(dataframe_after_removal.columns)
        # verifying if columns dropped
        modify_logger.save_logs(msg=f"Columns {', '.join(columns_to_drop)} dropped from data  verify={columns_to_drop not in list_of_columns_after_removal}")
        return dataframe_after_removal
    
# making new features from datetime_columns
def make_datetime_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    # copy the original dataframe
    new_df = dataframe.copy()
    # number of rows and columns before and after transformation
    org_num_of_rows, org_num_of_cols = new_df.shape

    # convert the column to datetime
    new_df['pickup_datetime'] = pd.to_datetime(new_df['pickup_datetime'])
    modify_logger.save_logs(msg=f"pickup_datetime column converted to datetime {new_df['pickup_datetime'].dtype}")

    # new features 
    new_df['pickup_hour'] = new_df['pickup_datetime'].dt.hour
    new_df['pickup_date'] = new_df['pickup_datetime'].dt.day
    new_df['pickup_month']= new_df['pickup_datetime'].dt.month
    new_df['pickup_day']  = new_df['pickup_datetime'].dt.weekday
    new_df['is_weekend']  = new_df.apply(lambda row: row['pickup_day']>=5, axis=1).astype('int')

    # drop the redundant date time column
    new_df = new_df.drop(columns='pickup_datetime')
    modify_logger.save_logs(msg = f"pickup_datetime columns dropped verify={'pickup_datetime' not in new_df.columns}")

    # number of rows and cols after transformation
    num_of_rows_after_transformation, num_of_cols_after_transformation = new_df.shape
    modify_logger.save_logs(msg=f'The number of columns increased by 4 {num_of_cols_after_transformation==org_num_of_cols+5-1}')
    modify_logger.save_logs(msg=f'Number of rows remained the same verify={org_num_of_rows==num_of_rows_after_transformation}')
    return new_df

# remove redundant passengers from passenger columns
def remove_passengers(dataframe: pd.DataFrame) -> pd.DataFrame:
    # list of passenger that required
    passenger_to_include = list(range(1, 7))
    # new_filter_df = dataframe['passenger_count'].isin(passenger_to_include)
    new_df = dataframe[((dataframe['passenger_count']>=1) & (dataframe['passenger_count']<=7))]

    
    # list of unique passenger values in the passenger_count_columns
    unique_passenger_count = list(np.sort(new_df['passenger_count'].unique()))
    modify_logger.save_logs(msg=f'The unique passenger list is {unique_passenger_count} verify={passenger_to_include==unique_passenger_count}')
    return new_df

def input_modifications(dataframe: pd.DataFrame)-> pd.DataFrame:
    
    # drop the un-neccessary columns
    new_df = drop_columns(dataframe)

    # remove the rows having excluded passenger values
    modified_passenger_df = remove_passengers(new_df)

    # add datetime features to data
    include_new_datetime_ftrs = make_datetime_features(modified_passenger_df)
    modify_logger.save_logs('Modifications with input feature complete')
    return include_new_datetime_ftrs

def target_modification(dataframe: pd.DataFrame, target_column:str=target_column) -> pd.DataFrame:

    # convert the target column from seconds to minutes
    minutes_df = convert_target_to_minutes(dataframe, target_column)

    # remove the target value greater than 200 minutes
    df_after_target_outliers_removal = drop_above_two_hundred_minutes(minutes_df, target_column)

    # plot the target column distribution
    plot_target(df_after_target_outliers_removal, target_column, save_path=root_path / plot_path)
    modify_logger.save_logs('Modifications with the target columns is complete')
    return df_after_target_outliers_removal

# read the dataframe from the location
def read_data(data_path):
    return pd.read_csv(data_path)

# save the dataframe to location
def save_data(dataframe: pd.DataFrame, save_path: Path):
    dataframe.to_csv(save_path, index=False)


def main(data_path, filename):
    # read the data into frame
    df = read_data(data_path)

    # modification on the input data
    df_input_modifications = input_modifications(df)

    # check whether the input files has target columns 
    if filename == 'train.csv' or filename == 'val.csv':
        df_final = target_modification(dataframe=df_input_modifications)
    else:
        df_final = df_input_modifications
    return df_final


if __name__ == '__main__':
    for ind in range(1,4):
        # read the input file name from command
        input_file_path = sys.argv[ind]
        # current file path
        current_path = Path(__file__)
        # root directory path
        root_path = current_path.parent.parent.parent
        # input data path
        data_path = root_path / input_file_path
        # get the file name
        filename = data_path.parts[-1]
        # call the main function
        df_final = main(data_path=data_path,filename=filename)
        # save the dataframe
        output_path = root_path / "data/processed/transformations"
        # make the directory if not available
        output_path.mkdir(parents=True,exist_ok=True)
        # save the data
        save_data(df_final,output_path / filename)
        modify_logger.save_logs(msg=f'{filename} saved at the destination folder')