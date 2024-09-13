import sys
import joblib

import numpy as np
import pandas as pd

from yaml import safe_load
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, PowerTransformer
from outlier_removal import OutliersRemover

column_names = ['pickup_latitude','pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
target = 'trip_duration'

def save_transformer(path, object):
    joblib.dump(value=object,
                filename=path)
    
def remove_outliers(dataframe: pd.DataFrame, percentiles: list, column_name:list)->pd.DataFrame:
    df = dataframe.copy()

    outlier_transformer = OutliersRemover(percentiles, column_name)

    # fit on the data 
    outlier_transformer.fit(dataframe)

    return outlier_transformer


def train_preprocessor(data: pd.DataFrame):
    ohe_columns = ['vendor_id']
    standard_scaler = ['haversine_distance', 'euclidean_distance', 'manhattan_distance']
    min_max_Scaler  = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']

    preprocessor = ColumnTransformer(transformers=[
        ('one-hot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), ohe_columns),
        ('min-max', MinMaxScaler(), min_max_Scaler),
        ('standard_scaler', StandardScaler(), standard_scaler)
    ],remainder='passthrough', verbose_feature_names_out=False, n_jobs=1)

    # set the output as df
    preprocessor.set_output(transform='pandas')

    # fit the preprocessor on the training data
    preprocessor.fit(data)
    return preprocessor

def transform_data(transformer, data:pd.DataFrame)->pd.DataFrame:

    # transform the data
    data_transformed = transformer.transform(data)
    return data_transformed

def read_dataframe(path):
    return pd.read_csv(path)


def save_dataframe(dataframe:pd.DataFrame, save_path):
    return dataframe.to_csv(save_path, index=False)

def transform_output(target: pd.Series):
    power_transform = PowerTransformer(method='yeo-johnson', standardize=True)

    # fit and transform the target
    transformed_target = power_transform.fit(target.values.reshape(-1, 1))
    return transformed_target


def main():
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    input_path = root_path/ 'data' / 'processed' / 'build_features'

    # read the parameters file 
    with open('params.yaml') as f:
        params = safe_load(f)

    # percentiles values
    percentiles = list(params['data_preprocessing']['percentiles'])

    # save transformer path and make directory
    save_transformers_path = root_path / 'models' / 'transformers'
    save_transformers_path.mkdir(exist_ok=True)

    # save output file and make directory
    save_data_path = root_path / 'data' / 'processed' / 'final'
    save_data_path.mkdir(exist_ok=True)

    for filename in sys.argv[1:]:
        complete_input_path = input_path/filename
        if filename=='train.csv':
            df = read_dataframe(complete_input_path)

            # make x and y
            x = df.drop(columns=target)
            y = df[target]

            # remove outlier from data 
            outlier_transformer = remove_outliers(df, percentiles, column_name=column_names)

            # save the transformer
            save_transformer(path=save_transformers_path / 'outliers.joblib', object=outlier_transformer)

            # dataframe without outlier
            df_without_outliers = transform_data(outlier_transformer, data=x)

            # train the preprocessor on the data and save the preprocessor
            preprocessor = train_preprocessor(df_without_outliers)
            save_transformer(path=save_transformers_path / 'preprocessor.joblib', object=preprocessor)

            # transform the data
            X_trans = transform_data(preprocessor, data=x)

            print('The shape of target column is : ', len(target))
            output_transformer = transform_output(y)
            print('The shape of the target column is: ', len(target))

            # transform the target 
            y_trans = transform_data(transformer=output_transformer, data=y.values.reshape(-1,1))

            # save the transformed data in X_trans
            X_trans['trip_duration'] = y_trans

            # save the output transformer
            save_transformer(path = save_transformers_path/'output_transformer.joblib', object=output_transformer)

            # save the tranformed data 
            save_dataframe(X_trans, save_data_path/filename)

        elif filename == 'val.csv':
            df = read_dataframe(complete_input_path)

            x = df.drop(columns=target)
            y = df[target]

            # load the transformer
            outlier_transformer = joblib.load(save_transformers_path / 'outliers.joblib')
            df_without_outliers = transform_data(outlier_transformer, data=x)

            # laod the preprocessor
            preprocessor = joblib.load(save_transformers_path / 'preprocessor.joblib')
            X_trans = transform_data(preprocessor, data=x)

            # laod the output transformer
            output_transformer = joblib.load(save_transformers_path / 'output_transformer.joblib')
            y_trans = transform_data(output_transformer, data=y.values.reshape(-1, 1))

            X_trans['trip_duration'] = y_trans
            # save the transformed data
            save_dataframe(X_trans, save_data_path/filename)
        
        elif filename == 'test.csv':
            df = read_dataframe(complete_input_path)

            #  load the transformer 
            outlier_transformer = joblib.load(save_transformers_path / 'outliers.joblib')
            df_without_outliers = transform_data(outlier_transformer, data=df)

            # load the preprcessor
            preprocessor = joblib.load(save_transformers_path / 'preprocessor.joblib')
            X_trans = transform_data(preprocessor, data=df)

            # save the transform data
            save_dataframe(X_trans, save_data_path/filename)
            


if __name__ == '__main__':
    main()