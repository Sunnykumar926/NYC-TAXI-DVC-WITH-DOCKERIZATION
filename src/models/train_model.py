import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor


from yaml import safe_load

target  = 'trip_duration'

def load_dataframe(path):
    df = pd.read_csv(path)
    return df

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def make_X_y(dataframe: pd.DataFrame, target_column:str):
    df_copy = dataframe.copy()

    x = df_copy.drop(columns=target_column)
    y = df_copy[target]
    return x, y

def save_model(model, save_path):
    joblib.dump(value=model, filename=save_path)


def main():
    # current file path
    current_path = Path(__file__)

    # root directory path
    root_path = current_path.parent.parent.parent

    # read the input file path
    training_data_path = root_path / sys.argv[1]

    # load the data 
    train_data = load_dataframe(training_data_path)

    # make the X_train y_train of the data
    X_train , y_train = make_X_y(train_data, target_column=target)


    # read the params from params.yaml
    with open('params.yaml', 'r') as f:
        params = safe_load(f)
    
    # access the parameters of random forest model
    xgb_params = params['train_model']['xgboost_regressor']

    # extract indivisual rf_params
    n_estimators = xgb_params['n_estimators']
    max_depth = xgb_params['max_depth']
    learning_rate = xgb_params['learning_rate']
    colsample_bytree = xgb_params['colsample_bytree']
    subsample = xgb_params['subsample']
    min_child_weight = xgb_params['min_child_weight']
    lambda1 = xgb_params['lambda']
    alpha = xgb_params['alpha']

    # make the model object
    xgb_regressor = XGBRegressor(n_estimators=n_estimators, 
                                 max_depth=max_depth, 
                                 learning_rate=learning_rate, 
                                 colsample_bytree=colsample_bytree, 
                                 subsample= subsample,
                                 min_child_weight=min_child_weight,
                                 reg_lambda=lambda1,
                                 alpha = alpha
                                 )
    xgb_regressor = train_model(model=xgb_regressor, X_train=X_train, y_train=y_train)


    # save the model after traning
    model_output_path = root_path / 'models' / 'models'
    model_output_path.mkdir(exist_ok=True)
    save_model(model=xgb_regressor, save_path = model_output_path / 'xgb_model.joblib')

if __name__ == '__main__':
    main()