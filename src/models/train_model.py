import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


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


def train_model_grid_search(model, X_train, y_train, param_grid):
    # initialize grid search cv
    grid_search = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        cv = 5, 
        verbose=2, 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_

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
    rf_params = params['train_model']['random_forest_regressor']
    param_grid = rf_params['param_grid']

    # extract indivisual rf_params
    # n_estimators = rf_params['n_estimator']
    # max_depth = rf_params['max_depth']
    # verbose = rf_params['verbose']
    # n_jobs = rf_params['n_jobs']

    # make the model object
    # base_regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, verbose=verbose, n_jobs=n_jobs)
    # base_regressor = train_model(model=base_regressor, X_train=X_train, y_train=y_train)
    tuned_regressor = RandomForestRegressor()
    tuned_regressor, best_params = train_model_grid_search(model=tuned_regressor, X_train=X_train, y_train=y_train, param_grid=param_grid)


    # save the model after traning
    model_output_path = root_path / 'models' / 'models'
    model_output_path.mkdir(exist_ok=True)
    save_model(model=tuned_regressor, save_path = model_output_path / 'tuned_model.joblib')

if __name__ == '__main__':
    main()