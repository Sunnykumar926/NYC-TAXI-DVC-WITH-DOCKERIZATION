import sys
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score

target = 'trip_duration'
model_name = 'xgb_model.joblib'

def load_datafram(path):
    return pd.read_csv(path)

def make_X_y(dataframe:pd.DataFrame, target_column:str):
    df_copy = dataframe

    x = df_copy.drop(columns=target)
    y = df_copy[target]

    return x, y


def get_predictions(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def calculate_r2_score(y_actual, y_predict):
    score = r2_score(y_actual, y_predict)
    return score


def main():
    # current file path
    current_path = Path(__file__)
    # root direcotry path
    root_path =current_path.parent.parent.parent

    for ind in range(1,3):
        # read the input file path
        data_path = root_path / 'data/processed/final' / sys.argv[ind]

        # load the dataframe
        data = load_datafram(data_path)

        X_test, y_test = make_X_y(data, target)

        # model path
        model_path = root_path / 'models' / 'models' / model_name

        # load the model
        model = joblib.load(model_path)

        # get the prediction from model
        y_pred = get_predictions(model = model, X_test=X_test)

        # calculate r2 score
        score = calculate_r2_score(y_actual = y_test, y_predict=y_pred)

        print(f'\nThe score for dataset {sys.argv[ind]} is : {score}')

if __name__ == '__main__':
    main()
    