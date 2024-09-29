import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

target = 'trip_duration'
model_name = 'rf_model.joblib'

def load_dataframe(path):
    df = pd.read_csv(path)
    return df

def make_X_y(dataframe: pd.DataFrame, target_column:str):
    df_copy = dataframe
    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]
    return X, y

def get_predictions(model, X):
    y_pred = model.predict(X)
    return y_pred

def final_r2_score(y_actual, y_predicted):
    score = r2_score(y_actual, y_predicted)
    return score

def main():
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    data_path = root_path / 'data/processed/final/'

    for ind in range(1, 3):
        file_name = sys.argv[ind]

        data = load_dataframe(data_path/file_name)
        X, y = make_X_y(dataframe=data, target_column=target)

        model_path = root_path / 'models/models'/ model_name
        model = joblib.load(model_path)

        if file_name == 'train.csv':
            cross_val = cross_val_score(estimator=model,
                                        X=X, y=y,
                                        cv=5, scoring='r2',
                                        n_jobs=-1)
            x_axis_list = [f'fold_{axis}' for axis in range(1, 6)]
            y_axis_list = list(cross_val)
        else:
            # calculated the y_pred
            y_pred = get_predictions(model=model, X=X)
            r2_score = final_r2_score(y, y_pred)
            x_axis_list.append('val')
            y_axis_list.append(r2_score)

    results_path = Path(root_path / 'plots/model_results')
    results_path.mkdir(exist_ok=True)

    # plot the graph
    fig = plt.figure(figsize=(15, 10))
    plt.bar(x=x_axis_list, height=y_axis_list)
    plt.xlabel('K Folds')
    plt.ylabel('R2 Score')
    fig.savefig(results_path / 'results.png')

if __name__ == '__main__':
    main()

