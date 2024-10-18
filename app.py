import joblib
import uvicorn

import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from sklearn.pipeline import Pipeline
from data_models import PredictionDataset

app = FastAPI()

current_path = Path(__file__).parent

model_path = current_path / 'models' / 'models' / 'xgb_model.joblib'
preprocessor_path = current_path/ 'models' / 'transformers' / 'preprocessor.joblib'
output_transformer_path = current_path/'models'/'transformers'/'output_transformer.joblib'

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)
output_transformer = joblib.load(output_transformer_path)

model_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor_model', model)
])

@app.get('/')
def home():
    return "WelCome to taxi price prediction app"


@app.post('/predictions')
def do_predictions(test_data:PredictionDataset):
    X_test = pd.DataFrame(
        data={
            'vendor_id': test_data.vendor_id,
            'pickup_latitude': test_data.pickup_latitude,
            'pickup_longitude': test_data.pickup_longitude,
            'dropoff_latitude': test_data.dropoff_latitude,
            'dropoff_longitude': test_data.dropoff_longitude,
            'haversine_distance' : test_data.haversine_distance,
            'euclidean_distance' : test_data.euclidean_distance,
            'manhattan_distance': test_data.manhattan_distance,
            'passenger_count' : test_data.passenger_count,
            'pickup_hour': test_data.pickup_hour,
            'pickup_date':test_data.pickup_date,
            'pickup_month': test_data.pickup_month,
            'pickup_day': test_data.pickup_day,
            'is_weekend' : test_data.is_weekend
        }, index=[0]
    )
    prediction = model_pipe.predict(X_test).reshape(-1, 1)
    output_inverse_transformed = output_transformer.inverse_transform(prediction)[0].item()

    return f"Trip duration for the trip is {output_inverse_transformed:.2f} minutes"



if __name__ == '__main__':
    uvicorn.run(app="app:app", host="0.0.0.0", port=8000)
    