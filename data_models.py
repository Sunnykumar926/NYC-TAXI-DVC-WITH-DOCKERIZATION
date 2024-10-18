# 1. BaseModel is a core class in Pydantic used to define data models with type validation.
# 2. It ensures that the data you pass into it matches the expected types (like str, int, float, etc.).
# 3. If the data doesn’t conform to the expected types or structure, Pydantic will raise an error.

from pydantic import BaseModel

class PredictionDataset(BaseModel):
    vendor_id:int 
    pickup_latitude: float
    pickup_longitude: float
    dropoff_latitude: float
    dropoff_longitude: float
    haversine_distance:float
    euclidean_distance: float
    manhattan_distance: float
    passenger_count: int
    pickup_hour: int
    pickup_date: int
    pickup_month: int
    pickup_day: int
    is_weekend: int