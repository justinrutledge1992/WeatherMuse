# This file is used to preprocess the available hourly weather data across all cities

import os
import pandas as pd
import numpy as np

def min_max(df):
    min_val = df.min().min()  # Minimum value in the entire dataset
    max_val = df.max().max()  # Maximum value in the entire dataset
    return (df - min_val) / (max_val - min_val)

def z_score(df):
    mean_val = df.to_numpy().mean()  # Mean of the entire dataset
    std_val = df.to_numpy().std()   # Standard deviation of the entire dataset
    return (df - mean_val) / std_val

def normalize_and_write_to_file(data, value_name):
    data = data.iloc[:, 1:]
    minmax_temp = min_max(data)
    z_score_temp = min_max(data)
    minmax_temp.to_csv(f'{data_path}{value_name}_minmax.csv', index=False)
    z_score_temp.to_csv(f'{data_path}{value_name}_zscore.csv', index=False)

# Use pandas to read and clean up the weather data:
# Use pandas to read each csv file:
data_path = os.path.join(os.getcwd(), "weathermuse", "hourly_weather_data", "original_data")
city_attributes = pd.read_csv(f'{data_path}city_attributes.csv')
weather_description = pd.read_csv(f'{data_path}weather_description.csv')
temperature = pd.read_csv(f'{data_path}temperature.csv')
humidity = pd.read_csv(f'{data_path}humidity.csv')
pressure = pd.read_csv(f'{data_path}pressure.csv')
wind_direction = pd.read_csv(f'{data_path}wind_direction.csv')
wind_speed = pd.read_csv(f'{data_path}wind_speed.csv')

normalize_and_write_to_file(temperature, 'temperature')
normalize_and_write_to_file(humidity, 'humidity')
normalize_and_write_to_file(pressure, 'pressure')
normalize_and_write_to_file(wind_direction, 'wind_direction')
normalize_and_write_to_file(wind_speed, 'wind_speed')
