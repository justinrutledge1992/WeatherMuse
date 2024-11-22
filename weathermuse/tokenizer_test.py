from audiocraft.models import MusicGen
import torch
import torchaudio
import pandas as pd
import numpy as np
from weathermuse import WeatherMuse
import weather_conditioner

'''
    city_name: should be lowercase; reference the data set for available names
    norm: normalization method - either 'minmax' or 'z_score' as a string value
    path: absolute path to the folder containing the dataset
'''
def preprocess_data(city_name, norm, path):
    weather_data = pd.read_csv(f'{path}{city_name}_{norm}.csv')
    weather_data = weather_data.dropna()
    weather_data = weather_data.iloc[:, 2:]
    weather_numpy = weather_data.to_numpy()
    weather_tensor = torch.tensor(weather_numpy, dtype=torch.float32)
    return weather_tensor


#### Generate music from weather data:

# Get weather data for processing and convert to tensor:
path = 'D:/File Storage/Documents/UTC/Thesis/WeatherMuse/audiocraft/hourly_weather_data/'
weather_tensor = preprocess_data('atlanta', 'minmax', path)
model = WeatherMuse.get_pretrained('facebook/musicgen-small')



# Inspect the model's output
