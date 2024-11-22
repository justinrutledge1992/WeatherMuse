# Driver code for the weather interpreter extension of MusicGen

from audiocraft.models import MusicGen
import torch
import torchaudio
import pandas as pd
import numpy as np
import weathermuse
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
print(weather_tensor)

'''
# Load a pretrained model:
model = MusicGen.get_pretrained('facebook/musicgen-small')

# Set the generation parameters (duration in seconds)
model.set_generation_params(duration=10)

# Generate the music (optionally, you can specify duration in seconds)
generated_audio = model.generate(weather_tensor, progress=True)

# Write the audio output
file_name = "audio.wav"
output_path = f"D:/File Storage/Documents/UTC/Thesis/WeatherMuse/audiocraft/weathermuse/output_files/{file_name}"
torchaudio.save(output_path, generated_audio[0].cpu(), model.sample_rate)
print(f"Audio saved to {output_path}")
'''