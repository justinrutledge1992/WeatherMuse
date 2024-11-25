# Driver code for the weather interpreter extension of MusicGen

from weathermuse import WeatherMuse
import torch
import torchaudio
import pandas as pd
CITY = 'nashville'
FOLDER = 'iteration_1/'

'''
    CITY should be lowercase; reference the data set for available names
    norm: normalization method - either 'minmax' or 'z_score' as a string value
    path: absolute path to the folder containing the dataset

    This setup will take the first 24 hours available from each city's dataset
    and condition the ouput wave form (music) on that data.
'''
def preprocess_data(city_name, norm, path):
    weather_data = pd.read_csv(f'{path}{city_name}_{norm}.csv')
    weather_data = weather_data.dropna()
    weather_data = weather_data.iloc[:, 2:]
    weather_numpy = weather_data.to_numpy()
    weather_tensor = torch.tensor(weather_numpy, dtype=torch.float32)
    # Ensure the tensor is 3D: [B, T, D]
    if weather_tensor.ndim == 2:  # If shape is [T, D], add batch dimension
        weather_tensor = weather_tensor.unsqueeze(0)
    return weather_tensor


#### Generate music from weather data:

# Get weather data for processing and convert to tensor:
path = 'D:/File Storage/Documents/UTC/Thesis/WeatherMuse/audiocraft/hourly_weather_data/'
weather_tensor = preprocess_data(CITY, 'minmax', path)

# Load a pretrained model and provide the cleaned weather data:
model = WeatherMuse.get_pretrained('facebook/musicgen-small')

attributes = model._prepare_tokens_and_attributes(weather_data=weather_tensor)

# TEST

# Set the generation parameters (duration in seconds)
model.set_generation_params(duration=10)

# Generate the music (optionally, you can specify duration in seconds)
generated_audio = model.generate_with_weather(weather_data=weather_tensor, progress=True)

# Write the audio output
file_name = f'{CITY}.wav'
output_path = f"D:/File Storage/Documents/UTC/Thesis/WeatherMuse/audiocraft/weathermuse/output_files/{FOLDER}{file_name}"
torchaudio.save(output_path, generated_audio[0].cpu(), model.sample_rate)
print(f"Audio saved to {output_path}")
