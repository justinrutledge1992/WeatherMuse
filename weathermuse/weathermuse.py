from audiocraft.models.musicgen import MusicGen
from audiocraft.modules.conditioners import ConditioningAttributes
from weather_conditioner import WeatherConditioner
import torch

class WeatherMuse(MusicGen):
    def __init__(self, name, compression_model, lm, max_duration=None):
        super().__init__(name, compression_model, lm, max_duration)
        # Add the weather conditioner to the conditioning provider
        self.lm.condition_provider.conditioners['weather'] = WeatherConditioner(
            num_attributes=5,  # temperature, humidity, pressure, wind direction, wind speed
            output_dim=512,    # embedding dimension
            max_hours=24       # a full day of data
        )

    def generate_with_weather(self, weather_data, progress=False, return_tokens=False):
        """
        Generate samples conditioned on weather data.

        Args:
            weather_data (list of lists): List of hourly weather data sequences.
            progress (bool): Flag to display progress during generation.
            return_tokens (bool): Whether to return audio tokens along with the audio.
        
        Returns:
            torch.Tensor or tuple: Generated audio, and optionally the audio tokens.
        """
        # Prepare conditioning attributes for weather data
        attributes = self._prepare_tokens_and_attributes(weather_data)
        tokens = self._generate_tokens(attributes, prompt_tokens=None, progress=progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)
    
    @torch.no_grad()
    def _prepare_tokens_and_attributes(self, weather_data):
        """
        Prepare model inputs for weather conditioning.

        Args:
            weather_data (list of lists): List of hourly weather data sequences.

        Returns:
            list: ConditioningAttributes with weather data tokens.
        """
        attributes = [
            ConditioningAttributes() for _ in weather_data  # One ConditioningAttributes per batch item
        ]

        # Set weather data conditioning for each item in the batch
        for attr, weather in zip(attributes, weather_data):
            attr.wav['weather'] = self.lm.condition_provider.conditioners['weather'].tokenize(weather)

        return attributes  # No prompt tokens required