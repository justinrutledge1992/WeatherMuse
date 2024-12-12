from audiocraft.models.musicgen import MusicGen
from weather_conditioner import WeatherConditioner
from weather_conditioner import WeatherConditioningAttributes
from weather_conditioner import WeatherConditioningProvider
from audiocraft.models.loaders import load_compression_model
from helpers import load_lm_model
import torch
import typing as tp

class WeatherMuse(MusicGen):
    def __init__(self, name, compression_model, lm, max_duration=None):
        super().__init__(name, compression_model, lm, max_duration)
        device = next(self.lm.parameters()).device  # Get the device of the model
        self.device = device
        self.lm.condition_provider = WeatherConditioningProvider(
            {
                "weather": WeatherConditioner(
                    num_attributes=5,  # temperature, humidity, pressure, wind direction, wind speed
                    output_dim=512,    # embedding dimension
                    max_hours=24       # a full day of data
                ).to(device)  # Move to the same device as the model
            }
        )

    @staticmethod
    def get_pretrained(name: str = 'facebook/musicgen-small', device=None):
        """Override to return an instance of WeatherMuse."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load_compression_model is from audiocraft.models.loaders:
        compression_model = load_compression_model(name, device=device)

        # load_lm_model is from helpers.py, a tailor-made function:
        lm = load_lm_model(name, device=device)

        # Create an instance of WeatherMuse with the custom language model
        return WeatherMuse(name, compression_model, lm)

    def generate_with_weather(self, weather_data, progress=False, return_tokens=False):
        """
        Generate samples conditioned only on weather data.

        Args:
            weather_data (list): List of hourly weather data sequences.
            progress (bool): Flag to display progress.
            return_tokens (bool): Whether to return audio tokens.

        Returns:
            torch.Tensor: Generated audio or a tuple of audio and tokens.
        """
        # Move weather data to the correct device
        weather_data = weather_data.to(self.device)

        # Prepare conditioning attributes for weather data only
        attributes = self._prepare_tokens_and_attributes(weather_data=weather_data)

        # Generate tokens based on the attributes
        for attr in attributes:
            print(attr.weather) # Debug statement
        tokens = self._generate_tokens(attributes, prompt_tokens=None, progress=progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)
    
    def _prepare_tokens_and_attributes(self, weather_data=None):
        if weather_data is None or weather_data.ndim != 3:
            raise ValueError("Weather data must be a 3D tensor of shape [B, T, num_attributes].")

        # Move weather data to the same device as the model
        weather_data = weather_data.to(self.device)

        # Tokenize weather data for each sequence
        tokenized_data = [
            self.lm.condition_provider.conditioners["weather"].tokenize(weather)
            for weather in weather_data
        ]

        attributes = []
        for data in tokenized_data:
            attr = WeatherConditioningAttributes()
            attr.weather["weather"] = data  # Store tokenized data directly
            attributes.append(attr)

        return attributes
    
    def _generate_tokens(self, attributes: tp.List[WeatherConditioningAttributes],
                        prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
        """Generate discrete audio tokens for up to 30 seconds.

        Args:
            attributes (list of WeatherConditioningAttributes): Conditions used for generation (weather tensor).
            prompt_tokens (torch.Tensor, optional): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.

        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T].
        """
        max_duration = min(self.duration, self.max_duration, 30)  # Ensure duration is <= 30 seconds
        total_gen_len = int(max_duration * self.frame_rate)

        if prompt_tokens is not None:
            assert total_gen_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than the maximum allowed audio generation length."

        callback = None
        if progress:
            def _progress_callback(generated_tokens: int, tokens_to_generate: int):
                print(f'{generated_tokens: 6d} / {tokens_to_generate: 6d}', end='\r')
            callback = _progress_callback

        # Generate tokens for the allowed duration
        with self.autocast:
            gen_tokens = self.lm.generate(
                prompt_tokens, attributes,
                callback=callback, max_gen_len=total_gen_len, **self.generation_params
            )
        
        return gen_tokens