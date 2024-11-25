import torch
import typing as tp
from collections import defaultdict
from audiocraft.modules.conditioners import BaseConditioner
from audiocraft.modules.conditioners import ConditioningProvider
from audiocraft.modules.conditioners import ConditioningAttributes
from audiocraft.utils.utils import length_to_mask
from dataclasses import dataclass, field
ConditionType = tp.Tuple[torch.Tensor, torch.Tensor]  # condition, mask

# Extend the ConditioningAttributes base class
@dataclass
class WeatherConditioningAttributes(ConditioningAttributes):
    weather: tp.Dict[str, torch.Tensor] = field(default_factory=dict)

class WeatherConditioningProvider(ConditioningProvider):
    def __init__(self, conditioners, device="cpu"):
        super().__init__(conditioners, device)

    @property
    def weather_conditions(self):
        return [k for k, v in self.conditioners.items() if k == 'weather']

    def _collate_weather(self, samples):
        weather = defaultdict(list)
        for sample in samples:
            for condition in self.weather_conditions:
                weather[condition].append(sample.weather.get(condition, None))
        return weather

class WeatherConditioner(BaseConditioner):
    def __init__(self, num_attributes: int, output_dim: int, max_hours: int):
        """
        Args:
            num_attributes (int): Number of weather attributes (e.g., temperature, humidity).
            output_dim (int): Output embedding dimension.
            max_hours (int): Maximum number of hours for weather data.
        """
        super().__init__(dim=num_attributes, output_dim=output_dim)
        self.num_attributes = num_attributes
        self.max_hours = max_hours

    def tokenize(self, data: torch.Tensor) -> tp.Dict[str, torch.Tensor]:
        """
        Tokenize weather data into embeddings.

        Args:
            data (torch.Tensor): Weather data tensor of shape [T, num_attributes].

        Returns:
            dict: Dictionary with:
                "embeddings": Condition embeddings.
                "lengths": Actual lengths of each input sequence.
        """
        if data.ndim != 2 or data.shape[1] != self.num_attributes:
            raise ValueError(f"Expected input shape [T, {self.num_attributes}], got {data.shape}")

        # Truncate to `max_hours` and return as is (no padding yet)
        truncated = data[:self.max_hours]
        return {"embeddings": truncated, "lengths": torch.tensor([len(truncated)])}

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        """
        Forward method for WeatherConditioner.

        Args:
            inputs (dict): Dictionary containing:
                - 'embeddings': Tensor of shape [B, T, num_attributes]
                - 'lengths': Tensor of shape [B], indicating sequence lengths.

        Returns:
            tuple: A tuple containing:
                - Tensor of shape [B, T, output_dim]: The embeddings.
                - Tensor of shape [B, T]: The mask.
        """
        embeddings = inputs['embeddings']  # Shape: [B, T, num_attributes]
        lengths = inputs['lengths']        # Shape: [B]

        # Project embeddings to output_dim
        embeddings = self.output_proj(embeddings)  # Shape: [B, T, output_dim]

        # Ensure the mask matches the sequence length of embeddings
        max_len = embeddings.shape[0]  # Match the sequence length of embeddings
        mask = length_to_mask(lengths, max_len=max_len).int()  # Shape: [B, T]

        # TROUBLSHOOTING
        print(f"embeddings.shape: {embeddings.shape}")
        print(f"lengths: {lengths}")
        print(f"mask.shape (before unsqueeze): {mask.shape}")
        print(f"inputs['embeddings'].shape: {inputs['embeddings'].shape}")
        print(f"inputs['lengths']: {inputs['lengths']}")

        # Apply mask to embeddings
        embeddings = embeddings * mask.unsqueeze(-1)  # Broadcast mask to match embedding shape

        return embeddings, mask
    
