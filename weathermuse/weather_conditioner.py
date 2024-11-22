import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from audiocraft.modules.conditioners import BaseConditioner

class WeatherConditioner(BaseConditioner):
    def __init__(self, num_attributes: int, output_dim: int, max_hours: int, pad_idx: int = 0):
        super().__init__(dim=num_attributes, output_dim=output_dim)
        self.num_attributes = num_attributes  # Number of weather attributes (e.g., temperature, humidity)
        self.max_hours = max_hours            # Max number of hours to consider
        self.pad_idx = pad_idx                # Padding index for missing hours

    def tokenize(self, data: List[Optional[List[float]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize weather data where each hour is a token, and dimensions are weather attributes.

        Args:
            data (List[List[float]]): List of hourly weather data, where each hour's data is a list of attribute values.
            e.g., [[temp1, humidity1, pressure1], [temp2, humidity2, pressure2], ...]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Condition tensor of shape [B, T, D] and mask tensor of shape [B, T].
        """
        # Initialize condition tensor with padding values and mask tensor
        condition = torch.full((len(data), self.max_hours, self.num_attributes), self.pad_idx, dtype=torch.float32)
        mask = torch.zeros(len(data), self.max_hours, dtype=torch.int)

        # Populate condition tensor and mask with actual data
        for i, hourly_data in enumerate(data):
            for j, hour in enumerate(hourly_data[:self.max_hours]):  # Limit to max_hours
                if hour is None or len(hour) != self.num_attributes:
                    continue  # Skip this hour if it contains incomplete data
                condition[i, j] = torch.tensor(hour)
                mask[i, j] = 1  # Mark as valid data

        return condition, mask

    def forward(self, tokenized_data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply embedding projection to weather data and apply mask.

        Args:
            tokenized_data (Tuple[torch.Tensor, torch.Tensor]): Condition tensor and mask tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Embedded condition tensor and mask tensor.
        """
        condition, mask = tokenized_data
        # Project condition tensor to output dimension
        condition_embedded = self.output_proj(condition)
        condition_embedded = condition_embedded * mask.unsqueeze(-1)  # Apply mask to remove padding effects
        return condition_embedded, mask
