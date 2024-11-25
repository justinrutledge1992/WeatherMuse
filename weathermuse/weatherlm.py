from audiocraft.models.lm import LMModel
from weather_conditioner import WeatherConditioningAttributes
import typing as tp
import torch
ConditionType = tp.Tuple[torch.Tensor, torch.Tensor]  # condition, mask
ConditionTensors = tp.Dict[str, ConditionType]

class WeatherLM(LMModel):
    def forward(self, sequence: torch.Tensor,
                conditions: tp.List[WeatherConditioningAttributes],
                condition_tensors: tp.Optional[ConditionTensors] = None,
                stage: int = -1) -> torch.Tensor:
        """
        Override forward method to handle weather embeddings gracefully.
        """
        B, K, S = sequence.shape
        assert K == self.num_codebooks, "Sequence shape must match the specified number of codebooks"
        input_ = sum([self.emb[k](sequence[:, k]) for k in range(K)])
        
        if condition_tensors is None:
            assert not self._is_streaming, "Conditions tensors should be precomputed when streaming."
            # Apply dropout modules
            conditions = self.cfg_dropout(conditions)
            conditions = self.att_dropout(conditions)
            tokenized = self.condition_provider.tokenize(conditions)
            # Encode conditions and fuse, both have a streaming cache to not recompute when generating.
            condition_tensors = self.condition_provider(tokenized)
        else:
            assert not conditions, "Shouldn't pass both conditions and condition_tensors."

        # Handle missing cross_attention_input gracefully
        input_, cross_attention_input = self.fuser(input_, condition_tensors)
        if cross_attention_input is None:  # Ensure it's not None
            cross_attention_input = torch.zeros_like(input_)

        out = self.transformer(input_, cross_attention_src=cross_attention_input,
                                src_mask=(self.attn_mask_per_stage[stage] if stage >= 0 else None))  # type: ignore
        if self.out_norm:
            out = self.out_norm(out)
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1)  # [B, K, S, card]

        # Remove the prefix from the model outputs
        if len(self.fuser.fuse2cond['prepend']) > 0:
            logits = logits[:, :, -S:]

        return logits  # [B, K, S, card]