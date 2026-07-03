from typing import Dict, Any

import torch
from torch import nn

from src.nn.user_encoder import UserEncoder


class EvalNIPModel(nn.Module):
    def __init__(
            self,
            num_items: int,
            embedding_dim: int,
            max_seq_len: int = 100,
            n_layers: int = 2,
            n_heads: int = 2,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = UserEncoder(
            num_items=num_items,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        encoder_output = self.encoder(inputs)
        cumulative_lengths = torch.cumsum(inputs['length'], dim=0)
        last_indices = cumulative_lengths - 1
        last_hidden_states = encoder_output[last_indices]

        scores = last_hidden_states @ self.encoder.item_embeddings.weight.T
        return scores
