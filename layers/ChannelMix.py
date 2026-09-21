import math

import torch
import torch.nn as nn


class ChannelMix(nn.Module):
    def __init__(self, num_channels, d_model, reduction=4, dropout=0.0):
        super().__init__()
        self.num_channels = int(num_channels)
        self.d_model = int(d_model)
        reduction = max(1, int(reduction))
        hidden_channels = max(1, math.ceil(self.num_channels / reduction))

        self.norm = nn.LayerNorm(self.d_model)
        self.fc1 = nn.Linear(self.num_channels, hidden_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_channels, self.num_channels)

    def forward(self, x):
        # x: [B, C, D], mix along channel axis C.
        if x.dim() != 3:
            raise ValueError('ChannelMix expects input with shape [B, C, D]')
        if x.size(1) != self.num_channels:
            raise ValueError(
                f'ChannelMix expected {self.num_channels} channels, got {x.size(1)}'
            )
        if x.size(2) != self.d_model:
            raise ValueError(
                f'ChannelMix expected feature dim {self.d_model}, got {x.size(2)}'
            )

        mixed = self.norm(x).transpose(1, 2)
        mixed = self.fc1(mixed)
        mixed = self.activation(mixed)
        mixed = self.dropout(mixed)
        mixed = self.fc2(mixed)
        mixed = self.dropout(mixed)
        return mixed.transpose(1, 2)
