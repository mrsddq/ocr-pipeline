from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 1, hidden_size: int = 256, lstm_layers: int = 2) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.rnn = nn.LSTM(256, hidden_size, lstm_layers, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        features = F.adaptive_avg_pool2d(features, (1, features.shape[-1])).squeeze(2)
        sequence = features.permute(0, 2, 1)
        output, _ = self.rnn(sequence)
        return self.classifier(output).log_softmax(dim=-1).permute(1, 0, 2)
