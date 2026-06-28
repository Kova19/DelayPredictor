'''
Bachelor thesis FIT VUT
Author: Martin Kováčik (xkovacm01)
Date: 21.6.2026

Neural network main class
'''

import torch.nn as nn


class DelayPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(512),
            nn.BatchNorm1d(512),
            nn.SiLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


