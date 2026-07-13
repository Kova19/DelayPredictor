'''
Bachelor thesis FIT VUT
Author: Martin Kováčik (xkovacm01)
Date: 21.6.2026

Neural network main class
'''

import torch
import torch.nn as nn


class DelayPredictor(nn.Module):
    def __init__(self, vocab_sizes: dict, embedding_dims: dict = None, n_continuous=18):

        super().__init__()

        # rozumné výchozí embedding dimenze podle velikosti kategorie
        default_dims = {
            "line": min(16, (vocab_sizes["line"] + 1) // 2),
            "route": min(32, (vocab_sizes["route"] + 1) // 2),
            "vehicleType": min(4, (vocab_sizes["vehicleType"] + 1) // 2),
            "stop": min(32, (vocab_sizes["stop"] + 1) // 2),
        }
        embedding_dims = embedding_dims or default_dims

        self.emb_line = nn.Embedding(vocab_sizes["line"], embedding_dims["line"])
        self.emb_route = nn.Embedding(vocab_sizes["route"], embedding_dims["route"])
        self.emb_vehicleType = nn.Embedding(vocab_sizes["vehicleType"], embedding_dims["vehicleType"])
        self.emb_stop = nn.Embedding(vocab_sizes["stop"], embedding_dims["stop"])

        total_emb_dim = sum(embedding_dims.values())
        input_dim = total_emb_dim + n_continuous

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
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
        # x: (batch, 23) -> prvních 4 sloupce jsou kategorické
        line = x[:, 0].long()
        route = x[:, 1].long()
        vehicleType = x[:, 2].long()
        stop = x[:, 3].long()
        cont_x = x[:, 4:]  # zbylých 19 hodnot

        emb = torch.cat([
            self.emb_line(line),
            self.emb_route(route),
            self.emb_vehicleType(vehicleType),
            self.emb_stop(stop),
        ], dim=1)

        out = torch.cat([emb, cont_x], dim=1)
        return self.net(out)

'''

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

'''
