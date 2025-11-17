"""
Baseline MLP model for skeleton-based action recognition.
"""

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout, num_classes):
        super(BaselineMLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(), nn.Dropout(dropout)]

        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        return self.model(x)


def create_baseline_model(config):
    return BaselineMLP(
        input_size=config['input_size'],
        hidden_sizes=config['hidden_sizes'],
        dropout=config['dropout'],
        num_classes=config['num_classes']
    )