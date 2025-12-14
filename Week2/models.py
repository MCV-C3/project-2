
import torch.nn as nn
import torch

from typing import *

class SimpleModel(nn.Module):

    def __init__(self, input_d: int, hidden_d: int, output_d: int):

        super(SimpleModel, self).__init__()

        self.input_d = input_d
        self.hidden_d = hidden_d
        self.output_d = output_d


        self.layer1 = nn.Linear(input_d, hidden_d)
        self.layer2 = nn.Linear(hidden_d, hidden_d)
        self.output_layer = nn.Linear(hidden_d, output_d)

        self.activation = nn.ReLU()


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)

        x = self.output_layer(x)
        
        return x

class DynamicMLP(nn.Module):
    def __init__(self, layer_sizes: list, activation: str = "ReLU", dropout: float = 0.0):
        """
        layer_sizes: list of tuples like [(in1, out1), (in2, out2), ...]
        activation: string name of activation: "ReLU", "Tanh", "Sigmoid", etc.
        dropout: dropout probability (0.0 disables dropout)
        """
        super().__init__()

        layers = []
        act_class = getattr(nn, activation)
        drop = nn.Dropout(dropout) if dropout > 0 else None

        for i, (inp, out) in enumerate(layer_sizes):
            # Add linear layer
            layers.append(nn.Linear(inp, out))
            
            # Add BatchNorm after each linear layer (except last layer)
            if i < len(layer_sizes) - 1:
                layers.append(nn.BatchNorm1d(out))
            
                # Apply activation after BatchNorm
                layers.append(act_class())

                # Add Dropout if needed
                if drop is not None:
                    layers.append(drop)

        self.model = nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = x.view(x.size(0), -1)
        features = []

        for layer in self.model:
            x = layer(x)
            features.append(x)

        if return_features:
            return features

        return x
