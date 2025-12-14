
import torch.nn as nn
import torch
import torch.nn.functional as F
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
    def __init__(self, layer_sizes: list, activation: str = "ReLU"):
        """
        layer_sizes: list of tuples like [(in1, out1), (in2, out2), ...]
        activation: string name of activation: "ReLU", "Tanh", "Sigmoid", etc.
        """
        super().__init__()

        layers = []
        act = getattr(nn, activation)()  # convert string → nn.Module()

        for (inp, out) in layer_sizes:
            layers.append(nn.Linear(inp, out))
            layers.append(act)

        # remove last activation if not desired
        layers = layers[:-1]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)




class CNNDecoderModel(nn.Module):

    def __init__(self, input_d: int, hidden_d: int, output_d: int,):

        super(SimpleModel, self).__init__()

        self.input_d = input_d
        self.hidden_d = 300
        self.output_d = output_d


        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_stride=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_stride=3, stride=1, padding=1)
        self.layer1 = nn.Linear(16*7*7, hidden_d)
        self.layer2 = nn.Linear(hidden_d, hidden_d)
        self.output_layer = nn.Linear(hidden_d, output_d)

        self.activation = nn.ReLU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.activation(x)
        x  = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)

        x = self.output_layer(x)
        
        return x