
import torch.nn as nn
import torch

from typing import *

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

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
            if isinstance(layer, (nn.ReLU, nn.SiLU)):
                features.append(x)
            

        if return_features:
            return features

        return x


class SimpleCNN(nn.Module):
    """
    A very simple CNN with:
    - 2 convolutional parts (Conv -> ReLU -> MaxPool)
    - 1 FC classifier
    """
    def __init__(self, input_channels: int = 3, num_classes: int = 8):
        super(SimpleCNN, self).__init__()

        # First convolutional part
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional part
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size after convolutions and pooling
        # Input: 224x224 -> after pool1: 112x112 -> after pool2: 56x56
        # With 64 channels: 64 * 56 * 56 = 200704
        self.fc_input_size = 64 * 56 * 56

        # Fully connected classifier
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        # First conv part
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Second conv part
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class DynamicMLPCNN(nn.Module):
    """
    A CNN-MLP hybrid model that:
    1. Applies CNN layers for feature extraction
    2. Flattens the CNN output
    3. Passes features through a dynamic MLP

    This combines the spatial feature learning of CNNs with the flexibility of MLPs.
    """
    def __init__(self,
                 cnn_layers: List[Dict],
                 mlp_layer_sizes: List[Tuple[int, int]],
                 activation: str = "ReLU",
                 dropout: float = 0.0,
                 input_channels: int = 3):
        """
        Args:
            cnn_layers: List of dicts defining CNN layers. Each dict has:
                - out_channels: number of output channels
                - kernel_size: kernel size (default: 3)
                - padding: padding (default: 1)
                - pool_size: pooling kernel size (default: 2)
                - pool_stride: pooling stride (default: 2)
            mlp_layer_sizes: List of tuples for MLP layers [(in1, out1), (in2, out2), ...]
            activation: Activation function name
            dropout: Dropout probability
            input_channels: Number of input channels (default: 3 for RGB)
        """
        super().__init__()

        # Build CNN layers
        cnn_modules = []
        in_channels = input_channels
        act_class = getattr(nn, activation)

        for layer_cfg in cnn_layers:
            out_channels = layer_cfg['out_channels']
            kernel_size = layer_cfg.get('kernel_size', 3)
            padding = layer_cfg.get('padding', 1)
            pool_size = layer_cfg.get('pool_size', 2)
            pool_stride = layer_cfg.get('pool_stride', 2)

            # Conv -> BatchNorm -> Activation -> MaxPool
            cnn_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            cnn_modules.append(nn.BatchNorm2d(out_channels))
            cnn_modules.append(act_class())
            cnn_modules.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride))

            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_modules)

        # Build MLP layers
        mlp_modules = []
        drop = nn.Dropout(dropout) if dropout > 0 else None

        for i, (inp, out) in enumerate(mlp_layer_sizes):
            # Add linear layer
            mlp_modules.append(nn.Linear(inp, out))

            # Add BatchNorm after each linear layer (except last layer)
            if i < len(mlp_layer_sizes) - 1:
                mlp_modules.append(nn.BatchNorm1d(out))
                mlp_modules.append(act_class())

                # Add Dropout if needed
                if drop is not None:
                    mlp_modules.append(drop)

        self.mlp = nn.Sequential(*mlp_modules)

    def forward(self, x, return_features=False):
        """
        Forward pass through CNN then MLP.

        Args:
            x: Input tensor of shape (batch, channels, height, width)
            return_features: If True, returns intermediate features

        Returns:
            Output logits or features
        """
        features = []

        # CNN feature extraction
        cnn_out = self.cnn(x)
        if return_features:
            features.append(cnn_out)

        # Flatten
        flat = cnn_out.view(cnn_out.size(0), -1)
        if return_features:
            features.append(flat)

        # MLP classification
        for layer in self.mlp:
            flat = layer(flat)
            if return_features:
                features.append(flat)

        if return_features:
            return features

        return flat
