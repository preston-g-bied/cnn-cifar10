import torch
import layers

class LeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.ConvolutionalLayer(3, 6, 5, padding=2)
        self.conv2 = layers.ConvolutionalLayer(6, 16, 5)
        self.pool = layers.AvgPool2DLayer(2, 2)
        self.tanh = layers.TanhLayer()
        self.flatten = torch.nn.Flatten()
        self.fc1 = layers.FullyConnectedLayer(576, 120)
        self.fc2 = layers.FullyConnectedLayer(120, 84)
        self.fc3 = layers.FullyConnectedLayer(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.tanh(x)

        x = self.fc2(x)
        x = self.tanh(x)

        x = self.fc3(x)

        return x