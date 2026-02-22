import torch
import layers

class AlexNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.ConvolutionalLayer(3, 64, 3, padding=1)
        self.conv2 = layers.ConvolutionalLayer(64, 128, 3, padding=1)
        self.conv3 = layers.ConvolutionalLayer(128, 256, 3, padding=1)
        self.conv4 = layers.ConvolutionalLayer(256, 256, 3, padding=1)
        self.pool = layers.MaxPool2DLayer(2, 2)
        self.relu = layers.ReLULayer()
        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc1 = layers.FullyConnectedLayer(4096, 1024)
        self.fc2 = layers.FullyConnectedLayer(1024, 256)
        self.fc3 = layers.FullyConnectedLayer(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x
    
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)