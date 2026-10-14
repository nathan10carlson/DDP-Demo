import torch
import torch.nn as nn

class CNN2D(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_CNN_layer = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.ReLU = nn.ReLU()
        self.second_CNN_layer = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.dense_1 = nn.LazyLinear(128)
        # there are 10 classes
        self.dense_2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.first_CNN_layer(x)
        x = self.ReLU(x)
        x = self.second_CNN_layer(x)
        x = self.ReLU(x)
        x = torch.flatten(x,1)
    
        x = self.dense_1(x)
        x = self.ReLU(x)
        x = self.dense_2(x)
        # this returns logits
        return x