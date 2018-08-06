import torch.nn as nn
import torch.utils.model_zoo as model_zoo

"""
Simple fully connected layers
Used to combine resnet features with category counts
"""
class FullyConnected(nn.Module):

    def __init__(self, num_in, num_out):
        super(FullyConnected, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(num_in, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_out)
        )

    def forward(self, x):
        x = self.model(x)
        return x
