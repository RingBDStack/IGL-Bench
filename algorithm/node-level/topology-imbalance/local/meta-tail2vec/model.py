import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TwoLayerNN(nn.Module):
    def __init__(self, input_dim=1433, hidden_dim1=64, hidden_dim2=32, output_dim=None):
        super(TwoLayerNN, self).__init__()
        if output_dim is None:
            raise ValueError("Please specify the number of output classes in 'output_dim'")
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(p=0.5)
        self.layer3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        # x = F.normalize(x, p=1, dim=1)
        return x

