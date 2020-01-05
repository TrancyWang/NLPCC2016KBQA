import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import code

class SNLIModel(nn.Module):
    def __init__(self, hidden_size=1024, num_layers=3):
        super(SNLIModel, self).__init__()
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        self.linear1 = nn.Linear(4*hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x, y):
        layer_weights = torch.exp(F.log_softmax(self.layer_weights)).unsqueeze(0).unsqueeze(2)
        x = (layer_weights * x).sum(1)
        y = (layer_weights * y).sum(1)
        return F.tanh(self.linear2(F.relu(self.linear1(torch.cat([x, y, torch.abs(x-y), x*y], 1)))))

class WeightedAvgModel(nn.Module):
    def __init__(self, hidden_size=1024, num_layers=3):
        super(WeightedAvgModel, self).__init__()
        self.layer_weights = nn.Parameter(torch.ones(num_layers)) 
        # self.register_parameter("layer_weights", self.layer_weights)
        
    def forward(self, x):
        # code.interact(local=locals())
        layer_weights = torch.exp(F.log_softmax(self.layer_weights)).unsqueeze(0).unsqueeze(2)
        # code.interact(local=locals())
        x = (layer_weights * x).sum(1)
        return x

