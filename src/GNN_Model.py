import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv

import HyperParameters

### HYPER PARAMETERS ###
CLASSES = HyperParameters.CLASSES
BATCH_SIZE = HyperParameters.BATCH_SIZE
HIDDEN_UNITS = HyperParameters.HIDDEN_UNITS
OUTPUT_SHAPE = len(CLASSES)
LEARNING_RATE = HyperParameters.LEARNING_RATE
EPOCHS = HyperParameters.LEARNING_RATE

#Create the GNN Model
class GNN(nn.Module):
    def __init__(self, input_dim=3*128*128, hidden_dim=HIDDEN_UNITS, output_dim=OUTPUT_SHAPE):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x