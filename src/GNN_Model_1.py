import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool, GATConv, SAGEConv, SAGEConv, BatchNorm
import HyperParameters

### HYPER PARAMETERS ###
CLASSES = HyperParameters.CLASSES
BATCH_SIZE = HyperParameters.BATCH_SIZE
HIDDEN_UNITS = HyperParameters.HIDDEN_UNITS
OUTPUT_SHAPE = len(CLASSES)
LEARNING_RATE = HyperParameters.LEARNING_RATE
EPOCHS = HyperParameters.LEARNING_RATE

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_UNITS, output_dim=OUTPUT_SHAPE, dropout_rate=.2, NUM_HEADS=16):
        super(GNN, self).__init__()
        self.gconv1 = GATConv(input_dim, 50, heads=NUM_HEADS)
        #self.gconv2 = GraphConv(hidden_dim*NUM_HEADS, hidden_dim)
        self.gconv2 = GATConv(50*NUM_HEADS, 40, heads=NUM_HEADS)
        self.gconv3 = GraphConv(40*NUM_HEADS, 40)
        self.gconv4 = SAGEConv(40, 40)
        '''self.gconv1 = GATConv(input_dim, hidden_dim, heads=NUM_HEADS)
        self.gconv2 = GraphConv(hidden_dim*NUM_HEADS, hidden_dim)

        self.bn = torch.nn.BatchNorm1d(HIDDEN_UNITS)

        self.gconv3 = SAGEConv(hidden_dim, hidden_dim)
        self.gconv4 = SAGEConv(hidden_dim, hidden_dim)
        self.gconv5 = GATConv(hidden_dim, hidden_dim, heads=NUM_HEADS)
        self.gconv6 = GATConv(hidden_dim*NUM_HEADS, hidden_dim, heads=NUM_HEADS)'''

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        
        self.fc1 = torch.nn.Linear(40, 32)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(32, output_dim)

    def forward(self, x, edge_index, batch):
        # Add this check
        #assert edge_index.max() < x.size(0), f"Max edge index {edge_index.max()} is >= number of nodes {x.size(0)}"
        #Pass the data through the Graph Conv Layers
        x = F.relu(self.gconv1(x, edge_index))
        x = F.relu(self.gconv2(x, edge_index))
        x = F.relu(self.gconv3(x, edge_index))
        #x = self.bn(x)
        x = F.relu(self.gconv4(x, edge_index))
        '''x = self.dropout(x)
        x = F.relu(self.gconv5(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gconv6(x, edge_index))'''
        #x = self.dropout(x)
        #x = F.relu(self.gconv6(x, edge_index))
        #Aggregate the output using global_mean_pool
        x = global_mean_pool(x, batch)
        #Apply Non-linearity with relu
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        #x = F.relu(self.fc2(x))
        #Pass to the output layer, grabbing the prediction logits.
        x = self.fc3(x)
        
        return x
