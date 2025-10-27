# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU, InstanceNorm1d, BatchNorm1d, Dropout
from torch_geometric.data import Data

class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = InstanceNorm1d(hidden_channels)
        self.dropout1 = Dropout(p=0.2)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn2 = InstanceNorm1d(out_channels)
        self.dropout2 = Dropout(p=0.2)

    def forward(self, x, edge_index):
        assert x.dim() == 2, f"Input x should have 2 dimensions, but got {x.dim()}"
        num_nodes = x.size(0)
        mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        edge_index = edge_index[:, mask]
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        return x


class GINModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GINModel, self).__init__()
        nn1 = Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn=nn1)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.dropout1 = Dropout(p=0.2)
        nn2 = Sequential(Linear(hidden_channels, out_channels))
        self.conv2 = GINConv(nn=nn2)
        self.bn2 = BatchNorm1d(out_channels)
        self.dropout2 = Dropout(p=0.2)

    def forward(self, x, edge_index, batch):
        if edge_index.dim() > 2:
            edge_index = edge_index.reshape(2, -1)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        batch = batch.to(x.device)
        x = global_add_pool(x, batch)
        return x

class PPIModel(torch.nn.Module):
    def __init__(self, gcn_in_channels=1024, gcn_hidden_channels=64, gcn_out_channels=64, 
                 gin_in_channels=64, gin_hidden_channels=64, gin_out_channels=32):
        super(PPIModel, self).__init__()
        self.gcn_model = GCNModel(gcn_in_channels, gcn_hidden_channels, gcn_out_channels)
        self.gin_model = GINModel(gin_in_channels, gin_hidden_channels, gin_out_channels)
        self.fc = Linear(gin_out_channels, 1)  
        self.dropout = Dropout(p=0.2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gcn_model(x, edge_index)
        output = self.gin_model(x, edge_index, batch)
        output = self.dropout(output)
        output = self.fc(output)
        return output


def load_ppi_model(model_path, device):
    model = PPIModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model