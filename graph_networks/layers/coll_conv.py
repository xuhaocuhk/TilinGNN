import torch
from torch.nn import Sequential, Linear
from torch_geometric.nn import GINConv
import torch.nn as nn
from graph_networks.layers.util import MLP

class CollConv(nn.Module):
    def __init__(self, node_feature_in_dim, node_feature_out_dim,
                 hidden_dims      = [32, 64],
                 aggr             = "mean",
                 batch_norm       = True,
                 mlp_activation   = torch.nn.Sigmoid(),
                 final_activation = torch.nn.LeakyReLU()):

        super(CollConv, self).__init__()

        mlp = MLP(in_dim=node_feature_in_dim, out_dim= node_feature_out_dim, hidden_layer_dims = hidden_dims, activation=mlp_activation, batch_norm = False)
        self.ginConv = GINConv(nn = mlp)

        self.activation = final_activation
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(node_feature_out_dim)

    def forward(self, x, edge_index):
        x = self.ginConv(x, edge_index)
        if self.activation is not None:
            x = self.activation(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return x, edge_index

