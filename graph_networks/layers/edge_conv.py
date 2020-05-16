import torch
from torch.nn import Sequential, Linear
from torch_geometric.nn.conv.nn_conv import NNConv
import torch.nn as nn
from graph_networks.layers.util import MLP

class GraphConv(nn.Module):
    def __init__(self, edge_feature_dim, node_feature_in_dim, node_feature_out_dim,
                 hidden_dims      = [32, 64],
                 aggr             = "mean",
                 batch_norm       = True,
                 mlp_activation   = torch.nn.Sigmoid(),
                 final_activation = torch.nn.LeakyReLU()):

        super(GraphConv, self).__init__()

        self.mlp = MLP(in_dim=edge_feature_dim, out_dim=node_feature_in_dim * node_feature_out_dim, hidden_layer_dims = hidden_dims, activation=mlp_activation, batch_norm = False)
        self.nnConv = NNConv(node_feature_in_dim, node_feature_out_dim, self.mlp, aggr=aggr)

        self.activation = final_activation
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(node_feature_out_dim)

    def forward(self, x, edge_index, edge_features):
        x = self.nnConv(x, edge_index, edge_features)
        if self.activation is not None:
            x = self.activation(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return x, edge_index, edge_features

