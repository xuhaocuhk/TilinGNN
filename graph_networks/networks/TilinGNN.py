from graph_networks.layers.edge_conv import GraphConv
from graph_networks.layers.coll_conv import CollConv
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Sequential
from inputs.config import environment
from graph_networks.layers.util import MLP, Linear_trans
import inputs.config as config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TilinGNN(torch.nn.Module):
    def __init__(self,
                 adj_edge_features_dim,
                 network_depth,
                 network_width,
                 output_dim = 1,
                 node_features_dim = environment.tile_count + 1
                 ):
        super(TilinGNN, self).__init__()

        self.network_depth = network_depth
        self.network_width = network_width
        self.residual_skip_num = 2

        self.brch_1_layer_feature_dims = [self.network_width] * (self.network_depth + 1)
        self.brch_2_layer_feature_dims = [self.network_width] * (self.network_depth + 1)

        ############################# MLP process the raw features #############################
        self.init_node_feature_trans = MLP(in_dim=node_features_dim, out_dim=self.network_width, hidden_layer_dims=[self.network_width], activation=torch.nn.LeakyReLU(), batch_norm=True)

        ############################# graph convolution layers #############################
        self.brch_1_graph_conv_layers = nn.ModuleList([GraphConv(edge_feature_dim     = adj_edge_features_dim,
                                                                 node_feature_in_dim  = self.brch_1_layer_feature_dims[i],
                                                                 node_feature_out_dim = self.brch_1_layer_feature_dims[i+1])
                                                       for i in range(len(self.brch_1_layer_feature_dims)-1)])

        self.brch_2_coll_conv_layers  = nn.ModuleList([CollConv(node_feature_in_dim  = self.brch_2_layer_feature_dims[i],
                                                            node_feature_out_dim = self.brch_2_layer_feature_dims[i+1])
                                                       for i in range(len(self.brch_2_layer_feature_dims)-1)])


        ############################# output layers #############################
        self.final_mlp = Sequential(MLP(in_dim=sum(self.brch_1_layer_feature_dims), out_dim=self.network_width,
                                        hidden_layer_dims=[256, 128, 64], activation=torch.nn.LeakyReLU()),
                                        Linear_trans(self.network_width, output_dim, activation = torch.nn.Sigmoid(), batch_norm=False)
                                    )


    def forward(self, x, adj_e_index, adj_e_features, col_e_idx, col_e_features = None):

        ############################# MLP process the raw features #############################
        brch_1_feature = self.init_node_feature_trans(x)
        brch_2_feature = brch_1_feature

        ############################# main procedure #############################
        middle_features = [brch_1_feature]
        for i in range(self.network_depth):
            conv_layer = self.brch_1_graph_conv_layers[i]
            coll_layer = self.brch_2_coll_conv_layers[i]
            brch_1_feature, *_ = conv_layer(brch_1_feature, adj_e_index, adj_e_features)
            brch_2_feature, *_ = coll_layer(brch_2_feature, col_e_idx )
            brch_1_feature = brch_1_feature * brch_2_feature

            # residual connection
            previous_layer_num = i - self.residual_skip_num
            if previous_layer_num >= 0:
                brch_1_feature = brch_1_feature + middle_features[previous_layer_num]

            middle_features.append(brch_1_feature)

        ############################# output layers #############################
        skip_connec_features = torch.cat(middle_features, 1)

        node_features = self.final_mlp(skip_connec_features)

        return node_features, adj_e_features


if __name__ == "__main__":
    pass


