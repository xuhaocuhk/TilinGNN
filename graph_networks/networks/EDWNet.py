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

class EDWNet(torch.nn.Module):
    def __init__(self,
                 adj_edge_features_dim,
                 col_edge_feature_dim,
                 output_dim,
                 node_features_dim = environment.tile_count + 1,
                 with_collision_branch = config.with_collision_branch
                 ):
        super(EDWNet, self).__init__()


        self.brch_1_layer_feature_dims = [config.middle_feature_size] + [config.middle_feature_size] * config.network_depth
        self.brch_2_layer_feature_dims = [1]                          + [config.middle_feature_size] * config.network_depth

        self.residual_skip_num = 2
        self.with_collision_branch = with_collision_branch

        ############################# MLP process the raw features #############################
        self.brch_1_init_node_trans = MLP(in_dim=node_features_dim, out_dim=config.middle_feature_size, hidden_layer_dims=[config.middle_feature_size], activation=torch.nn.LeakyReLU(), batch_norm=True)

        ############################# graph convolution layers #############################
        self.brch_1_graph_conv_layers = nn.ModuleList([GraphConv(edge_feature_dim     = adj_edge_features_dim,
                                                                 node_feature_in_dim  = self.brch_1_layer_feature_dims[i],
                                                                 node_feature_out_dim = self.brch_1_layer_feature_dims[i+1])
                                                       for i in range(len(self.brch_1_layer_feature_dims)-1)])
        if self.with_collision_branch:
            self.brch_2_coll_conv_layers  = nn.ModuleList([CollConv(node_feature_in_dim  = self.brch_2_layer_feature_dims[i],
                                                                node_feature_out_dim = self.brch_2_layer_feature_dims[i+1])
                                                           for i in range(len(self.brch_2_layer_feature_dims)-1)])


        ############################# output layers #############################
        self.final_mlp = Sequential(MLP(in_dim=sum(self.brch_1_layer_feature_dims), out_dim=config.middle_feature_size,
                                        hidden_layer_dims=[256, 128, 64], activation=torch.nn.LeakyReLU()),
                                        Linear_trans(config.middle_feature_size, output_dim, activation = torch.nn.Sigmoid(), batch_norm=False)
                                    )


    def forward(self, x, adj_e_index, adj_e_features, col_e_idx, col_e_features = None):
        brch_1_x = x
        brch_2_x = torch.ones((x.shape[0],1)).to(device)

        ############################# MLP process the raw features #############################
        brch_1_x       = self.brch_1_init_node_trans(brch_1_x)

        ############################# main procedure #############################
        middle_features = [brch_1_x]
        for i in range(len(self.brch_1_layer_feature_dims)-1):
            conv_layer = self.brch_1_graph_conv_layers[i]
            coll_layer = self.brch_2_coll_conv_layers[i]
            brch_1_x, *_ = conv_layer(brch_1_x, adj_e_index, adj_e_features)
            if self.with_collision_branch:
                brch_2_x, *_ = coll_layer(brch_2_x, col_e_idx )
                brch_1_x = brch_1_x * brch_2_x

            # residual connection
            previous_layer_num = i - self.residual_skip_num
            if previous_layer_num >= 0:
                brch_1_x = brch_1_x + middle_features[previous_layer_num]

            middle_features.append(brch_1_x)


        ############################# output layers #############################
        skip_connec_features = torch.cat(middle_features, 1)

        node_features = self.final_mlp(skip_connec_features)

        return node_features, adj_e_features


if __name__ == "__main__":
    pass


