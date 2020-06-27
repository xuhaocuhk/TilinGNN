from tiling.brick_layout import BrickLayout
import os
import torch
import numpy as np
from solver.base_solver import BaseSolver
from util.data_util import write_bricklayout, load_bricklayout
from copy import deepcopy
from solver.ml_solver.trainer import Trainer
from solver.ml_solver.losses import Losses
from graph_networks.network_utils import get_network_prediction
import util.algorithms as algorithms

class ML_Solver(BaseSolver):
    def __init__(self,
                 debugger,
                 device,
                 complete_graph,
                 network,
                 num_prob_maps
                 ):
        super(ML_Solver, self).__init__()
        self.debugger = debugger
        self.device = device
        self.complete_graph = complete_graph
        self.network = network
        self.random_network = deepcopy(self.network)
        self.num_prob_maps = num_prob_maps

    def predict(self, brick_layout : BrickLayout):
        predictions = None
        if len(brick_layout.collide_edge_index) == 0 or len(brick_layout.align_edge_index) == 0:  # only collision edges left, select them all
            predictions = torch.ones((brick_layout.node_feature.shape[0], self.num_prob_maps)).float().to(self.device)
        else:
            # convert to torch tensor
            x, adj_edge_index, adj_edge_features, collide_edge_index, collide_edge_features = \
                brick_layout.get_data_as_torch_tensor(self.device)

            # get network prediction
            predictions, *_ = self.network(x=x,
                                adj_e_index=adj_edge_index,
                                adj_e_features=adj_edge_features,
                                col_e_idx=collide_edge_index,
                                col_e_features=collide_edge_features)

        ## get the minimium loss map
        best_map_index = get_best_prob_map(self, predictions, brick_layout)
        selected_prob = predictions[:, best_map_index].detach().cpu().numpy()

        return selected_prob

    def get_unsupervised_losses_from_layout(self, brick_layout, probs):
        x, adj_edge_index, adj_edge_features, collide_edge_index, collide_edge_features = \
            brick_layout.get_data_as_torch_tensor(self.device)

        _, _, losses = Losses.calculate_unsupervised_loss(probs, x, collide_edge_index,
                                                          adj_edges_index=adj_edge_index, adj_edge_features=adj_edge_features)
        return losses

    def solve(self, brick_layout : BrickLayout):
        output_solution, score, predict_order = algorithms.solve_by_probablistic_greedy(self, brick_layout)

        output_layout = deepcopy(brick_layout)
        output_layout.predict_order = predict_order
        output_layout.predict = output_solution
        output_layout.predict_probs = self.predict(brick_layout)

        return output_layout, score

    def get_predict_probs(self, brick_layout : BrickLayout):
        x, adj_edge_index, adj_edge_features, collide_edge_index, collide_edge_features = \
            brick_layout.get_data_as_torch_tensor(self.device)

        probs = get_network_prediction(
                     network = self.network,
                     x=x,
                     adj_e_index=adj_edge_index,
                     adj_e_features=adj_edge_features,
                     col_e_idx=collide_edge_index,
                     col_e_features=collide_edge_features)

        return probs

    # select one probability map from all possibilities
    def select_solution(self, brick_layout, probs, is_by_supervise_loss):
        x, adj_edge_index, adj_edge_features, collide_edge_index, collide_edge_features = \
            brick_layout.get_data_as_torch_tensor(self.device)

        assert len(probs) > 0
        _, min_index, _ = Trainer.calculate_loss_unsupervise(probs, x, collide_edge_index, adj_edge_index, adj_edge_features)
        return min_index

    def save_debug_info(self, plotter, sample_data, data_path, save_dir_root):
        if not os.path.isdir(self.debugger.file_path(save_dir_root)):
            os.makedirs(self.debugger.file_path(save_dir_root))

        for i, data_idx in enumerate(sample_data):
            # random choose a sample in data
            rand_data = f"data_{data_idx}.pkl"

            # create folder for each sample
            save_dir = os.path.join(save_dir_root, f"data_{i}")
            if not os.path.isdir(self.debugger.file_path(save_dir)):
                os.mkdir(self.debugger.file_path(save_dir))

            brick_layout = load_bricklayout(os.path.join(data_path, rand_data), self.complete_graph)

            # solve by greedy tree search
            predict, score, predict_order = algorithms.solve_by_probablistic_greedy(self, brick_layout)
            brick_layout.predict = predict
            brick_layout.predict_order = predict_order
            brick_layout.predict_probs = self.predict(brick_layout)

            brick_layout.show_predict(plotter, self.debugger, os.path.join(save_dir,
                                                               f"greddy_predict_{score}.png"))
            write_bricklayout(self.debugger.file_path(save_dir), f"greedy_layout_{score}.pkl", brick_layout, with_features = False)

            ####### DEBUG FOR ASSERTION
            reloaded_bricklayout = load_bricklayout(os.path.join(self.debugger.file_path(save_dir), f"greedy_layout_{score}.pkl"),
                             complete_graph = self.complete_graph)
            BrickLayout.assert_equal_layout(reloaded_bricklayout, brick_layout)

            # visualize the result by all thershold and all map
            brick_layout.show_predict_prob(plotter, self.debugger, os.path.join(save_dir, f"network_prob_visualization.png"))

            brick_layout.show_candidate_tiles(plotter, self.debugger, os.path.join(save_dir, f"superset.png"))
            brick_layout.show_super_contour(plotter, self.debugger, os.path.join(save_dir, f"super_contour.png"))


    def load_saved_network(self, net_path):
        self.network.load_state_dict(torch.load(net_path, map_location = self.device))
        self.network.train()

def get_best_prob_map(ml_solver, prob_tensor, temp_layout):
    losses = ml_solver.get_unsupervised_losses_from_layout(temp_layout, prob_tensor)
    selected_map = np.argsort(losses)[0]
    return selected_map