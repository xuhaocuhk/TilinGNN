from solver.minizinc_solver.minizinc_solver import MinizincSolver
from tiling.brick_layout import BrickLayout
import os
import torch
import numpy as np
import random
from solver.base_solver import BaseSolver
from util.data_util import load_bricklayout, write_bricklayout, load_bricklayout, to_torch_tensor
from copy import deepcopy
from solver.ml_solver.trainer import Trainer
from solver.ml_solver.losses import Losses
from graph_networks.network_utils import get_network_prediction
import inputs.config as config
import traceback
import solver.algorithms.algorithms as algorithms

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
        if len(brick_layout.collide_edge_index) == 0:  # only collision edges left, select them all
            # print("empty edges!")
            return np.ones((brick_layout.node_feature.shape[0], self.num_prob_maps))
        elif len(brick_layout.align_edge_index) == 0:
            solver = MinizincSolver(model_file='./solver/minizinc_solver/solve_contour_multiTile_with_align.mzn',
                                    solver_type='coin-bc',
                                    debugger=self.debugger)
            brick_layout, _ = solver.solve(brick_layout, verbose = False)
            return np.tile(np.array(brick_layout.predict),(self.num_prob_maps,1)).T
        else:
            # get prediction
            ################ WARNING
            # the result might have some problem whens using only 1 type of edge due to empty edge set
            x, adj_edge_index, adj_edge_features, collide_edge_index, collide_edge_features = \
                brick_layout.get_data_as_torch_tensor(self.device)
            # print('node_size:', x.size(0))
            # print('adj_edge:', edge_index)
            # print('collide_edge:', collide_edge_index)
            probs = get_network_prediction(network = self.network,
                                           x = x,
                                           adj_e_index = adj_edge_index,
                                           adj_e_features = adj_edge_features,
                                           col_e_idx = collide_edge_index,
                                           col_e_features = collide_edge_features)
            ### return None for the functions if cuda memory error
            if probs is None:
                input("network output is None!!!")
                return None

            return probs.detach().cpu().numpy()

    def get_unsupervised_losses_from_layout(self, brick_layout, probs):
        x, adj_edge_index, adj_edge_features, collide_edge_index, collide_edge_features = \
            brick_layout.get_data_as_torch_tensor(self.device)

        _, _, losses = Losses.calculate_unsupervised_loss(probs, x, collide_edge_index,
                                                          adj_edges_index=adj_edge_index, adj_edge_lengths=adj_edge_features[:, 1])
        return losses

    def solve(self, brick_layout : BrickLayout, intermediate_results_dir = None):
        output_solution, score, predict_order = algorithms.solve_by_probablistic_greedy(self, brick_layout, tree_search_layout_dir=intermediate_results_dir)

        output_layout = deepcopy(brick_layout)
        output_layout.predict_order = predict_order
        output_layout.predict = output_solution

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

    def visualise_result_by_transparent_color(self, brick_layout : BrickLayout, plotter, save_dir):

        probs = self.get_predict_probs(brick_layout)
        # No picture can be drawn if out of memory
        if probs is None:
            return

        ### get unsupervised loss for all mapping####
        losses = self.get_unsupervised_losses_from_layout(brick_layout, probs)
        min_index = np.argsort(losses)[0]
        brick_layout.predict_probs = probs[:, min_index].view(-1).float().detach().cpu().numpy()

        brick_layout.show_predict_with_transparent_color(plotter, os.path.join(self.debugger.file_path(save_dir),
                                                        f"data_prob_map_trans_predict.png"))

    def save_debug_info(self, plotter, sample_data, data_path, save_dir_root):
        if not os.path.isdir(self.debugger.file_path(save_dir_root)):
            os.makedirs(self.debugger.file_path(save_dir_root))

        for i, data_idx in enumerate(sample_data):
            # random choose a sample in data
            rand_data = f"data_{data_idx}.pkl"

            # create folder for each sample
            save_dir = os.path.join(self.debugger.file_path(save_dir_root), f"data_{i}")
            obj_save_dir = os.path.join(save_dir, 'objs')
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            brick_layout = load_bricklayout(os.path.join(data_path, rand_data), self.debugger, self.complete_graph)

            # solve by greedy tree search
            results, _ = self.solve_by_algorithms(brick_layout, time_limit = config.search_time_limit, is_random_network= False, top_k = 1)
            for idx, res in enumerate(results):
                prediction, score, _ = res
                brick_layout.predict = prediction
                brick_layout.show_predict(plotter, os.path.join(self.debugger.file_path(save_dir),
                                                                   f"tree_search_predict_greedy_{idx}_{score}.png"))
                brick_layout.save_predict_as_objs(os.path.join(obj_save_dir,
                                                               f"tree_search_predict_greedy_{idx}_objs"), file_name = "tile")
                write_bricklayout(self.debugger.file_path(save_dir), f"tree_search_layout_{idx}_{score}.pkl", brick_layout, with_features = False)

                ####### DEBUG FOR ASSERTION
                reloaded_bricklayout = load_bricklayout(os.path.join(self.debugger.file_path(save_dir), f"tree_search_layout_{idx}_{score}.pkl"), debugger = self.debugger,
                                 complete_graph = self.complete_graph)
                BrickLayout.assert_equal_layout(reloaded_bricklayout, brick_layout)


            # solve by tree search !!!! removed
            # TREE_SEARCH_RESULT_CNT = 10
            # results, _ = self.solve_by_treesearch_new(brick_layout, time_limit = config.search_time_limit, is_random_network=False, top_k = 4)
            # results = results[:TREE_SEARCH_RESULT_CNT]
            # for idx, res in enumerate(results):
            #     prediction, score, _ = res
            #     brick_layout.predict = prediction
            #     brick_layout.show_predict(plotter, os.path.join(self.debugger.file_path(save_dir),
            #                                                        f"tree_search_predict_top_4_{idx}_{score}.png"))
            #     brick_layout.save_predict_as_objs(os.path.join(obj_save_dir,
            #                                                    f"tree_search_predict_top_4_{idx}_objs"), file_name = "tile")

            # visualize the result by all thershold and all map
            self.visualise_result_by_transparent_color(brick_layout = brick_layout, plotter = plotter, save_dir = save_dir)

            brick_layout.show_candidate_tiles(plotter, os.path.join(self.debugger.file_path(save_dir), f"supper_graph.png"))
            brick_layout.show_super_contour(plotter,  os.path.join(self.debugger.file_path(save_dir), f"super_contour.png"))

    def fine_tune_on_one_data(self, brick_layout : BrickLayout, plotter):

        fine_tune_times = 300
        ##### OPTIMIZER ####
        optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)

        ##### DATA ######
        x, adj_edge_index, adj_edge_features, collide_edge_index, *_ = to_torch_tensor(device = self.device,
                        node_feature = brick_layout.node_feature,
                        align_edge_index = brick_layout.align_edge_index,
                        align_edge_features = brick_layout.align_edge_features,
                        collide_edge_index = brick_layout.collide_edge_index,
                        collide_edge_features = brick_layout.collide_edge_features,
                        ground_truth = np.array([]))

        ########### Training #####################
        before_fine_tune_result_path = brick_layout.debugger.file_path('before_fine_tune')
        if not os.path.isdir(before_fine_tune_result_path):
            os.mkdir(before_fine_tune_result_path)
        self.visualise_result_by_transparent_color(brick_layout=brick_layout, plotter = plotter, save_dir= before_fine_tune_result_path)

        for i in range(fine_tune_times):
            probs = get_network_prediction(network=self.network,
                                           x=x,
                                           adj_e_index=adj_edge_index,
                                           adj_e_features=adj_edge_features,
                                           col_e_idx= collide_edge_index,
                                           col_e_features=None)
            if probs is None:
                continue
            try:
                optimizer.zero_grad()

                unsupervise_train_loss, *_ = Losses.calculate_unsupervised_loss(probs, x,
                                                                                collide_edge_index,
                                                                                adj_edges_index= adj_edge_index,
                                                                                adj_edge_lengths= adj_edge_features[:, 1])
                unsupervise_train_loss.backward()
                optimizer.step()
            except:
                print(traceback.format_exc())
                continue
            print (f"loss after training for {i} times : {unsupervise_train_loss}")

            if i % 20 == 0:
                fine_tune_path = brick_layout.debugger.file_path(f'fine_tune_{i}')
                if not os.path.isdir(fine_tune_path):
                    os.mkdir(fine_tune_path)
                self.visualise_result_by_transparent_color(brick_layout=brick_layout, plotter=plotter,
                                                           save_dir=fine_tune_path, top_k=4)


    def load_saved_network(self, net_path):
        self.network = torch.load(net_path, map_location = self.device)
        self.network.train()