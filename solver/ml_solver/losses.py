import torch
import numpy as np
import time
import inputs.config as config
import math
import itertools
import traceback
from graph_networks.network_utils import get_network_prediction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = 1e-7

class Losses:
    # evaluate loss for a given data set
    @staticmethod
    def evaluate_loss(network, data_set_loader, training_experiment=False):
        losses = []
        avg_collision_probs_list = []
        filled_area_list = []
        avg_align_length_list = []
        for batch in data_set_loader:

            try:
                data = batch.to(device)
                probs = get_network_prediction(
                    network=network,
                    x=data.x,
                    adj_e_index=data.edge_index,
                    adj_e_features=data.edge_features,
                    col_e_idx=data.collide_edge_index,
                    col_e_features=None)

                if probs is None:
                    continue

                loss, min_index, _ = Losses.calculate_unsupervised_loss(probs, data.x, data.collide_edge_index,
                                                                        adj_edges_index=data.edge_index,
                                                                        adj_edge_lengths=data.edge_features[:, 1])

                if training_experiment:
                    prob_selected = probs[:, min_index]
                    avg_collision_probs, filled_area, avg_align_length = Losses.get_statistics_for_prediction(
                        predict=prob_selected,
                        x=data.x,
                        adj_edge_index=data.edge_index,
                        adj_edge_features=data.edge_features,
                        collide_edge_index=data.collide_edge_index
                    )
                    avg_collision_probs_list.append(avg_collision_probs.detach().cpu().numpy())
                    filled_area_list.append(filled_area.detach().cpu().numpy())
                    avg_align_length_list.append(avg_align_length.detach().cpu().numpy())

                losses.append(loss.detach().cpu().numpy())
            except:
                print(traceback.format_exc())

        return np.mean((losses)), np.mean((avg_collision_probs_list)), np.mean((filled_area_list)), np.mean(
            (avg_align_length_list))


    @staticmethod
    def calculate_supervised_loss(probs, target):
        losses = []
        for p in range(probs.size()[1]):
            probs_selected = probs[:, p].view(-1)
            assert (probs_selected >= 0).all()
            assert (probs_selected <= 1).all()
            loss_each = torch.nn.BCELoss()(probs_selected, target)
            losses.append(loss_each)
        losses = torch.stack(losses)
        loss = torch.min(losses)
        min_index = torch.argmin(losses).detach().cpu().numpy()
        return loss, min_index


    @staticmethod
    def calculate_unsupervised_loss(probs, node_feature, collide_edge_index, adj_edges_index, adj_edge_lengths):
        # start time
        start_time = time.time()
        N = probs.shape[0]  # number of nodes
        M = probs.shape[1]  # number of output features
        E_col = collide_edge_index.shape[1]  # number of collision edges
        E_adj = adj_edge_lengths.shape[0]
        losses = []
        COLLISION_WEIGHT    = config.COLLISION_WEIGHT
        ALIGN_LENGTH_WEIGHT = config.ALIGN_LENGTH_WEIGHT
        AVG_AREA_WEIGHT     = config.AVG_AREA_WEIGHT

        for sol in range(M):
            solution_prob = probs[:, sol]
            ########### average node area loss
            avg_tile_area = torch.clamp(torch.mean(node_feature[:, -1] * solution_prob), min=eps)
            loss_ave_area = torch.log(avg_tile_area)

            ########### collision feasibility loss
            first_index = collide_edge_index[0, :]
            first_prob = torch.gather(solution_prob, dim=0, index=first_index)
            second_index = collide_edge_index[1, :]
            second_prob = torch.gather(solution_prob, dim=0, index=second_index)
            if collide_edge_index.size(1) > 0:
                prob_product = torch.clamp(first_prob * second_prob, min=eps, max=1 - eps)
                loss_per_edge = torch.log(1 - prob_product)
                loss_per_edge = loss_per_edge.view(-1)
                loss_feasibility = torch.sum(loss_per_edge) / E_col
            else:
                loss_feasibility = 0.0

            ########### edge length loss
            first_index = adj_edges_index[0, :]
            first_prob = torch.gather(solution_prob, dim=0, index=first_index)
            second_index = adj_edges_index[1, :]
            second_prob = torch.gather(solution_prob, dim=0, index=second_index)
            if adj_edges_index.size(1) > 0:
                if not (first_prob * second_prob * adj_edge_lengths >= 0).all() or not (first_prob * second_prob * adj_edge_lengths <= 1).all():
                    input()
                prob_product = torch.clamp(first_prob * second_prob * adj_edge_lengths, min=eps)
                loss_per_adjedge = torch.log(prob_product) / math.log(10)
                loss_per_adjedge = loss_per_adjedge.view(-1)
                loss_align_length = torch.sum(loss_per_adjedge) / E_adj
            else:
                loss_align_length = 0.0

            assert loss_feasibility <= 0
            assert loss_ave_area <= 0
            assert loss_align_length <= 0

            loss = (1 - AVG_AREA_WEIGHT     * loss_ave_area   ) * \
                   (1 - COLLISION_WEIGHT    * loss_feasibility) * \
                   (1 - ALIGN_LENGTH_WEIGHT * loss_align_length)

            assert loss >= 1.0

            losses.append(loss)

        losses = torch.stack(losses)
        loss = torch.min(losses)

        # print(f"unsupverised loss : {loss}, time_used = {time.time() - start_time}")
        min_index = torch.argmin(losses).detach().cpu().numpy()
        losses = losses.detach().cpu().numpy()
        return loss, min_index, losses


    @staticmethod
    def _calculate_pairwise_cross_entropy(probs):
        # transpose prob_1 and prob_2
        probs = probs.permute(1, 0)

        # get combinatorial
        combinations = itertools.product(range(probs.size(0)), range(probs.size(0)))
        index_1, index_2 = zip(*combinations)
        index_1 = torch.Tensor(index_1).long().to(device)
        index_2 = torch.Tensor(index_2).long().to(device)
        # print("probs :", probs)
        # print("index_1 :", index_1)
        # print("index_2 :", index_2)
        probs_1 = torch.index_select(probs, dim=0, index=index_1).to(device)
        probs_2 = torch.index_select(probs, dim=0, index=index_2).to(device)
        # print("probs_1 :", probs_1)
        # print("probs_2 :", probs_2)

        cross_entropy = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(probs_1, probs_2))
        # print("cross_entropy :", cross_entropy)
        return - cross_entropy

    # to evaluate the quality of a collision-free solution
    @staticmethod
    def solution_score(predict, brick_layout):
        predict = torch.from_numpy(np.array(predict)).float().to(device)
        x, adj_edge_index, adj_edge_features, collide_edge_index, collide_edge_features = \
            brick_layout.get_data_as_torch_tensor(device)

        ######## calculate total area
        filled_area = predict.dot(x[:,-1]*brick_layout.complete_graph.max_area) / brick_layout.get_super_contour_poly().area
        assert filled_area >= -1e-7 and filled_area <= 1+1e-7

        ######## calculate alignment length
        if adj_edge_features.size(0) > 0:
            adj_edge_lengths = adj_edge_features[:, 1] * brick_layout.complete_graph.max_align_length
            first_index = adj_edge_index[0, :]
            first_prob = torch.gather(predict, dim=0, index=first_index)
            second_index = adj_edge_index[1, :]
            second_prob = torch.gather(predict, dim=0, index=second_index)
            if not (first_prob * second_prob * adj_edge_lengths >= 0).all() or not (
                    first_prob * second_prob * adj_edge_lengths <= brick_layout.complete_graph.max_align_length).all():
                input()
            loss_align_length = (first_prob * second_prob).dot(adj_edge_lengths) # total edge length
        else:
            loss_align_length = 0.0

        all_edge_length = sum([brick_layout.complete_graph.tiles[brick_layout.inverse_index[i]].get_perimeter() for i in range(len(predict))
         if predict[i] == 1])

        assert (loss_align_length/all_edge_length) > -1e-7 and (loss_align_length/all_edge_length) < 1+1e-7

        return (config.AVG_AREA_WEIGHT * filled_area + config.ALIGN_LENGTH_WEIGHT * (loss_align_length/all_edge_length)).detach().cpu().item()

    @staticmethod
    def get_statistics_for_prediction(predict, x, adj_edge_index, adj_edge_features, collide_edge_index):

        ######## calculate total area
        filled_area = predict.dot(x[:, -1]) / sum(x[:, -1])
        assert filled_area >= -1e-7 and filled_area <= 1 + 1e-7

        ######## calculate alignment length
        if adj_edge_features.size(0) > 0:
            adj_edge_lengths = adj_edge_features[:, 1]
            first_index = adj_edge_index[0, :]
            first_prob = torch.gather(predict, dim=0, index=first_index)
            second_index = adj_edge_index[1, :]
            second_prob = torch.gather(predict, dim=0, index=second_index)
            avg_align_length = (first_prob * second_prob).dot(adj_edge_lengths) / sum(
                adj_edge_lengths)  # total edge length
            assert avg_align_length >= -1e-7 and avg_align_length <= 1 + 1e-7
        else:
            avg_align_length = 0.0

        ########### collision feasibility loss
        first_index = collide_edge_index[0, :]
        first_prob = torch.gather(predict, dim=0, index=first_index)
        second_index = collide_edge_index[1, :]
        second_prob = torch.gather(predict, dim=0, index=second_index)
        if collide_edge_index.size(1) > 0:
            prob_product = torch.clamp(first_prob * second_prob, min=eps, max=1 - eps)
            loss_per_edge = prob_product
            loss_per_edge = loss_per_edge.view(-1)
            avg_collision_probs = torch.mean(loss_per_edge)
            assert avg_collision_probs >= -1e-7 and avg_collision_probs <= 1 + 1e-7
        else:
            avg_collision_probs = 0.0

        return avg_collision_probs, filled_area, avg_align_length

    @staticmethod
    def loss_prediction_by_epsilon_greedy(probs, node_feature, collide_edge_index, adj_edges_index, adj_edge_lengths, epsilon):

        ## clamping probs
        probs = torch.clamp(probs, min = eps)

        _, min_index, _ = Losses.calculate_unsupervised_loss(probs, node_feature, collide_edge_index, adj_edges_index, adj_edge_lengths)
        min_prob = probs[:, min_index]

        if np.random.uniform() > epsilon:
            if config.greedy_sampling:
                prediction = Losses.greedy_sampling(min_prob, collide_edge_index)
            else:
                sampling_dist = torch.distributions.bernoulli.Bernoulli(min_prob)
                prediction = sampling_dist.sample().detach().to(device)  # ensure no gradient will be calculated here
        else:
            sampling_prob = torch.full((probs.size(0),), 0.5).to(device)
            sampling_dist = torch.distributions.bernoulli.Bernoulli(sampling_prob)
            prediction = sampling_dist.sample().detach().to(device)  # ensure no gradient will be calculated here

        ## loss
        prediction_loss, *_  = Losses.calculate_unsupervised_loss(prediction.unsqueeze(1), node_feature, collide_edge_index, adj_edges_index, adj_edge_lengths)

        ## actual sampling dist
        actual_sampling_dist = torch.distributions.bernoulli.Bernoulli(min_prob)
        log_prob = torch.mean(actual_sampling_dist.log_prob(prediction))

        ## baseline
        baseline_comparsion = torch.full((probs.size(0),), 0.5).to(device)
        prediction_loss_baseline, *_  = Losses.calculate_unsupervised_loss(baseline_comparsion.unsqueeze(1), node_feature, collide_edge_index, adj_edges_index, adj_edge_lengths)


        # policy gradient loss
        loss = (prediction_loss - prediction_loss_baseline) * log_prob
        
        print(f"loss : {loss}, log_prob : {log_prob}, prediction_loss : {prediction_loss}, prediction_loss_baseline : {prediction_loss_baseline}")
        return loss

    @staticmethod
    def greedy_sampling(min_prob : torch.Tensor, collide_edge_index : torch.Tensor):

        # convert to numpy
        min_prob = min_prob.detach().cpu().numpy()
        collide_edge_index = collide_edge_index.detach().cpu().numpy()

        # construct a dict for searching
        edge_dic = { i : {} for i in range(min_prob.shape[0])}
        for i in range(collide_edge_index.shape[1]):
            edge_dic[collide_edge_index[0,i]][collide_edge_index[1,i]] = 1

        sorted_indices = np.argsort(-min_prob)

        labelled_node = {}
        prediction = torch.zeros((min_prob.shape[0])).to(device)

        # argsort search solution
        for idx in sorted_indices:
            if idx not in labelled_node:
                prediction[idx] = 1
                labelled_node[idx] = 1
                for collide_idx in edge_dic[idx].keys():
                    labelled_node[collide_idx] = 1

        return prediction



