import os
import torch
import numpy as np
import random
from copy import deepcopy
import time
from shapely.geometry import Polygon
import shapely
from collections import OrderedDict
from util.algo_util import softmax
from util.data_util import write_brick_layout_data, write_bricklayout
from tiling.tile_factory import save_all_layout_info
from solver.ml_solver.losses import Losses
from inputs import config
import pickle
EPS = 1e-7

def solve_by_probablistic_greedy(ml_solver, origin_layout):

    node_num = origin_layout.node_feature.shape[0]
    collision_edges = origin_layout.collide_edge_index

    ## Initial the variables
    current_solution = SelectionSolution(node_num)
    round_cnt = 1

    while len(current_solution.unlabelled_nodes) > 0:

        ## create layout for currently unselected nodes
        temp_layout, node_re_index = origin_layout.compute_sub_layout(current_solution)
        prob = ml_solver.predict(temp_layout)

        ### compute new probs_value
        previous_prob = np.array(list(current_solution.unlabelled_nodes.values()))
        prob_per_node = np.power(np.power(previous_prob, round_cnt - 1) * prob, 1 / round_cnt)

        ## update the prob saved
        for i in range(len(prob_per_node)):
            current_solution.unlabelled_nodes[node_re_index[i]] = prob_per_node[i]  ## update the prob

        ## argsort the prob in descending
        sorted_indices = np.argsort(-prob_per_node)

        for idx in sorted_indices:
            origin_idx = node_re_index[idx]

            ## collision handling
            if not origin_idx in current_solution.unlabelled_nodes:
                break

            ## conditional adding the node
            if np.exp((prob_per_node[idx] - 1)*1.0) > np.random.uniform():
                current_solution.label_node(origin_idx, 1, origin_layout)
                current_solution = label_collision_neighbor(collision_edges, current_solution, origin_idx, origin_layout)

        # update the count
        round_cnt += 1

    ### create bricklayout with prediction
    score, selection_predict, predict_order = create_solution(current_solution, origin_layout)

    return selection_predict, score, predict_order



def solve_by_treesearch(ml_solver, origin_layout,  is_random_network, time_limit = 200, top_k = 4, check_holes = False,
                        tree_search_layout_dir = None, plotter = None):
    print("start tree search")
    node_num = origin_layout.node_feature.shape[0]
    collision_edges = origin_layout.collide_edge_index

    queue = []

    # saving of result and to prevent same sol
    results = []
    results_dic = {}

    predict = SelectionSolution(node_num)
    queue.append(predict)
    start_time = time.time()

    layout_cnt = 0
    predict_cnt = 0
    original_selected = []  # selected tile in this step

    while len(queue) > 0 and time.time() - start_time < time_limit:

        # pop the queue
        choosen_predict_index = random.randint(0, len(queue) - 1)
        choosen_predict = queue.pop(choosen_predict_index)

        temp_layout, node_re_index = origin_layout.compute_sub_layout(choosen_predict)

        ## assert correctness of temp layout
        # assert_temp_layout(node_re_index, origin_layout, temp_layout)

        prob = ml_solver.predict(temp_layout, random_network=is_random_network)
        predict_cnt += 1
        ### if cuda out of memory then continue --> give up a layout
        if prob is None:
            continue

        ## get the top k losses index for selection
        prob_tensor = torch.from_numpy(prob).float().to(ml_solver.device)
        selected_map = get_best_prob_map(ml_solver, prob_tensor, temp_layout, top_k)

        for m in selected_map:
            prob_network = prob[:, m]

            ## update by the previous prediction prob
            log_prob_network = np.log(np.clip(prob_network, EPS, 1.0))
            log_prob_m = np.array(list(choosen_predict.unlabelled_nodes.values())) + log_prob_network
            prob_m = np.exp(log_prob_m / predict_cnt)


            # print(f"{predict_cnt} prob_network : {prob_network}")
            # print(f"{predict_cnt} log_prob_network : {log_prob_network}")
            # print(f"{predict_cnt} log_prob_m_before : {np.array(list(choosen_predict.unlabelled_nodes.values()))}")
            # print(f"{predict_cnt} log_prob_m : {log_prob_m}")
            # print(f"{predict_cnt} prob_m : {prob_m}")

            # create a new prediction for remove the nodes
            new_predict = deepcopy(choosen_predict)
            for i in range(len(log_prob_m)):
                new_predict.unlabelled_nodes[node_re_index[i]] = log_prob_m[i]  ## update the prob

            # save the layout if necessary
            if tree_search_layout_dir is not None and config.output_tree_search_layout:
                new_temp_layout, new_reindex = origin_layout.compute_sub_layout(new_predict)
                np.save(os.path.join(tree_search_layout_dir, f'{layout_cnt}_predict_prob.npy'), prob_network)
                save_temp_layout(layout_cnt, new_temp_layout, tree_search_layout_dir, plotter)
                layout_cnt += 1

            ## get the order for nodes for selection
            node_order_array = get_nodes_order_array(prob_m, top_k)


            for idx in node_order_array:

                assert np.array(origin_layout.predict).shape[0] == len(new_predict.labelled_nodes) + len(new_predict.unlabelled_nodes)
                origin_idx = node_re_index[idx]

                ## collision handling
                if not origin_idx in new_predict.unlabelled_nodes:
                    if np.random.random() < prob_m[idx]:
                        continue
                    else:
                        break

                ## holes handling
                if check_holes and exist_hole(origin_layout, origin_idx, new_predict):
                    continue

                ### label the node if no constraint violated
                new_predict.label_node(origin_idx, 1, origin_layout)
                original_selected.append(origin_idx)
                new_predict = label_collision_neighbor(collision_edges, new_predict, origin_idx, origin_layout)

            if tree_search_layout_dir is not None and config.output_tree_search_layout:
                for i in original_selected:
                    origin_layout.predict[i] = 1
                # origin_layout.show_predict(plotter, os.path.join(tree_search_layout_dir, f'{layout_cnt-1}_image.png'))
                pickle.dump(original_selected, open(os.path.join(tree_search_layout_dir, f'{layout_cnt-1}_selected.pkl'), "wb"))

            # if not empty amd have changes then add back to queue and
            if len(new_predict.unlabelled_nodes) != 0 and len(choosen_predict.labelled_nodes) != len(new_predict.labelled_nodes):
                queue.append(new_predict)
            else:
                score, temp_sol, predict_order = create_solution(new_predict, origin_layout)

                # check whether solution exists
                hash_key = ''.join([str(int(s)) for s in temp_sol])
                if hash_key not in results_dic:
                    results_dic[hash_key] = 1
                    results.append((temp_sol, score, predict_order))

    if tree_search_layout_dir is not None and config.output_tree_search_layout:
        pickle.dump(original_selected, open(os.path.join(tree_search_layout_dir, f'final_selected.pkl'), "wb"))

    results = sorted(results, key=lambda tup: tup[1], reverse=True)
    return results, predict_cnt


def save_temp_layout(layout_cnt, temp_layout, tree_search_layout_dir, plotter = None):
    write_bricklayout(tree_search_layout_dir, f'{layout_cnt}_data.pkl', temp_layout)
    if plotter is not None:
        temp_layout.show_candidate_tiles(plotter = plotter, file_name = os.path.join(tree_search_layout_dir, f'{layout_cnt}_data.png'))


def exists_holes_or_collisions(new_predict, origin_layout, origin_idx, check_holes):
    has_collision = not origin_idx in new_predict.unlabelled_nodes
    has_hole = check_holes and exist_hole(origin_layout, origin_idx, new_predict)

    return has_collision or has_hole

def label_collision_neighbor(collision_edges, new_predict, origin_idx, origin_layout):
    # label neigbhour
    if not len(collision_edges) == 0:
        neighbor_collid_tiles = collision_edges[1][collision_edges[0] == origin_idx]
    else:
        neighbor_collid_tiles = []
    for adj_n in neighbor_collid_tiles:
        if adj_n in new_predict.unlabelled_nodes:
            assert adj_n not in new_predict.labelled_nodes
            new_predict.label_node(adj_n, 0, origin_layout)

    return new_predict


def create_solution(new_predict, origin_layout):
    # calculate the score
    temp_sol = np.zeros(origin_layout.node_feature.shape[0])
    order_predict = []
    for key, value in new_predict.labelled_nodes.items():
        if value == 1:
            temp_sol[key] = 1
            order_predict.append(key)

    score = Losses.solution_score(temp_sol, origin_layout)
    return score, temp_sol, order_predict


def get_nodes_order_array(prob_m, top_k):
    # sampling from the distribution if top k != 1
    # if top_k == 1:
    #     sampled_elem_array = np.argsort(-prob_m)
    # else:
    ALPHA = 100.0
    sampling_prob = np.clip(softmax(prob_m * ALPHA), a_min=1e-7, a_max=1 - 1e-7)
    sampling_prob = sampling_prob / sum(sampling_prob)
    assert len(sampling_prob) > 0
    sampled_elem_array = np.random.choice(len(sampling_prob), len(sampling_prob), replace=False,
                                          p=sampling_prob)
    return sampled_elem_array


def assert_temp_layout(node_re_index, origin_layout, temp_layout):
    ####################### Checking the generated bricklayout
    for i in range(temp_layout.node_feature.shape[0]):
        assert np.all(temp_layout.node_feature[i] == origin_layout.node_feature[node_re_index[i]])
    if len(temp_layout.collide_edge_index) != 0:
        for i in range(temp_layout.collide_edge_index.shape[1]):
            assert [node_re_index[temp_layout.collide_edge_index[0, i]],
                    node_re_index[temp_layout.collide_edge_index[1, i]]] in origin_layout.collide_edge_index.T
    if len(temp_layout.align_edge_index) != 0:
        for i in range(temp_layout.align_edge_index.shape[1]):
            assert [node_re_index[temp_layout.align_edge_index[0, i]],
                    node_re_index[temp_layout.align_edge_index[1, i]]] in origin_layout.align_edge_index.T
    ####################### Checking the generated bricklayout


def exist_hole(origin_layout, origin_idx, new_predict):
    index_in_complete_graph = origin_layout.inverse_index[origin_idx]
    new_tile = origin_layout.complete_graph.tiles[index_in_complete_graph].tile_poly
    new_polygon = new_predict.current_polygon.union(new_tile.buffer(EPS))
    if isinstance(new_polygon, shapely.geometry.polygon.Polygon):
        if len(list(new_polygon.interiors)) > 0:
            return True
    elif isinstance(new_polygon, shapely.geometry.multipolygon.MultiPolygon):
        if any([len(list(new_polygon[i].interiors)) > 0 for i in range(len(new_polygon))]):
            return True
    else:
        print("error occurs in hole checking!!!!")
        input()

    return False

def check_connected(origin_layout, origin_idx, new_predict):
    index_in_complete_graph = origin_layout.inverse_index[origin_idx]
    new_tile = origin_layout.complete_graph.tiles[index_in_complete_graph].tile_poly
    new_polygon = new_predict.current_polygon.union(new_tile.buffer(EPS))

    if isinstance(new_polygon, shapely.geometry.polygon.Polygon):
        return True
    elif isinstance(new_polygon, shapely.geometry.multipolygon.MultiPolygon):
        return False
    else:
        print("error occurs in connection checking!!!!")
        input()
        return False

class SelectionSolution(object):
    def __init__(self, node_num):
        self.labelled_nodes = OrderedDict(sorted({}.items()))
        self.unlabelled_nodes = OrderedDict(sorted({key: 1.0 for key in range(node_num)}.items()))
        self.current_polygon = Polygon([])

    def label_node(self, node_idx, node_label, brick_layout):
        self.labelled_nodes[node_idx] = node_label
        self.unlabelled_nodes.pop(node_idx)
        ## update polygon if the node_label is 1
        if node_label == 1:
            node_index_in_complete_graph = brick_layout.inverse_index[node_idx]
            self.current_polygon = self.current_polygon.union(brick_layout.complete_graph.tiles[node_index_in_complete_graph].tile_poly.buffer(EPS))
