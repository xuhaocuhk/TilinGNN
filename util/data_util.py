import pickle
import os
from collections import defaultdict

import numpy as np
import time
import urllib.request
import ssl
import tiling.brick_layout
import torch

from tiling.tile_graph import TileGraph

ssl._create_default_https_context = ssl._create_unverified_context
optional_variables_names = ['node_features', 'collide_edge_index', 'collide_edge_features', 'align_edge_index',
                            'align_edge_features', 'predict', 'predict_order', 'target_shape', 'predict_probs']

def write_brick_layout_data(save_path, re_index,
                            node_features = None, collide_edge_index = None, collide_edge_features = None, align_edge_index = None,
                            align_edge_features = None, prefix = None, predict = None, predict_order = None, target_shape = None,
                            predict_probs = None):
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    dic = {
        "re_index": re_index
    }

    for optional_variable_name in optional_variables_names:
        save_var = locals()[optional_variable_name]
        if save_var is not None:
            dic[optional_variable_name] = save_var

    pickle.dump(dic, open(os.path.join(prefix, save_path), "wb"))

# def write_re_index(re_index, save_path):
#     pickle.dump(re_index, open(save_path, "wb"))

def load_brick_layout_data(save_path):
    f = pickle.load(open(save_path, "rb"))

    assert (
            're_index' in f.keys()
    )
    dic = {
        're_index' : f['re_index']
    }

    ### check variable exist in the list
    for optional_variable_name in optional_variables_names:
        if optional_variable_name in f.keys():
            dic[optional_variable_name] = f[optional_variable_name]
        else:
            dic[optional_variable_name] = None

    return dic['re_index'], dic['node_features'], dic['collide_edge_index'], dic['collide_edge_features'], dic['align_edge_index'], \
                            dic['align_edge_features'], dic['predict'], dic['predict_order'], dic['target_shape'], dic['predict_probs']

def load_bricklayout(file_path, complete_graph):
    re_index, node_features, collide_edge_index, collide_edge_features, align_edge_index,\
    align_edge_features, predict, predict_order, target_polygon, predict_probs = load_brick_layout_data(file_path)

    ### reconstruct the features if needed
    if node_features is None or collide_edge_index is None or \
        collide_edge_features is None or align_edge_index is None or align_edge_features is None:
        node_features, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features = recover_features_from_reindex(re_index, complete_graph)

    output_layout = tiling.brick_layout.BrickLayout(complete_graph, node_features, collide_edge_index,
                               collide_edge_features, align_edge_index, align_edge_features, re_index)
    if predict is not None:
        output_layout.predict = predict
    if predict_order is not None:
        output_layout.predict_order = predict_order
    if target_polygon is not None:
        output_layout.target_polygon = target_polygon
    if predict_probs is not None:
        output_layout.predict_probs = predict_probs

    return output_layout

def write_bricklayout(folder_path, file_name, brick_layout, with_features = True):

    if with_features:
        node_features = brick_layout.node_feature
        collide_edge_index = brick_layout.collide_edge_index
        collide_edge_features = brick_layout.collide_edge_features
        align_edge_index = brick_layout.align_edge_index
        align_edge_features = brick_layout.align_edge_features
    else:
        node_features = None
        collide_edge_index = None
        collide_edge_features = None
        align_edge_index = None
        align_edge_features = None


    write_brick_layout_data(save_path = file_name,
                            node_features= node_features,
                            collide_edge_index = collide_edge_index,
                            collide_edge_features = collide_edge_features,
                            align_edge_index = align_edge_index,
                            align_edge_features = align_edge_features,
                            re_index = brick_layout.re_index,
                            prefix = folder_path,
                            predict = brick_layout.predict,
                            predict_order = brick_layout.predict_order,
                            target_shape = brick_layout.target_polygon,
                            predict_probs = brick_layout.predict_probs
                            )

def to_torch_tensor(device, node_feature, align_edge_index, align_edge_features, collide_edge_index, collide_edge_features):
    x                     = torch.from_numpy(node_feature).float().to(device)
    adj_edge_index        = torch.from_numpy(align_edge_index).long().to(device)
    adj_edge_features     = torch.from_numpy(align_edge_features).float().to(device)
    collide_edge_index    = torch.from_numpy(collide_edge_index).long().to(device)
    collide_edge_features = torch.from_numpy(collide_edge_features).float().to(device)

    return x, adj_edge_index, adj_edge_features, collide_edge_index, collide_edge_features

def download_file(url, save_path):
    print('Beginning file download with urllib2...')
    print(f'download link : {url}')
    start_time = time.time()
    urllib.request.urlretrieve(url, save_path)
    print(f"Download done in {time.time() - start_time}s!")

def write_tree_search_layout(save_path, temp_layout, node_re_index):

    print(f"Writing temp_layout to {save_path}....")
    dic = {
        "temp_layout" : temp_layout,
        "node_re_index" : node_re_index
    }

    pickle.dump(dic, open(save_path, "wb"))
    print(f"Done writing temp_layout to {save_path}....")

def load_tree_search_layout(load_path):
    f = pickle.load(open(load_path, "rb"))
    assert (
        "temp_layout" in f.keys() and
        "node_re_index" in f.keys()
    )
    temp_layout = f["temp_layout"]
    node_re_index = f["node_re_index"]

    return temp_layout, node_re_index

def recover_features_from_reindex(re_index, complete_graph):

    tiles_super_set = list(re_index.keys())

    filtered_collided_edges = [edge for edge in complete_graph.colli_edges if
                               edge[0] in tiles_super_set and edge[1] in tiles_super_set]
    filtered_adj_edges = [edge for edge in complete_graph.adj_edges if
                               edge[0] in tiles_super_set and edge[1] in tiles_super_set]
    node_features, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features, new_re_index = \
        generate_brick_layout_data(complete_graph, tiles_super_set, filtered_collided_edges, filtered_adj_edges)

    for key, item in new_re_index.items():
        assert re_index[key] == item

    return node_features, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features

def generate_brick_layout_data(graph: TileGraph, super_tiles: list, collide_edges, adj_edges):
    # edge feature
    super_edge_features_collide = [graph.edges_features[edge[0]][edge[1]] for edge in collide_edges]
    super_edge_features_adj = np.array([graph.edges_features[edge[0]][edge[1]] for edge in adj_edges])
    if len(super_edge_features_adj) > 0:
        super_edge_features_adj[:,1] = super_edge_features_adj[:,1] / graph.max_align_length

    for edge in super_edge_features_collide:
        assert edge is not None
    for edge in super_edge_features_adj:
        assert edge is not None

    # print("Edge feature")
    # print(super_edge_features)

    # re-index: mapping from complete graph to the current super graph
    re_index = defaultdict(int)
    for i in range(len(super_tiles)):
        re_index[super_tiles[i]] = i

    # node feature
    node_feature = np.zeros((len(super_tiles), graph.tile_type_count+1 ))
    for i, tile_idx in enumerate(super_tiles):
        current_tile = graph.tiles[tile_idx]
        node_feature[i][current_tile.id] = 1
        node_feature[i][-1] = current_tile.area() / graph.max_area


    # print("Node feature")
    # print(node_feature)

    # edge index
    edge_index_collide = list(map(lambda e: (re_index[e[0]], re_index[e[1]]), collide_edges))
    edge_index_align = list(map(lambda e: (re_index[e[0]], re_index[e[1]]), adj_edges))

    # print("Edge index")
    # print(edge_index)


    return node_feature, np.array(edge_index_collide).T, np.array(super_edge_features_collide), \
           np.array(edge_index_align).T, np.array(super_edge_features_adj), re_index

if __name__ == "__main__":
    download_file("https://www.dropbox.com/s/72vfss3nl1vqhaf/train_net1.89.png", './train_net1.89.png')

