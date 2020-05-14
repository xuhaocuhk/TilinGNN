import networkx as nx
import os
from tiling.TileGraph import TileGraph
import tiling.brick_layout
from util.data_util import load_brick_layout_data
from util.debugger import MyDebugger
from interfaces.qt_plot import Plotter
import numpy as np
import matplotlib.pyplot as plt

def visual_brick_layout_graph(brick_layout, save_path, edge_type = "all", is_vis_prob = True, node_size = 40, edge_width = 0.7, xlim = (-1, 1.6), ylim = (-1, 1.6)):

    # create Graph
    G_symmetric = nx.Graph()
    col_edges = [ tuple(brick_layout.collide_edge_index[:, i]) for i in range(brick_layout.collide_edge_index.shape[1])]
    adj_edges = [tuple(brick_layout.align_edge_index[:, i]) for i in range(brick_layout.align_edge_index.shape[1])]
    if edge_type == "all":
        edges = col_edges + adj_edges
    elif edge_type == "collision":
        edges = col_edges
    elif edge_type == "adjacent":
        edges = adj_edges
    else:
        print(f"error edge type!!! {edge_type}")

    edge_color = ["gray" for i in range(len(edges))]

    # draw networks
    G_symmetric.add_nodes_from(range(brick_layout.node_feature.shape[0]))
    node_color = [brick_layout.predict_probs[i] if is_vis_prob else "blue" for i in range(brick_layout.node_feature.shape[0])]
    tile_indices = [ brick_layout.inverse_index[i] for i in range(brick_layout.node_feature.shape[0])]
    node_pos_pts = [brick_layout.complete_graph.tiles[index].tile_poly.centroid for index in tile_indices]
    node_pos = list(map(lambda pt : [pt.x, - pt.y], node_pos_pts))
    print(node_pos)
    # G_symmetric.add_edges_from(edges)

    vmin, vmax = 0.0, 1.0
    cmap = plt.cm.Reds
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    nx.draw_networkx(G_symmetric, pos = node_pos, node_size = node_size, node_color=node_color, cmap = cmap, width = edge_width, edgelist = edges, edge_color = edge_color,
                     vmin = vmin, vmax = vmax, with_labels = False, style = "dashed" if col_edges else "solid")

    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.savefig(save_path, dpi=400)
    print(f'saving file {save_path}...')
    plt.close()
    return

if __name__ == "__main__":
    MyDebugger.pre_fix = os.path.join(MyDebugger.pre_fix, "debug")
    debugger = MyDebugger("brick_layout", fix_rand_seed= True)
    plotter = Plotter()

    data_path = '../data/graph/raw/data_0.pkl'
    # env_name = '30-60-90'
    env_name = '30-60-90-normal-small'
    # env_name = '30-60-90-36-36-72'

    data_prefix = os.path.join('..', 'data', env_name)
    complete_graph_file = "complete_graph_ring4.pkl"

    # loading the complete graph
    tile_count = 1
    complete_graph = TileGraph(tile_type_count = tile_count)
    complete_graph.load_graph_state(os.path.join(data_prefix, complete_graph_file))

    # node_features, edge_index, edge_features, target, reindex = factory.gen_one_train_data(plotter, complete_graph, low=10, high=11)

    # brick_layout = BrickLayout(debugger, complete_graph, node_features, edge_index, edge_features, target, reindex)
    # brick_layout.probs = [random.random() for i in range(brick_layout.node_feature.shape[0])]
    # super graph
    # brick_layout.show_super_graph(plotter, f"supper_graph.png")

    # visual_brick_layout(brick_layout, debugger.file_path('.'))

