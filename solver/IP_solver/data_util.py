
from tiling.brick_layout import  BrickLayout
from util.tiling_util import polygon_align_length
import itertools
import numpy as np
EPS = 1e-7
def to_dzn_file(brick_layout : BrickLayout, save_path : str):

    # get the features needed
    nums_edge_collision, num_edge_adjacent, collision_edges, adjacent_edge = convert_layout_to_dzn_features(brick_layout)

    # write the data
    f = open(save_path, 'w')
    f.write(f"nums_node = {len(brick_layout.node_features)};\n")
    f.write(f"nums_edge_collision = {nums_edge_collision};\n")
    f.write(f"nums_edge_adjacent = {num_edge_adjacent};\n")
    f.write("from_adjacent = [" + ", ".join([str(edge + 1) for edge in adjacent_edge[0, :]]) + "];\n")
    f.write("to_adjacent = [" + ", ".join([str(edge + 1) for edge in adjacent_edge[1, :]]) + "];\n")
    f.write("from_collision = [" + ", ".join([str(edge + 1) for edge in collision_edges[0, :]]) + "];\n")
    f.write("to_collision = [" + ", ".join([str(edge + 1) for edge in collision_edges[1, :]]) + "];\n")

    f.close()

def convert_layout_to_dzn_features(brick_layout : BrickLayout, recompute_graph = False):

    # assert len(brick_layout.collide_edge_features.shape) > 1
    num_edge_collision = brick_layout.collide_edge_features.shape[0]
    collision_edges = brick_layout.collide_edge_index

    if not recompute_graph:
        num_edge_adjacent = brick_layout.align_edge_features.shape[0]
        adjacent_edge = brick_layout.align_edge_index
        align_edge_lengths = brick_layout.align_edge_features[:,1] if len(brick_layout.align_edge_features.shape) > 1 else []
    else:
        adjacent_edge = []
        align_edge_lengths = []
        num_edge_adjacent = 0
        for idx_i, idx_j in itertools.combinations(range(brick_layout.node_feature.shape[0]), 2):
            tile_i = brick_layout.complete_graph.tiles[brick_layout.inverse_index[idx_i]]
            tile_j = brick_layout.complete_graph.tiles[brick_layout.inverse_index[idx_j]]
            align_length = polygon_align_length(tile_i, tile_j)
            if  align_length > EPS:
                adjacent_edge.append([idx_i, idx_j])
                adjacent_edge.append([idx_j, idx_i])
                align_edge_lengths = align_edge_lengths + [align_length] * 2
                num_edge_adjacent += 2
        adjacent_edge = np.array(adjacent_edge).T
        align_edge_lengths = np.array(align_edge_lengths)


    return num_edge_collision, num_edge_adjacent, collision_edges, adjacent_edge, align_edge_lengths

if __name__ == '__main__':
    pass