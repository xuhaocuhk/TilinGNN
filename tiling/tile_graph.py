# Python3 Program to print BFS traversal
# from a given source vertex. BFS(int s)
# traverses vertices reachable from s.
from collections import defaultdict
import numpy as np
import itertools
import math
import pickle
import util.tiling_util as tiling_util
from tiling.tile import Tile
EPS = 1e-5
ONE_HOT_EPS = 1e-3
# from inputs import config

def get_tile_instance(base_tile: Tile, base_edge_idx, align_tile: Tile, align_edge_idx, align_mode):
    base_edge_p0, base_edge_p1 = base_tile.get_edge(base_edge_idx)
    tile_instance = tiling_util.align_tile(base_edge_p1, base_edge_p0, align_tile.tile_poly, align_edge_idx, align_mode)
    return Tile(tile_instance, id = align_tile.id)

def get_all_tiles(base_tile: Tile, align_tile: Tile, integer_align):
    result_tiles = []
    align_tags = [] # a tag tuple to indicate the alignment types
    for base_edge_idx in range(base_tile.get_edge_num()):
        for align_neighbor_idx in range(align_tile.get_edge_num()):
            for align_mode in [0, 1]:
                align_tag = (base_tile.id, align_tile.id, base_edge_idx, align_neighbor_idx, align_mode)
                if integer_align:
                    base_edge_length = base_tile.get_edge_length(base_edge_idx)
                    tile_edge_length = align_tile.get_edge_length(align_neighbor_idx)
                    if abs(math.floor(base_edge_length / tile_edge_length + EPS) - base_edge_length / tile_edge_length) > EPS and \
                       abs(math.floor(tile_edge_length / base_edge_length + EPS) - tile_edge_length / base_edge_length) > EPS:
                        continue

                new_tile = get_tile_instance(base_tile, base_edge_idx, align_tile, align_neighbor_idx, align_mode=align_mode)
                if tiling_util.intersection_area(new_tile, base_tile) < EPS:
                    result_tiles.append(new_tile)
                    align_tags.append(align_tag)
    return result_tiles, align_tags

def find_candidate_tile_locations(num_rings, base_tile: Tile, align_tiles: list, integer_align=True):
    # the resulting tiles
    result_tiles = [base_tile]
    # the tiles in the last ring
    last_ring = [base_tile]

    for i in range(0, num_rings):
        print(f"computing ring_{i}")
        last_ring_num = len(last_ring)
        for last_ring_idx in range(last_ring_num):
            print(f"last ring_{last_ring_idx}")
            last_layer_tile = last_ring.pop(0)
            for align_tile in align_tiles:
                neighbour_tiles, _ = get_all_tiles(last_layer_tile, align_tile, integer_align = integer_align)
                for elem in neighbour_tiles:
                    if elem not in result_tiles:
                        result_tiles.append(elem)
                        last_ring.append(elem)

    return result_tiles

# This class represents a directed brick_layouts
class TileGraph:
    # Constructor
    def __init__(self, tile_type_count: int, tiles = None, one_hot = True, proto_tiles = None):
        # self.plotter = plotter
        # self.debugger = debugger
        # default dictionary to store brick_layouts
        self.tile_type_count = tile_type_count # number of different tile types
        self.tiles = None if tiles is None else tiles
        self.graph = defaultdict(list)
        self.one_hot = one_hot
        # remove duplicated tiles first
        if tiles is not None:
            self.tiles = self._remove_reductant(tiles)

        self.edges_features = defaultdict(list)
        self.adj_edges = []
        self.colli_edges = []
        self.align_start_index = 2
        self.max_align_length = 1e-10

        # produce the feature map if possible
        if proto_tiles is not None:
            self.unique_adj_features = self.edge_features_mapping(proto_tiles)
            self.total_feature_dim = self.align_start_index + len(self.unique_adj_features)

        # form the complete graph
        if tiles is not None:
            self._form__graph()
            self.max_area = max([t.area() for t in tiles])


    def edge_features_mapping(self, proto_tiles):

        unique_features = []

        # for each center tiles and align tiles:
        cnt_i = 0
        for center_tile in proto_tiles:
            cnt_i += 1
            cnt_j = 0
            for align_tile in proto_tiles:
                cnt_j += 1
                # get one ring neigbhour
                one_ring_tiles, _ = get_all_tiles(center_tile, align_tile, integer_align = True)
                one_ring_tiles = self._remove_reductant(one_ring_tiles)

                # calculating unique features
                for neigbor_tile in one_ring_tiles:
                    colli_area = tiling_util.intersection_area(center_tile, neigbor_tile)
                    align_length = tiling_util.polygon_align_length(center_tile, neigbor_tile)
                    edge_feature_back = self.to_edge_feature(colli_area, align_length)

                    if align_length > 1e-6 and colli_area < 1e-6:
                        tile_i_align, tile_j_align = tiling_util.polygon_align_type(center_tile, neigbor_tile)
                    else:
                        tile_i_align, tile_j_align = 0, 0
                    edge_feature = self.to_edge_feature_new(colli_area, align_length, tile_i_align, tile_j_align, center_tile.id, neigbor_tile.id)
                    reflected_feature = self.to_edge_feature_new(colli_area, align_length, tile_j_align, tile_i_align, neigbor_tile.id, center_tile.id)
                    ## assertion of correctness
                    assert edge_feature_back is None or \
                           edge_feature is None or \
                           (edge_feature[0] == edge_feature_back[0] and \
                            edge_feature[1] == edge_feature_back[1] and \
                            edge_feature[2] == tile_i_align and
                            edge_feature[3] == tile_j_align)

                    if edge_feature is not None:

                        # get direct (i -> j) or reflected features(j -> i)
                        current_edge_feature = edge_feature[self.align_start_index:]
                        current_reflected_feature = reflected_feature[self.align_start_index:]
                        # not in existing list:
                        for u in unique_features:
                            assert len(u) == len(current_edge_feature)

                        ## ensure both of alignment and reflection do not exist in the list
                        if not any( np.allclose(np.array(unique_feature),np.array(current_edge_feature), atol = ONE_HOT_EPS) for unique_feature in unique_features) \
                            and not any( np.allclose(np.array(unique_feature),np.array(current_reflected_feature), atol = ONE_HOT_EPS) for unique_feature in unique_features) \
                            and edge_feature[0] < EPS:
                            unique_features.append(current_edge_feature)
                            assert edge_feature[1] > 0


        # assertion for correctness
        for i, j in itertools.combinations(range(len(unique_features)), 2):
            assert (
                    (
                        not (unique_features[i][2] == unique_features[j][2] and
                             unique_features[i][3] == unique_features[j][3])
                        or
                        abs(unique_features[i][0] - unique_features[j][0]) + abs(
                            unique_features[i][1] - unique_features[j][1]) > EPS
                    )
                    or
                        not (unique_features[i][2] == unique_features[j][3] and
                             unique_features[i][3] == unique_features[j][2])
                        or
                        abs(unique_features[i][0] - unique_features[j][1]) + abs(
                            unique_features[i][1] - unique_features[j][0]) > EPS
            )

        print("one_hot_cnt: ", len(unique_features))
        return unique_features

    def _remove_reductant(self, tiles):
        # remove redundant tiles
        new_tiles = []
        for idx, elem in enumerate(tiles):
            assert elem.id <= self.tile_type_count
            if elem not in new_tiles:
                new_tiles.append(elem)
        print("#tiles after filtring repeat:", len(new_tiles))

        return new_tiles

    def _form__graph(self):
        print("start computing adjacency brick_layouts...")
        for i, j in itertools.combinations(range(len(self.tiles)), 2):
            colli_area = tiling_util.intersection_area(self.tiles[i], self.tiles[j])
            align_length = tiling_util.polygon_align_length(self.tiles[i], self.tiles[j])
            self.max_align_length = max(align_length, self.max_align_length)
            edge_feature_back = self.to_edge_feature(colli_area, align_length)

            if align_length > 1e-6 and colli_area < 1e-6:
                tile_i_align, tile_j_align = tiling_util.polygon_align_type(self.tiles[i], self.tiles[j])
            else:
                tile_i_align, tile_j_align = 0, 0

            # reflected feature
            edge_feature = self.to_edge_feature_new(colli_area, align_length, tile_i_align, tile_j_align, self.tiles[i].id,
                                                    self.tiles[j].id)
            reflected_feature = self.to_edge_feature_new(colli_area, align_length, tile_j_align, tile_i_align, self.tiles[j].id,
                                                         self.tiles[i].id)
            ## assertion of correctness
            assert edge_feature_back is None or \
                   edge_feature is None or \
                   (edge_feature[0] == edge_feature_back[0] and \
                    edge_feature[1] == edge_feature_back[1] and \
                    edge_feature[2] == tile_i_align and
                    edge_feature[3] == tile_j_align)

            if edge_feature is not None:
                if colli_area < 1e-6:
                    tile_i_align, tile_j_align = tiling_util.polygon_align_type(self.tiles[i], self.tiles[j])
                self._addEdge(i, j, edge_feature)
                self._addEdge(j, i, reflected_feature)

        print("Computing adjacency brick_layouts complete.")
        print(f"Node count: {len(self.tiles)}")
        print(f"Edge count: {len(self.adj_edges + self.colli_edges)}")

        print("Tiles:")
        for idx, tile in enumerate(self.tiles):
            print(f":{idx}->{tile.id}")

    @staticmethod
    def to_edge_feature(colli_area, align_length):
        if colli_area > 1e-6:  # contain intersection
            return [colli_area, 0]
        elif align_length > 1e-6:
            return [0, align_length]
        else:
            return None

    ## features:
    # return
    # index 0 : colli_area
    # index 1 : align length of elements
    # index 2 : align_portion of tile 1
    # index 3 : align_portion of tile 2
    # index 4 : tile idx of tile 1
    # index 5 : tile idx of tile 2
    @staticmethod
    def to_edge_feature_new(colli_area, align_length, align_portion_1, align_portion_2, tile_index_1, tile_index_2):
        if colli_area > 1e-6:  # contain intersection
            return [colli_area, 0, align_portion_1, align_portion_2, tile_index_1, tile_index_2]
        elif align_length > 1e-6:
            return [0, align_length, align_portion_1, align_portion_2, tile_index_1, tile_index_2]
        else:
            return None

    def feature_to_one_hot(self, input_feature):
        keep_feature = input_feature[:self.align_start_index]
        input_adj_feature = input_feature[self.align_start_index:]
        input_reflected_feature = [input_adj_feature[1], input_adj_feature[0], input_adj_feature[3], input_adj_feature[2]]
        adj_feature = [0] * len(self.unique_adj_features)
        if input_feature[0] > EPS:
            return list(keep_feature) + list(adj_feature)
        else:
            for idx, item in enumerate(self.unique_adj_features):
                if np.allclose(np.array(input_adj_feature),np.array(item), atol = ONE_HOT_EPS) \
                   or np.allclose(np.array(input_reflected_feature), np.array(item), atol = ONE_HOT_EPS):
                    adj_feature[idx] = 1
                    return list(keep_feature) + list(adj_feature)
            # return None for non existing method
            return None

    # function to add an edge to brick_layouts
    def _addEdge(self, u, v, input_feature):

        if self.one_hot:
            one_hot_feature = self.feature_to_one_hot(input_feature)

            # assert for feature
            if one_hot_feature is not None and one_hot_feature[1] > 0:
                for i in range(self.align_start_index, self.total_feature_dim):

                    ## assertion
                    if one_hot_feature[i] > EPS:
                        selected_features = self.unique_adj_features[i - self.align_start_index]
                        assert abs(selected_features[0] - input_feature[2]) + \
                            abs(selected_features[1] - input_feature[3]) < EPS \
                        or abs(selected_features[0] - input_feature[3]) + \
                           abs(selected_features[1] - input_feature[2]) < EPS
            if one_hot_feature is None:
                return

        self.graph[u].append(v)

        # add the feature to the dict of dict
        if u not in self.edges_features.keys():
            self.edges_features[u] = defaultdict(list)

        self.edges_features[u][v] = one_hot_feature
        if input_feature[0] > 0:
            # print("collide edges:", one_hot_feature)
            # print("collide input feature", input_feature)
            self.colli_edges.append((u,v))
        else:
            # print("align edges:", one_hot_feature)
            # print("align input feature", input_feature)
            assert input_feature[1] > 0
            self.adj_edges.append((u,v))

    def save_current_state(self, path):
        dic = {
            'tiles' : self.tiles,
            'graph' : self.graph,
            'edges_features': self.edges_features,
            'colli_edges':self.colli_edges,
            'adj_edges': self.adj_edges,
            'unique_adj_features' : self.unique_adj_features,
            'max_area' : self.max_area,
            'max_align_length' : self.max_align_length,
            'align_start_index' : self.align_start_index
        }
        pickle.dump(dic, open(path, "wb"))

    def load_graph_state(self, path):
        temp = pickle.load(open(path, "rb"))
        assert (
            'tiles' in temp.keys() and \
            'graph' in temp.keys() and \
            'edges_features' in temp.keys() and \
            'colli_edges' in temp.keys() and \
            'adj_edges' in temp.keys() and \
            'unique_adj_features' in temp.keys() and \
            'max_area' in temp.keys() and \
            'max_align_length' in temp.keys() and \
            'align_start_index' in temp.keys()
        )
        self.tiles = temp['tiles']
        self.graph = temp['graph']
        self.edges_features = temp['edges_features']
        self.colli_edges = temp['colli_edges']
        self.adj_edges = temp['adj_edges']
        self.unique_adj_features = temp['unique_adj_features']
        self.max_area = temp['max_area']
        self.align_start_index = temp['align_start_index']
        self.max_align_length = temp['max_align_length']
        self.total_feature_dim = self.align_start_index + len(self.unique_adj_features)

    def _get_graph_statistics(self):
        num_of_nodes = len(self.tiles)
        num_of_adj_edges = len(self.adj_edges) // 2
        num_of_collide_edges = len(self.colli_edges)  // 2
        return num_of_nodes, num_of_adj_edges, num_of_collide_edges

    def show_complete_super_graph(self, plotter, debugger, file_name):
        plotter.draw_contours(debugger.file_path(file_name),
                              [tile.get_plot_attribute("blue_trans") for tile in self.tiles])


if __name__ == '__main__':
    # Driver code

    # Create a brick_layouts given in
    # the above diagram
    g = TileGraph()
    g._addEdge(0, 1)
    g._addEdge(0, 2)
    g._addEdge(1, 2)
    g._addEdge(2, 0)
    g._addEdge(2, 3)
    g._addEdge(3, 3)

    print("Following is Breadth First Traversal"
          " (starting from vertex 2)")
    g.gen_target_tiles(2)
