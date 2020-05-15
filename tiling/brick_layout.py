import os
import numpy as np
from tiling.TileGraph import TileGraph
from shapely.ops import unary_union
import shapely
from collections import defaultdict
import util.data_util
import copy
from util import fabrication
from util.algo_util import interp
from tiling.color_table import color_map, tile_set_color_map
import random
import networkx as nx
import itertools
from util.tiling_util import polygon_align_length
import matplotlib.pyplot as plt
# episolon for area err
EPS = 1e-5
BUFFER_TILE_EPS = EPS * 1e-5
SIMPLIFIED_TILE_EPS = BUFFER_TILE_EPS * 1e3


class BrickLayout():
    def __init__(self, complete_graph: TileGraph, node_feature, collide_edge_index,
                 collide_edge_features, align_edge_index, align_edge_features, ground_truth, re_index,
                 target_polygon=None):
        self.complete_graph = complete_graph
        self.node_feature = node_feature
        self.collide_edge_index = collide_edge_index
        self.collide_edge_features = collide_edge_features
        self.align_edge_index = align_edge_index
        self.align_edge_features = align_edge_features

        ## assertion for brick_layout
        align_edge_index_list = align_edge_index.T.tolist()
        collide_edge_index_list = collide_edge_index.T.tolist()

        ## assertion
        # for f in align_edge_index_list:
        #     assert [f[1], f[0]] in align_edge_index_list
        # for f in collide_edge_index_list:
        #     assert [f[1], f[0]] in collide_edge_index_list

        # mapping from index of complete graph to index of super graph
        self.re_index = re_index
        # mapping from index of super graph to index of complete graph
        self.inverse_index = defaultdict(int)
        for k, v in self.re_index.items():
            self.inverse_index[v] = k

        self.predict = np.zeros(len(self.node_feature))
        self.predict_probs = []
        self.predict_order = []
        self.target_polygon = target_polygon

        ### save super poly
        self.super_contour_poly = None

    def __deepcopy__(self, memo):
        new_inst = type(self).__new__(self.__class__)  # skips calling __init__
        new_inst.complete_graph = self.complete_graph
        new_inst.node_feature = self.node_feature
        new_inst.collide_edge_index = self.collide_edge_index
        new_inst.collide_edge_features = self.collide_edge_features
        new_inst.align_edge_index = self.align_edge_index
        new_inst.align_edge_features = self.align_edge_features
        new_inst.re_index = self.re_index
        new_inst.inverse_index = self.inverse_index
        new_inst.ground_truth = self.ground_truth
        new_inst.predict = copy.deepcopy(self.predict)
        new_inst.predict_probs = copy.deepcopy(self.predict_probs)
        new_inst.super_contour_poly = self.super_contour_poly
        new_inst.predict_order = self.predict_order
        new_inst.target_polygon = self.target_polygon

        return new_inst

    def is_solved(self):
        return len(self.predict) != 0

    ############ ALL PLOTTING FUNCTIONS #################
    def show_candidate_tiles(self, plotter, debugger, file_name, style ="blue_trans"):
        tiles = self.complete_graph.tiles
        selected_indices = [k for k in self.re_index.keys()]
        selected_tiles = [tiles[s] for s in selected_indices]
        plotter.draw_contours(debugger.file_path(file_name),
                              [tile.get_plot_attribute(style) for tile in selected_tiles])

    def show_predict(self, plotter, debugger, file_name):
        tiles = self.predict
        plotter.draw_contours(debugger.file_path(file_name),
                              [self.complete_graph.tiles[self.inverse_index[i]].get_plot_attribute("blue_trans") for i in
                               range(len(tiles)) if
                               tiles[i] == 1])

    def show_super_contour(self, plotter, debugger, file_name):
        super_contour_poly = self.get_super_contour_poly()
        exteriors_contour_list, interiors_list = BrickLayout.get_polygon_plot_attr(super_contour_poly, show_line = True)
        plotter.draw_contours(debugger.file_path(file_name), exteriors_contour_list + interiors_list)

    ############ BACK UP #################



    def show_predict_with_transparent_color(self, plotter, file_name):
        # show prediction probs with color
        predict_probs = self.predict_probs

        min_fill_color, max_fill_color = np.array([255,255,255,50]), np.array([255,0,0,50])
        min_pen_color, max_pen_color = np.array([255, 255, 255, 0]), np.array([127, 127, 127, 255])

        super_contour_poly = self.get_super_contour_poly()
        exteriors_contour_list, interiors_list = BrickLayout.get_polygon_plot_attr(super_contour_poly, show_line = True)

        #### sort by prob
        sorted_indices = np.argsort(self.predict_probs)

        plotter.draw_contours(self.debugger.file_path(file_name),
                              exteriors_contour_list + interiors_list + [self.complete_graph.tiles[self.inverse_index[i]].get_plot_attribute(
                                  (
                                      tuple(interp(predict_probs[i], vec1 = min_fill_color, vec2 = max_fill_color)),
                                      tuple(interp(predict_probs[i], vec1=min_pen_color, vec2=max_pen_color))
                                  )
                              ) for i in
                               sorted_indices])

    def show_partial_result_with_tile_set_colors(self, env_name, select_tile_index, plotter, file_name):
        selected_tiles = [self.complete_graph.tiles[self.inverse_index[i]] for i in
                          range(len(select_tile_index)) if
                          select_tile_index[i] == 1]

        if self.target_polygon is not None:
            exteriors_contour_list, interior_list = BrickLayout.get_polygon_plot_attr(self.target_polygon)
        else:
            exteriors_contour_list, interior_list = [], []

        selected_tiles_attr = [tile.get_plot_attribute(tile_set_color_map[env_name][tile.id]) for idx, tile
                               in enumerate(selected_tiles)]

        super_polygon = self.get_super_contour_poly()
        exteriors_contour_list, interiors_list = BrickLayout.get_polygon_plot_attr(super_polygon,
                                                                        style=((0, 0, 0, 0), (0, 0, 0, 0)))

        plotter.draw_contours(self.debugger.file_path(file_name), exteriors_contour_list + interiors_list + exteriors_contour_list + interior_list + selected_tiles_attr)

    def show_predict_with_given_colors(self, color_result, select_tile_index, plotter, file_name):

        selected_tiles = [self.complete_graph.tiles[self.inverse_index[i]] for i in
                          range(len(select_tile_index)) if
                          select_tile_index[i] == 1]

        complete_graph_idx = [self.inverse_index[i] for i in
                          range(len(select_tile_index)) if
                          select_tile_index[i] == 1]

        selected_tiles_attr = [tile.get_plot_attribute( color_map[color_result[complete_graph_idx[idx]]] ) for idx, tile in enumerate(selected_tiles)]

        super_polygon = self.get_super_contour_poly()
        exteriors_contour_list, interiors_list = BrickLayout.get_polygon_plot_attr(super_polygon, style = ((0,0,0,0), (0,0,0,0)))

        plotter.draw_contours(self.debugger.file_path(file_name), exteriors_contour_list + interiors_list + selected_tiles_attr)

    def show_predict_with_single_color(self, plotter, file_name, color_style):

        if self.target_polygon is not None:
            exteriors_contour_list_out, interiors_list_out = self.get_target_shape_shadow(self.target_polygon)
        else:
            exteriors_contour_list_out, interiors_list_out = None

        selected_tiles = [self.complete_graph.tiles[self.inverse_index[i]] for i in
                               range(len(self.predict)) if
                               self.predict[i] == 1]

        selected_tiles_attr = [tile.get_plot_attribute(color_style) for idx, tile in enumerate(selected_tiles)]


        plotter.draw_contours(self.debugger.file_path(file_name), exteriors_contour_list_out+ interiors_list_out + selected_tiles_attr)

    def show_predict_with_background(self, plotter, file_name, color_style):
        # show input polygon
        if self.target_polygon is not None:
            exteriors_contour_list_out, interiors_list_out = self.get_target_shape_shadow(self.target_polygon)
        else:
            exteriors_contour_list_out, interiors_list_out = None

        # show cropped region
        super_contour_poly = self.get_super_contour_poly()
        exteriors_contour_list, interiors_list = BrickLayout.get_polygon_plot_attr(super_contour_poly, style='lightblue')

        selected_tiles = [self.complete_graph.tiles[self.inverse_index[i]] for i in
                               range(len(self.predict)) if
                               self.predict[i] == 1]

        selected_tiles_attr = [tile.get_plot_attribute(color_style) for idx, tile in enumerate(selected_tiles)]


        plotter.draw_contours(self.debugger.file_path(file_name), exteriors_contour_list_out+ interiors_list_out + exteriors_contour_list + interiors_list + selected_tiles_attr)

    def show_predict_with_tile_set_color_map(self, plotter, file_name, env_name):
        
        ### GET COLOR FROM THE COLOR MAP FILE
        if self.target_polygon is not None:
            exteriors_contour_list_out, interiors_list_out = self.get_target_shape_shadow(self.target_polygon)
        else:
            exteriors_contour_list_out, interiors_list_out = [], []

        selected_tiles = [self.complete_graph.tiles[self.inverse_index[i]] for i in
                               range(len(self.predict)) if
                               self.predict[i] == 1]


        selected_tiles_attr = [ tile.get_plot_attribute(
            tile_set_color_map[env_name][tile.id]
        ) for idx, tile in enumerate(selected_tiles)]

        plotter.draw_contours(self.debugger.file_path(file_name), exteriors_contour_list_out + interiors_list_out + selected_tiles_attr
                              )

    def show_predict_with_super_contour(self, plotter, file_name, has_line = False):

        super_contour_poly = self.get_super_contour_poly()
        exteriors_contour_list, interiors_list = BrickLayout.get_polygon_plot_attr(super_contour_poly, show_line = has_line)

        ###### add the contour first to
        predict_list = [self.complete_graph.tiles[self.inverse_index[i]].get_plot_attribute("pink_blue") for i in
                               range(len(self.predict)) if
                               self.predict[i] == 1]

        plotter.draw_contours(self.debugger.file_path(file_name), exteriors_contour_list + interiors_list + predict_list)

    def show_predict_with_input_shape(self, plotter, file_name):

        if self.target_polygon is not None:
            exteriors_contour_list, interiors_list = BrickLayout.get_polygon_plot_attr(self.target_polygon)
        else:
            exteriors_contour_list, interiors_list = [], []


        ###### add the contour first to
        predict_list = [self.complete_graph.tiles[self.inverse_index[i]].get_plot_attribute("pink_blue") for i in
                               range(len(self.predict)) if
                               self.predict[i] == 1]

        plotter.draw_contours(self.debugger.file_path(file_name), exteriors_contour_list + interiors_list + predict_list)

    def show_predict_with_4_color(self, plotter, file_name):

        selected_tiles = [self.complete_graph.tiles[self.inverse_index[i]] for i in
                               range(len(self.predict)) if
                               self.predict[i] == 1]


        #### Calculate color ####
        alignment_edges_from, alignment_edges_to = self.build_graph_from_prediction()
        color_result = self.solve_color(len(selected_tiles), alignment_edges_from, alignment_edges_to)
        # print(result)

        selected_tiles_attr = [ tile.get_plot_attribute(
            color_map[color_result[idx]]
        ) for idx, tile in enumerate(selected_tiles)]

        plotter.draw_contours(self.debugger.file_path(file_name), selected_tiles_attr)

    def show_predict_with_random_color(self, plotter, file_name):

        selected_tiles = [self.complete_graph.tiles[self.inverse_index[i]] for i in
                               range(len(self.predict)) if
                               self.predict[i] == 1]

        selected_tiles_attr = [ tile.get_plot_attribute(
            color_map[random.randint(0, len(color_map) - 1)]
        ) for tile in selected_tiles]

        plotter.draw_contours(self.debugger.file_path(file_name), selected_tiles_attr)


    def show_complete_graph(self, plotter, file_name):
        plotter.draw_contours(self.debugger.file_path(file_name),
                              [tile.get_plot_attribute("blue_trans") for tile in self.complete_graph.tiles])


    ##### OTHER UTILS

    def compute_tile_colors(self, predict):
        selected_tiles = [self.complete_graph.tiles[self.inverse_index[i]] for i in
                          range(len(predict)) if
                          self.predict[i] == 1]

        #### Calculate color ####
        alignment_edges_from, alignment_edges_to = self.build_graph_from_prediction()
        color_result = self.solve_color(len(selected_tiles), alignment_edges_from, alignment_edges_to)

        # convert to dict
        tile_to_color = {}
        idx = 0
        for i in range(len(self.predict)):
            if self.predict[i] == 1:
                tile_to_color[self.inverse_index[i]] = color_result[idx]
                idx = idx + 1

        return tile_to_color

    def get_target_shape_shadow(self, target_shape, style = None):
        if target_shape is not None:
            exteriors_contour_list_out, interiors_list_out = BrickLayout.get_polygon_plot_attr(target_shape, style = style)
        else:
            exteriors_contour_list_out, interiors_list_out = [], []
        return exteriors_contour_list_out, interiors_list_out

    def save_predict_as_objs(self, sav_dir, file_name):
        if not os.path.isdir(sav_dir):
            os.makedirs(sav_dir)
        boundary_points_list = [self.complete_graph.tiles[self.inverse_index[i]].get_plot_attribute() for i in range(len(self.predict)) if self.predict[i] == 1]
        for idx, tile_boundary in enumerate(boundary_points_list):
            fabrication.generate_2d_obj(os.path.join(sav_dir, f"{file_name}_{idx}.obj"), tile_boundary[1][:-1,:])

    def get_super_contour_poly(self):
        ### return super contour poly if already calculated
        if self.super_contour_poly is None:
            tiles = self.complete_graph.tiles
            selected_indices = [k for k in self.re_index.keys()]
            selected_tiles = [tiles[s].tile_poly.buffer(1e-6) for s in selected_indices]
            total_polygon = unary_union(selected_tiles).simplify(1e-6)
            self.super_contour_poly = total_polygon
            return total_polygon
        else:
            return self.super_contour_poly

    @staticmethod
    def get_polygon_plot_attr(input_polygon, show_line = False, style = None):
        # return color plot attribute given a shapely polygon
        # return poly attribute with color
        
        exteriors_contour_list = []
        interiors_list = []

        ### set the color for exterior and interiors

        if style is None:
            color = 'light_gray_border' if show_line else 'light_gray'
        else:
            color = (style[0], (127, 127, 127, 0)) if show_line else style

        background_color = 'white_border' if show_line else 'white'

        if isinstance(input_polygon, shapely.geometry.polygon.Polygon):
            exteriors_contour_list = [(color, np.array(list(input_polygon.exterior.coords)))]
            interiors_list = [(background_color, np.array(list(interior_poly.coords))) for interior_poly in
                              input_polygon.interiors]

        elif isinstance(input_polygon, shapely.geometry.multipolygon.MultiPolygon):
            exteriors_contour_list = [(color, np.array(list(polygon.exterior.coords))) for polygon in input_polygon]
            for each_polygon in input_polygon:
                one_interiors_list = [(background_color, np.array(list(interior_poly.coords))) for interior_poly in
                                      each_polygon.interiors]
                interiors_list = interiors_list +  one_interiors_list

        return exteriors_contour_list, interiors_list

    def get_selected_tiles(self):
        return [self.complete_graph.tiles[self.inverse_index[i]].tile_poly for i in range(len(self.predict)) if self.predict[i] == 1]

    def get_selected_tiles_union_polygon(self):
        return unary_union(self.get_selected_tiles())

    def detect_holes(self):
        ### DETECT HOLE
        selected_tiles = [self.complete_graph.tiles[self.inverse_index[i]].tile_poly.buffer(1e-7) for i in range(len(self.predict)) if self.predict[i] == 1]
        unioned_shape = unary_union(selected_tiles)
        if isinstance(unioned_shape, shapely.geometry.polygon.Polygon):
            if len(list(unioned_shape.interiors)) > 0:
                return True
        elif isinstance(unioned_shape, shapely.geometry.multipolygon.MultiPolygon):
            if any([len(list(unioned_shape[i].interiors)) > 0 for i in range(len(unioned_shape))]):
                return True

        return False

    def get_data_as_torch_tensor(self, device):
        x, adj_edge_index, adj_edge_features, collide_edge_index, collide_edge_features, gt = \
            util.data_util.to_torch_tensor(device, self.node_feature, self.align_edge_index, self.align_edge_features, self.collide_edge_index, self.collide_edge_features, self.ground_truth)

        return x, adj_edge_index, adj_edge_features, collide_edge_index, collide_edge_features, gt

    def compute_sub_layout(self, predict):
        assert len(self.node_feature) == len(predict.labelled_nodes) + len(predict.unlabelled_nodes)
        sorted_dict = sorted(predict.unlabelled_nodes.items(), key=lambda x: x[0])
        predict.unlabelled_nodes.clear()
        predict.unlabelled_nodes.update(sorted_dict)
        # compute index mapping from original index to current index
        node_re_index = {}
        for idx, key in enumerate(predict.unlabelled_nodes):
            node_re_index[key] = idx

        complete_graph = self.complete_graph
        node_feature = self.node_feature[list(predict.unlabelled_nodes.keys())]

        # index
        collide_edge_index = [ [node_re_index[self.collide_edge_index[0, i]], node_re_index[self.collide_edge_index[1, i]]] for i in range(self.collide_edge_index.shape[1]) if self.collide_edge_index[0, i] in predict.unlabelled_nodes and self.collide_edge_index[1, i] in predict.unlabelled_nodes ] \
            if self.collide_edge_index.shape[0] > 0 else np.array([])
        collide_edge_index = np.array(collide_edge_index).T

        align_edge_index = [ [node_re_index[self.align_edge_index[0, i]], node_re_index[self.align_edge_index[1, i]]] for i in range(self.align_edge_index.shape[1]) if self.align_edge_index[0, i] in predict.unlabelled_nodes and self.align_edge_index[1, i] in predict.unlabelled_nodes ] \
            if self.align_edge_index.shape[0] > 0 else np.array([])
        align_edge_index = np.array(align_edge_index).T

        # feature
        collide_edge_features = np.array([ self.collide_edge_features[i, :] for i in range(self.collide_edge_index.shape[1]) if self.collide_edge_index[0, i] in predict.unlabelled_nodes and self.collide_edge_index[1, i] in predict.unlabelled_nodes ]) \
            if self.collide_edge_features.shape[0] > 0 else np.array([])
        align_edge_features = np.array([ self.align_edge_features[i, :] for i in range(self.align_edge_index.shape[1]) if self.align_edge_index[0, i] in predict.unlabelled_nodes and self.align_edge_index[1, i] in predict.unlabelled_nodes ]) \
            if self.align_edge_features.shape[0] > 0 else np.array([])


        # compute index mapping from current index to original index
        node_inverse_index = {}
        for idx, key in enumerate(predict.unlabelled_nodes):
            node_inverse_index[idx] = key

        fixed_re_index = {}
        for i in range(node_feature.shape[0]):
            fixed_re_index[self.inverse_index[node_inverse_index[i]]] = i

        return BrickLayout(complete_graph, node_feature, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features, None, fixed_re_index, target_polygon=self.target_polygon), node_inverse_index

    def build_graph_from_prediction(self):

        selected_nodes = [ i for i in range(len(self.predict)) if self.predict[i] == 1]

        alignment_edges_from = []
        alignment_edges_to = []
        for idx_i, idx_j in itertools.combinations(range(len(selected_nodes)), 2):
            tile_i = self.complete_graph.tiles[self.inverse_index[selected_nodes[idx_i]]]
            tile_j = self.complete_graph.tiles[self.inverse_index[selected_nodes[idx_j]]]
            if polygon_align_length(tile_i, tile_j) > EPS:
                alignment_edges_from.append(idx_i)
                alignment_edges_to.append(idx_j)

        return alignment_edges_from, alignment_edges_to

    def solve_color(self, num_of_nodes, adj_from, adj_to):
        from minizinc import Solver, Model, Instance

        solver = Solver.lookup('coin-bc')
        model = Model()
        model.add_file('./tiling/coloring.mzn')

        instance = Instance(solver, model)
        instance['nums_node'] = num_of_nodes
        instance['from_adjacent'] = adj_from
        instance['to_adjacent'] = adj_to
        instance['nums_edge_adjacent'] = len(adj_to)

        # print(f"num_of_nodes : {num_of_nodes}")
        # print(f"adj_from : {adj_from}")
        # print(f"adj_to : {adj_to}")
        # print(f"len : {len(adj_to)}")

        result = instance.solve()
        # getting result of nodes if exist
        assert (result.solution)

        color = np.argmax(result['node_color'], axis = 1)


        return color

    @staticmethod
    def assert_equal_layout(brick_layout_1, brick_layout_2):
        assert np.array_equal(brick_layout_1.node_feature, brick_layout_2.node_feature)
        assert np.array_equal(brick_layout_1.collide_edge_index, brick_layout_2.collide_edge_index)
        assert np.array_equal(brick_layout_1.collide_edge_features, brick_layout_2.collide_edge_features)
        assert np.array_equal(brick_layout_1.align_edge_index, brick_layout_2.align_edge_index)
        assert np.array_equal(brick_layout_1.align_edge_features, brick_layout_2.align_edge_features)

        # mapping from index of complete graph to index of super graph
        for key in brick_layout_1.re_index.keys():
            assert brick_layout_1.re_index[key] == brick_layout_2.re_index[key]
        for key in brick_layout_2.re_index.keys():
            assert brick_layout_2.re_index[key] == brick_layout_1.re_index[key]

        ### assert others
        assert np.array_equal(brick_layout_1.predict, brick_layout_2.predict)
        assert np.array_equal(brick_layout_1.predict_probs, brick_layout_2.predict_probs)
        assert brick_layout_1.predict_order == brick_layout_2.predict_order
        assert brick_layout_1.target_polygon == brick_layout_2.target_polygon


if __name__ == "__main__":
    pass