from util.shape_processor import getSVGShapeAsNp, load_polygons
from tiling.TileFactory import crop_multiple_layouts_from_contour
from interfaces.graph_visualization import visual_brick_layout_graph
import pickle
import numpy as np
from tiling.TileFactory import run_one_layout
from util.debugger import MyDebugger
import os
import torch
from interfaces.qt_plot import Plotter
from solver.ml_solver.ml_solver import ML_Solver
from inputs.env import Environment
from inputs import config
import torch.multiprocessing as mp
from util.data_util import load_bricklayout
from shapely.geometry import Polygon
from tiling.brick_layout import BrickLayout

EPS = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tiling_a_region():
    debugger, plotter = init()
    environment = config.environment # must be 30-60-90
    environment.load_complete_graph(config.complete_graph_size)

    solver = ML_Solver(debugger, device, environment.complete_graph, None, num_prob_maps= 1)
    solver.load_saved_network(config.network_path)

    ##### select a silhouette as a tiling region
    silhouette_path = r'./silhouette/bunny.txt'
    silhouette_file_name = os.path.basename(silhouette_path)
    exterior_contour, interior_contours = load_polygons(silhouette_path)

    ##### plot the select tiling region
    base_polygon = Polygon(exterior_contour, holes=interior_contours)
    exteriors_contour_list, interiors_list = BrickLayout.get_polygon_plot_attr(base_polygon, show_line=True)
    plotter.draw_contours(debugger.file_path(f'tiling_region_{silhouette_file_name[:-4]}.png'),
        exteriors_contour_list + interiors_list)

    ##### get candidate tile placements inside the tiling region by cropping
    cropped_brick_layouts = crop_multiple_layouts_from_contour(exterior_contour, interior_contours, environment.complete_graph,
                                                              start_angle=0, end_angle=30, num_of_angle=1,
                                                              movement_delta_ratio=[0, 0.5], margin_padding_ratios=[0.5])

    ##### show the cropped tile placements
    for idx, (brick_layout, coverage) in enumerate(cropped_brick_layouts):
        brick_layout.show_candidate_tiles(plotter, debugger, f"candi_tiles_{idx}_{coverage}.png")

    # tiling solving
    solutions = []
    for idx, (result_brick_layout, coverage) in enumerate(cropped_brick_layouts):
        result_brick_layout, score = solver.solve(result_brick_layout)
        solutions.append((result_brick_layout, score))

    for solved_layout in solutions:
        has_hole = result_brick_layout.detect_holes()
        solved_layout[0]
        # TODO:
        #  1. save brick layout here
        #  2. make sure the following things are stored and implemented in bricklayout: 1) the input tiling region 2) the tiling order in algorithm 1 3) the probabilities 4) the graph visualization function
        result_brick_layout.show_predict(plotter, debugger, debugger.file_path(f'{score}_{idx}_{trial_idx}_predict.png'))
        result_brick_layout.show_super_contour(plotter, debugger.file_path(f'{score}_{idx}_{trial_idx}_super_contour.png'))
        visual_brick_layout_graph(result_brick_layout, os.path.join(save_path, f'{score}_{idx}_{trial_idx}_vis_graph.png'))




def init():
    MyDebugger.pre_fix = os.path.join(MyDebugger.pre_fix, "debug")
    debugger = MyDebugger(f"region_tiling", fix_rand_seed=config.rand_seed,
                          save_print_to_file=False)
    plotter = Plotter()
    return debugger, plotter


def load_and_show_train_data():
    debugger, plotter = init()
    env_name = "30-60-90"
    symmetry_tiles, complete_graph_size, number_of_data, data_folder = config.env_attribute_dict[env_name]
    env_location = os.path.join('.', 'data', "30-60-90")
    environment = Environment(env_location, symmetry_tiles=symmetry_tiles)
    environment.load_complete_graph(9)

    for i in range(0,4):
        brick_layout = load_bricklayout(f"/home/edwardhui/data/figures/overview/version4/raw/data_{i}.pkl", debugger, environment.complete_graph)
        predict = pickle.load(open(f"/home/edwardhui/data/figures/overview/version4/raw/{i}_selected.pkl", 'rb')) # the node selection
        predict_probs = np.load(f"/home/edwardhui/data/figures/overview/version4/raw/{i}_predict_prob.npy") # the network output prob
        brick_layout.predict_probs = np.ones(brick_layout.node_feature.shape[0])
        brick_layout.predict_probs = predict_probs

        visual_brick_layout_graph(brick_layout, debugger.file_path(f"vis_graph_{i}.png"), is_vis_prob = False, node_size = 30, edge_width = 0.4, xlim = (-3, 3), ylim = (-3, 3))
        brick_layout.show_candidate_tiles(plotter, f"super_graph_{i}.png")
        brick_layout.show_super_contour(plotter, f"super_contour_{i}.png")

def load_and_show_one_data():
    debugger, plotter = init()
    env_name = "30-60-90"
    symmetry_tiles, complete_graph_size, number_of_data, data_folder = config.env_attribute_dict[env_name]
    env_location = os.path.join('.', 'data', "30-60-90")
    environment = Environment(env_location, symmetry_tiles=symmetry_tiles)
    environment.load_complete_graph(9)


    brick_layout = load_bricklayout(f"/home/edwardhui/data/figures/overview/version4/raw/data_{i}.pkl", debugger, environment.complete_graph)
    predict = pickle.load(open(f"/home/edwardhui/data/figures/overview/version4/raw/{i}_selected.pkl", 'rb')) # the node selection
    predict_probs = np.load(f"/home/edwardhui/data/figures/overview/version4/raw/{i}_predict_prob.npy") # the network output prob
    brick_layout.predict_probs = np.ones(brick_layout.node_feature.shape[0])
    brick_layout.predict_probs = predict_probs

    visual_brick_layout_graph(brick_layout, debugger.file_path(f"vis_graph_{i}.png"), is_vis_prob = False, node_size = 30, edge_width = 0.4, xlim = (-3, 3), ylim = (-3, 3))
    brick_layout.show_candidate_tiles(plotter, f"super_graph_{i}.png")
    brick_layout.show_super_contour(plotter, f"super_contour_{i}.png")



if __name__ == '__main__':
    tiling_a_region()
    print("done")
