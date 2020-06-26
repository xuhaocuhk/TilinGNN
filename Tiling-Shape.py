from util.shape_processor import getSVGShapeAsNp, load_polygons
from tiling.tile_factory import crop_multiple_layouts_from_contour
import pickle
import numpy as np
from util.debugger import MyDebugger
import os
import torch
from interfaces.qt_plot import Plotter
from solver.ml_solver.ml_solver import ML_Solver
from inputs.env import Environment
from inputs import config
import torch.multiprocessing as mp
from util.data_util import load_bricklayout, write_bricklayout
from shapely.geometry import Polygon
from tiling.brick_layout import BrickLayout
from graph_networks.networks.TilinGNN import TilinGNN

EPS = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init():
    MyDebugger.pre_fix = os.path.join(MyDebugger.pre_fix, "debug")
    debugger = MyDebugger(f"region_tiling", fix_rand_seed=config.rand_seed,
                          save_print_to_file=False)
    plotter = Plotter()
    return debugger, plotter

def tiling_a_region():
    debugger, plotter = init()
    environment = config.environment # must be 30-60-90
    environment.load_complete_graph(config.complete_graph_size)
    environment.complete_graph.show_complete_super_graph(plotter, debugger, "complete_graph.png")

    network = TilinGNN(adj_edge_features_dim=environment.complete_graph.total_feature_dim,
                       network_depth=config.network_depth, network_width=config.network_width).to(device)

    solver = ML_Solver(debugger, device, environment.complete_graph, network, num_prob_maps= 1)
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

    for idx, solved_layout in enumerate(solutions):
        has_hole = result_brick_layout.detect_holes()
        result_brick_layout, score = solved_layout

        ### hacking for probs
        result_brick_layout.predict_probs = result_brick_layout.predict

        write_bricklayout(folder_path = debugger.file_path('./'),
                          file_name = f'{score}_{idx}_data.pkl', brick_layout = result_brick_layout,
                          with_features = False)

        reloaded_layout = load_bricklayout(file_path = debugger.file_path(f'{score}_{idx}_data.pkl'),
                                           complete_graph = environment.complete_graph)

        ## asserting correctness
        BrickLayout.assert_equal_layout(result_brick_layout, reloaded_layout)

        result_brick_layout.show_predict(plotter, debugger, f'{score}_{idx}_predict.png', do_show_super_contour=True, do_show_tiling_region=True)
        result_brick_layout.show_super_contour(plotter, debugger, f'{score}_{idx}_super_contour.png')
        result_brick_layout.show_adjacency_graph(debugger.file_path(f'{score}_{idx}_vis_graph.png'))
        result_brick_layout.show_predict_prob(plotter, debugger, f'{score}_{idx}_prob.png')


if __name__ == '__main__':
    tiling_a_region()
    print("done")
