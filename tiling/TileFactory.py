import numpy as np
import math
from util.shape_processor import getSVGShapeAsNp, load_polygons
import torch.multiprocessing as mp
import torch
import random
from tiling.TileGraph import TileGraph
from tiling.brick_layout import BrickLayout
from shapely.ops import unary_union
from util.algo_util import contain
from shapely.geometry import Polygon
from inputs import config
import tiling.brick_layout
import shapely.affinity
from copy import deepcopy
import os
import pickle
import itertools
import glob
import traceback
import time
from util.data_util import write_bricklayout, generate_brick_layout_data, load_bricklayout

EPS = 1e-5



def gen_one_train_data(plotter, graph: TileGraph, low, high):
    target = graph.gen_target_tiles(low=low, high=high)

    # getting sub-graph from data
    super_tiles, filtered_collided_edges, filtered_adj_edges = compute_super_graph(graph, target)

    # get the data for brick_layout
    node_feature, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features, target, re_index = \
        generate_brick_layout_data(graph, super_tiles, filtered_collided_edges, filtered_adj_edges, target)

    return node_feature, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features, target, re_index


def compute_super_graph(graph: TileGraph, generated_target):
    assert len(generated_target) > 0
    # select all possible tiles from super brick_layouts
    selected_tiles = [graph.tiles[s].tile_poly.buffer(1e-6) for s in generated_target]
    total_polygon = unary_union(selected_tiles)

    # getting placement inside polygon
    tiles_super_set, filtered_collided_edges, filtered_adj_edges = get_all_placement_in_polygon(graph, total_polygon)

    return tiles_super_set, filtered_collided_edges, filtered_adj_edges

def get_all_placement_in_polygon(graph: TileGraph, polygon : Polygon):
    # get all tile placements in the supergraph
    tiles_super_set = [i for i in range(len(graph.tiles)) if contain(polygon, graph.tiles[i].tile_poly)]

    # fliter all edges
    filtered_collided_edges = [edge for edge in graph.colli_edges if
                               edge[0] in tiles_super_set and edge[1] in tiles_super_set]
    filtered_adj_edges = [edge for edge in graph.adj_edges if
                               edge[0] in tiles_super_set and edge[1] in tiles_super_set]

    return tiles_super_set, filtered_collided_edges, filtered_adj_edges

def create_brick_layout_from_polygon(graph: TileGraph, polygon : Polygon):

    # get filter graph
    tiles_super_set, filtered_collided_edges, filtered_adj_edges = get_all_placement_in_polygon(graph, polygon)

    # produce data needed
    node_feature, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features, target, re_index = \
        generate_brick_layout_data(graph, tiles_super_set, filtered_collided_edges, filtered_adj_edges, [])

    return node_feature, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features, target, re_index

def get_graph_bound(graph : TileGraph):
    tiles = [ np.array(t.tile_poly.exterior.coords) for t in graph.tiles]

    # getting the bound
    x_min = np.min([np.min(tile[:, 0]) for tile in tiles])
    x_max = np.max([np.max(tile[:, 0]) for tile in tiles])
    y_min = np.min([np.min(tile[:, 1]) for tile in tiles])
    y_max = np.max([np.max(tile[:, 1])for tile in tiles])
    return x_min, x_max, y_min, y_max

def generate_random_inputs(graph: TileGraph, max_vertices : float = 10, low = 0.2, high = 0.7,
                           plotter = None, debugger = None, plot_shape = False):

    # try until can create
    while True:
        try:
            x_min, x_max, y_min, y_max = get_graph_bound(graph)

            base_radius = min(x_max - x_min, y_max - y_min) / 2
            radius_random = random.uniform(low, high)
            irregularity = random.random()
            spikeyness = random.random()
            number_of_vertices = random.randint(3, max_vertices)

            # generation of the random polygon
            vertices = generatePolygon((x_max + x_min) / 2, (y_max + y_min) / 2, base_radius * radius_random, irregularity, spikeyness, number_of_vertices)
            polygon = Polygon(vertices)

            if plot_shape:
                assert plotter is not None
                assert debugger is not None
                plotter.draw_contours(debugger.file_path('generated_shape.png'), [('green', np.array(vertices))])

            node_feature, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features, target, re_index = \
                create_brick_layout_from_polygon(graph, polygon)

            assert len(node_feature.shape) > 0

            # skip
            if len(collide_edge_index) == 0 or len(align_edge_index) == 0:
                continue

            target = None

            return node_feature, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features, target, re_index
        except Exception as e:
            # print(traceback.format_exc())
            continue


def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts ):
    '''Start with the centre of the polygon at ctrX, ctrY,
    then creates the polygon by sampling points on a circle around the centre.
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order.
    '''

    irregularity = clip( irregularity, 0,1 ) * 2*math.pi / numVerts
    spikeyness = clip( spikeyness, 0,1 ) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2*math.pi / numVerts) - irregularity
    upper = (2*math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = random.uniform(lower, upper)
        angleSteps.append( tmp )
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*math.pi)
    for i in range(numVerts) :
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2*math.pi)
    for i in range(numVerts) :
        r_i = clip( random.gauss(aveRadius, spikeyness), 0, 2*aveRadius )
        x = ctrX + r_i*math.cos(angle)
        y = ctrY + r_i*math.sin(angle)
        points.append( (x,y) )

        angle = angle + angleSteps[i]
    return points

def clip(x, min, max) :
    if( min > max ) :  return x
    elif( x < min ) :  return min
    elif( x > max ) :  return max
    else :             return x

def crop_multiple_layouts_from_contour(exterior_contour, interior_contours, complete_graph, start_angle = 0.0, end_angle = 60.0, num_of_angle = 1,
                                       movement_delta_ratio = [0], margin_padding_ratios = [0.2]):
    print("cropping from super set...")
    result_brick_tuples = []

    tile_movement_delta = get_tile_movement_delta(complete_graph, movement_delta_ratio)

    for margin_padding_ratio in margin_padding_ratios:
        for rotate_angle in np.linspace(start_angle, end_angle, num_of_angle):
            for x_delta, y_delta in itertools.product(tile_movement_delta, tile_movement_delta):
                base_diameter, target_polygon = shape_transform(complete_graph,
                                                                exterior_contour,
                                                                interior_contours,
                                                                margin_padding_ratio,
                                                                rotate_angle,
                                                                x_delta,
                                                                y_delta)
                try:
                    node_feature, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features, target, re_index = \
                        create_brick_layout_from_polygon(complete_graph, target_polygon)
                except:
                    print(traceback.format_exc())
                    continue

                ## if no tile is contained in graph
                if len(node_feature) == 0:
                    continue

                brick_layout = tiling.brick_layout.BrickLayout(complete_graph, node_feature,
                                                               collide_edge_index, collide_edge_features, align_edge_index,
                                                               align_edge_features, target, re_index, target_polygon = target_polygon)
                # initialize selection probability
                brick_layout.predict_probs = [0.5 for i in range(brick_layout.node_feature.shape[0])]

                # cropping score calculation
                coverage_score = brick_layout.get_super_contour_poly().area / target_polygon.area
                result_brick_tuples.append((brick_layout, coverage_score))
                print(f"cropping done at {margin_padding_ratio} {rotate_angle} {x_delta} {y_delta}")

    return result_brick_tuples


def get_tile_movement_delta(complete_graph, movement_delta_ratio):
    ## calculate the x-delta and y-delta
    tile_bound = complete_graph.tiles[0].tile_poly.bounds
    tile_delta = min((tile_bound[2] - tile_bound[0]), (tile_bound[3] - tile_bound[1]))
    tile_movement_delta = np.array(movement_delta_ratio) * tile_delta
    return tile_movement_delta


def solve_silhouette_dir(silhouette_dir, comeplete_graph, solver, debugger, plotter, workers = 3):

    files_path = glob.glob(os.path.join(silhouette_dir, '*.txt'))
    scores = []
    mp.set_start_method('spawn')
    solver.network.share_memory()
    pool = mp.Pool(workers)

    # start from a file if needed
    if config.start_file_name is not None:
        file_name_base_name = [ os.path.basename(file_path) for file_path in files_path]
        assert config.start_file_name in file_name_base_name
        start_index = file_name_base_name.index(config.start_file_name)
        files_path = files_path[start_index:]

    for file_path in files_path:
        file_scores = []
        file_name = os.path.basename(file_path)
        contour = getSVGShapeAsNp(file_path)

        exterior_contour, interior_contours = load_polygons(file_path)

        ## assertion for correctness
        # assert np.all(contour == exterior_contour)

        # create saving path
        save_path = debugger.file_path(file_name)
        os.mkdir(save_path)
        tree_search_layout_base_dir = os.path.join(save_path, 'tree_search')
        if not os.path.isdir(tree_search_layout_base_dir):
            os.mkdir(tree_search_layout_base_dir)

        try:
            result_brick_tuples = crop_multiple_layouts_from_contour(exterior_contour, interior_contours, comeplete_graph,
                                                                      start_angle = config.start_angle,
                                                                      end_angle = config.end_angle,
                                                                      num_of_angle = config.num_of_angle,
                                                                      movement_delta_ratio = config.movement_delta_ratio,
                                                                      margin_padding_ratios = config.margin_padding_ratios,
                                                                      debugger = debugger,
                                                                      plotter = plotter,
                                                                      plot_graph = config.plot_target,
                                                                      fig_name = file_name,
                                                                      save_path = save_path)
        except:
            print(traceback.format_exc())
            continue

        if config.sort_by_coverage:
            result_brick_tuples = sorted(result_brick_tuples, key = lambda key : key[1], reverse = True)
            result_brick_tuples = result_brick_tuples[:config.sort_keep_num]


        for idx, result_brick_tuple in enumerate(result_brick_tuples):
            result_brick_layout, _, target_polygon = result_brick_tuple
            try:
                trial_scores = run_one_layout(idx, interior_contours, pool, result_brick_layout, save_path, solver,
                                              tree_search_layout_base_dir, plotter, target_polygon)
                file_scores.append(max(trial_scores, default = 0))
            except:
                print(traceback.format_exc())



        text_file = open(os.path.join(save_path, f"average_score:{np.mean(file_scores)}.txt"), "w")
        text_file.close()

        scores = scores + file_scores

    return np.mean(scores)


def run_one_layout(idx, interior_contours, pool, input_brick_layout, save_path, solver, tree_search_layout_base_dir, plotter, target_polygon):
    # preparation for parms
    idx_parms = [idx] * config.trial_times
    interior_contours_parms = [interior_contours] * config.trial_times
    result_brick_layout_parms = [deepcopy(input_brick_layout) for i in range(config.trial_times)]
    save_path_parms = [save_path] * config.trial_times
    solver_parms = [solver] * config.trial_times
    tree_search_layout_base_dir_parms = [tree_search_layout_base_dir] * config.trial_times
    trial_idx_parms = range(config.trial_times)
    plotter_parms = [None] * config.trial_times
    start_time = time.time()
    all_parms = zip(idx_parms, interior_contours_parms, result_brick_layout_parms,
                    save_path_parms, solver_parms, tree_search_layout_base_dir_parms, trial_idx_parms, plotter_parms)
    trial_tuples = pool.map(solve_one_trial, all_parms)
    trial_scores, _, _, _, _, _ = zip(*trial_tuples)
    trial_total_times = time.time() - start_time

    trial_tuples = sorted(trial_tuples, key = lambda tup : tup[0], reverse = True)
    for score, time_used, trial_idx, result_brick_layout, predict_cnt in trial_tuples:
        if score != 0:
            has_hole = result_brick_layout.detect_holes()

            if config.save_layout and (not config.detect_holes or len(interior_contours) > 0 or not has_hole):
                file_prefix = f'{score}_{idx}_{trial_idx}'
                save_all_layout_info(file_prefix, result_brick_layout, save_path, target_polygon)
                save_bricklayout_stat(result_brick_layout,
                                      os.path.join(save_path, f'{score}_{idx}_{trial_idx}_stat.txt'),
                                      predict_cnt)
            if config.save_tiling_pictures and (not config.detect_holes or len(interior_contours) > 0 or not has_hole):
                result_brick_layout.show_predict(plotter, os.path.join(save_path,
                                                                       f'{score}_{idx}_{trial_idx}_predict.png'))
                result_brick_layout.show_predict_with_super_contour(plotter, os.path.join(save_path,
                                                                       f'{score}_{idx}_{trial_idx}_predict_with_super.png'))
                result_brick_layout.show_candidate_tiles(plotter, os.path.join(save_path,
                                                                       f'{score}_{idx}_{trial_idx}_super_graph.png'))
                result_brick_layout.show_super_contour(plotter, os.path.join(save_path,
                                                                       f'{score}_{idx}_{trial_idx}_super_contour.png'))
                if target_polygon is not None:
                    result_brick_layout.show_predict_with_input_shape(target_polygon,
                                                                      plotter,
                                                                      os.path.join(save_path, f'{score}_{idx}_{trial_idx}_predict_with_target.png'))


    f = open(os.path.join(save_path, f"timeing_{idx}_{trial_total_times}.txt"), "w+")
    f.close()

    return list(trial_scores)


def save_all_layout_info(file_prefix, result_brick_layout : BrickLayout, save_path, with_features = False):
    write_bricklayout(save_path, f'{file_prefix}_data.pkl', result_brick_layout, with_features = with_features)

    #### assertion for correctness
    reloaded_bricklayout = load_bricklayout(
        os.path.join(save_path, f'{file_prefix}_data.pkl'),
        debugger=result_brick_layout.debugger,
        complete_graph=result_brick_layout.complete_graph)
    BrickLayout.assert_equal_layout(reloaded_bricklayout, result_brick_layout)


'''
# single thread version
def run_one_layout(idx, interior_contours, pool, input_brick_layout, save_path, solver, tree_search_layout_base_dir, plotter, target_polygon):
    # preparation for parms
    idx_parms = [idx] * config.trial_times
    interior_contours_parms = [interior_contours] * config.trial_times
    result_brick_layout_parms = [deepcopy(input_brick_layout) for i in range(config.trial_times)]
    save_path_parms = [save_path] * config.trial_times
    solver_parms = [solver] * config.trial_times
    tree_search_layout_base_dir_parms = [tree_search_layout_base_dir] * config.trial_times
    trial_idx_parms = range(config.trial_times)
    plotter_parms = [plotter] * config.trial_times
    start_time = time.time()
    all_parms = zip(idx_parms, interior_contours_parms, result_brick_layout_parms,
                    save_path_parms, solver_parms, tree_search_layout_base_dir_parms, trial_idx_parms, plotter_parms)
    trial_tuples = list(map(solve_one_trial, all_parms))
    trial_scores, _, _, _, _, _ = zip(*trial_tuples)
    trial_total_times = time.time() - start_time

    trial_tuples = sorted(trial_tuples, key = lambda tup : tup[0], reverse = True)
    for score, time_used, trial_idx, result_brick_layout, predict_cnt in trial_tuples:
        if score != 0:
            has_hole = result_brick_layout.detect_holes()

            if config.save_layout and (not config.detect_holes or len(interior_contours) > 0 or not has_hole):
                # write_bricklayout(save_path, f'{score}_{idx}_{trial_idx}_data.pkl', result_brick_layout)
                write_re_index(result_brick_layout.re_index, os.path.join(save_path, f'{score}_{idx}_{trial_idx}_data.pkl'))
                np.save(os.path.join(save_path, f'{score}_{idx}_{trial_idx}_predict.npy'), np.array(result_brick_layout.predict))
                save_bricklayout_stat(result_brick_layout,
                                      os.path.join(save_path, f'{score}_{idx}_{trial_idx}_stat.txt'),
                                      predict_cnt)
            if config.save_tiling_pictures and (not config.detect_holes or len(interior_contours) > 0 or not has_hole):
                result_brick_layout.show_predict(plotter, os.path.join(save_path,
                                                                       f'{score}_{idx}_{trial_idx}_predict.png'))
                result_brick_layout.show_predict_with_super_contour(plotter, os.path.join(save_path,
                                                                       f'{score}_{idx}_{trial_idx}_predict_with_super.png'))
                result_brick_layout.show_super_graph(plotter, os.path.join(save_path,
                                                                       f'{score}_{idx}_{trial_idx}_super_graph.png'))
                result_brick_layout.show_super_contour(plotter, os.path.join(save_path,
                                                                       f'{score}_{idx}_{trial_idx}_super_contour.png'))
                if target_polygon is not None:
                    result_brick_layout.show_predict_with_input_shape(target_polygon,
                                                                      plotter,
                                                                      os.path.join(save_path, f'{score}_{idx}_{trial_idx}_predict_with_target.png'))

    f = open(os.path.join(save_path, f"timeing_{idx}_{trial_total_times}.txt"), "w+")
    f.close()

    return list(trial_scores)


'''

def solve_one_trial(parms):
    # save the intermediate result if needed
    idx, interior_contours, input_brick_layout, save_path, solver, tree_search_layout_base_dir, trial_idx, plotter = parms
    # hacking creating new plotter
    print(f"Solving index {idx} + trial index {trial_idx}.......")
    tree_search_layout_dir = None
    if config.output_tree_search_layout:
        tree_search_layout_dir = os.path.join(tree_search_layout_base_dir, f'{idx}_{trial_idx}')
        os.mkdir(tree_search_layout_dir)

    start_time = time.time()
    result_brick_layout = input_brick_layout
    score = 0
    predict_cnt = 0
    try:
        result_brick_layout, score, predict_cnt = solver.solve(input_brick_layout,
                                                               time_limit=config.evaluation_search_time_limit,
                                                               intermediate_results_dir=tree_search_layout_dir)
    except:
        print(traceback.format_exc())

    time_used = time.time() - start_time
    print(f"Done Solving index {idx} + trial index {trial_idx} + score : {score}.......")

    return score, time_used, trial_idx, result_brick_layout, predict_cnt

def save_bricklayout_stat(brick_layout, save_path, predict_cnt):
    f = open(save_path, "w+")

    node_cnt = brick_layout.node_feature.shape[0]
    collide_edges_cnt = brick_layout.collide_edge_features.shape[0] // 2
    align_edges_cnt = brick_layout.align_edge_features.shape[0] // 2
    selected_cnt = np.sum(brick_layout.predict)

    coverage = calculate_coverage(brick_layout)

    f.write(f"node_cnt : {node_cnt} \n")
    f.write(f"collide_edges_cnt : {collide_edges_cnt} \n")
    f.write(f"align_edges_cnt : {align_edges_cnt} \n")
    f.write(f"selected_cnt : {selected_cnt} \n")
    f.write(f"coverage : {coverage} \n")
    f.write(f"predict_cnt : {predict_cnt} \n")

    f.close()


def calculate_coverage(brick_layout):
    ### Calculating coverage
    selected_tiles_area = list(map(lambda tile: tile.area, brick_layout.get_selected_tiles()))
    total_polygon = brick_layout.get_super_contour_poly()
    coverage = np.sum(selected_tiles_area) / total_polygon.area
    return coverage


def recover_bricklayout_from_redix_file(re_index_path, debugger, complete_graph):

    re_index =  pickle.load(open(re_index_path, "rb"))
    tiles_super_set = list(re_index.keys())

    filtered_collided_edges = [edge for edge in complete_graph.colli_edges if
                               edge[0] in tiles_super_set and edge[1] in tiles_super_set]
    filtered_adj_edges = [edge for edge in complete_graph.adj_edges if
                               edge[0] in tiles_super_set and edge[1] in tiles_super_set]
    node_feature, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features, target, new_re_index = \
        generate_brick_layout_data(complete_graph, tiles_super_set, filtered_collided_edges, filtered_adj_edges, [])

    output_layout = tiling.brick_layout.BrickLayout(debugger, complete_graph, node_feature, collide_edge_index,
                               collide_edge_features, align_edge_index, align_edge_features, target, new_re_index)

    for key, item in new_re_index.items():
        assert re_index[key] == item

    return output_layout


def shape_transform(complete_graph, exterior_contour, interior_contours, margin_padding_ratio, rotate_angle, x_delta,
                    y_delta):
    poly_bound = Polygon(exterior_contour, holes=interior_contours).bounds
    max_axis = max((poly_bound[2] - poly_bound[0]), (poly_bound[3] - poly_bound[1]))
    # calculate the stats for the complete graph
    x_min, x_max, y_min, y_max = get_graph_bound(complete_graph)
    graph_center = (x_max + x_min) / 2, (y_max + y_min) / 2
    base_diameter = min(x_max - x_min, y_max - y_min)
    exterior_contour_resized = exterior_contour / max_axis * base_diameter * margin_padding_ratio
    interior_contour_resized = [interior_contour / max_axis * base_diameter * margin_padding_ratio for
                                interior_contour in interior_contours]
    input_polygon = Polygon(exterior_contour_resized, holes=interior_contour_resized)
    input_polygon_centroid = np.array(input_polygon.centroid)
    target_polygon = shapely.affinity.translate(input_polygon, -input_polygon_centroid[0],
                                                -input_polygon_centroid[1])
    target_polygon = shapely.affinity.rotate(target_polygon, rotate_angle, origin="centroid")
    target_polygon = shapely.affinity.translate(target_polygon, graph_center[0] + x_delta,
                                                graph_center[1] + y_delta)
    return base_diameter, target_polygon.buffer(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    pass
