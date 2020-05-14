import numpy as np
import shapely
from util.algo_util import align
from tiling.tile import Tile

'''
@param base_seg_p0 and p1: the edge to be aligned
@param tile_poly: the new tile to align
@param edge_idx: the edge of the new tile
@param reflect: if reflect
@param align_mode: align mode
@return: new tile instance of shapely Polygon
'''
EPS = 1e-5
def align_tile(base_seg_p0, base_seg_p1, tile_poly, edge_idx, align_mode):
    new_tile = tile_poly
    tile_edge_p0 = np.array(list(new_tile.exterior.coords))[edge_idx]
    tile_edge_p1 = np.array(list(new_tile.exterior.coords))[edge_idx + 1]
    R, T = align(base_seg_p0, base_seg_p1,
                 tile_edge_p0, tile_edge_p1,
                 align_mode)
    trans_mat = (R[0, 0], R[0, 1], R[1, 0], R[1, 1], T[0, 0], T[0, 1])
    new_tile = shapely.affinity.affine_transform(new_tile, trans_mat)
    return new_tile

def intersection_area(tile1: Tile, tile2: Tile):
    return tile1.tile_poly.intersection(tile2.tile_poly).area

def normalize(x):
    norm = np.linalg.norm(x)
    if abs(norm) < EPS:
        return [0, 0]
    else:
        return x / norm

def is_partial_edge_connected(tile_1: Tile, tile_2: Tile):
    # always clockwise orientation
    assert not tile_1.tile_poly.exterior.is_ccw
    assert not tile_2.tile_poly.exterior.is_ccw
    ##########################
    # checking the whether two triangle align
    trinagle_1_points = list(tile_1.tile_poly.exterior.coords)
    trinagle_2_points = list(tile_2.tile_poly.exterior.coords)

    for i in range(tile_1.get_edge_num()):
        for j in range(tile_2.get_edge_num()):
            # 4 points for checking
            a_1 = np.array([trinagle_1_points[i][0], trinagle_1_points[i][1]])
            a_2 = np.array([trinagle_1_points[i+1][0], trinagle_1_points[i+1][1]])

            line_a = a_2 - a_1

            b_1 = np.array([trinagle_2_points[j][0], trinagle_2_points[j][1]])
            b_2 = np.array([trinagle_2_points[j+1][0], trinagle_2_points[j+1][1]])

            line_b = b_2 - b_1

            # check whether the slope is the same
            if abs(abs(normalize(line_a).dot(normalize(line_b))) - 1.0) < EPS \
                    and (abs(abs(normalize(line_a).dot(normalize(b_1 - a_2))) - 1.0) < EPS or
                         abs(abs(normalize(line_a).dot(normalize(b_2 - a_2))) - 1.0) < EPS):

                base_vec = normalize(line_a)
                unit_a_1 = 0
                unit_a_2 = line_a.dot(base_vec)
                unit_b_1 = (b_1 - a_1).dot(base_vec)
                unit_b_2 = (b_2 - a_1).dot(base_vec)

                current_overlap = max(0.0,
                                      min(unit_a_2, max(unit_b_1, unit_b_2)) - max(unit_a_1, min(unit_b_1, unit_b_2)))
                if current_overlap > EPS:
                    tile_1_edge_length = tile_1.get_edge_length(i)
                    tile_2_edge_length = tile_2.get_edge_length(j)
                    if abs(tile_1_edge_length - tile_2_edge_length) > EPS:
                        return True

    return False

def polygon_align_length(tile_1: Tile, tile_2: Tile):
    # always clockwise orientation
    assert not tile_1.tile_poly.exterior.is_ccw
    assert not tile_2.tile_poly.exterior.is_ccw
    ##########################
    # checking the whether two triangle align
    trinagle_1_points = list(tile_1.tile_poly.exterior.coords)
    trinagle_2_points = list(tile_2.tile_poly.exterior.coords)

    total_overlap = 0.0

    for i in range(tile_1.get_edge_num()):
        for j in range(tile_2.get_edge_num()):
            # 4 points for checking
            a_1 = np.array([trinagle_1_points[i][0], trinagle_1_points[i][1]])
            a_2 = np.array([trinagle_1_points[i+1][0], trinagle_1_points[i+1][1]])

            line_a = a_2 - a_1

            b_1 = np.array([trinagle_2_points[j][0], trinagle_2_points[j][1]])
            b_2 = np.array([trinagle_2_points[j+1][0], trinagle_2_points[j+1][1]])

            line_b = b_2 - b_1

            # check whether the slope is the same
            if abs(abs(normalize(line_a).dot(normalize(line_b))) - 1.0) < EPS \
               and (abs(abs(normalize(line_a).dot(normalize(b_1 - a_2))) - 1.0) < EPS or
                    abs(abs(normalize(line_a).dot(normalize(b_2 - a_2))) - 1.0) < EPS):

                base_vec = normalize(line_a)
                unit_a_1 = 0
                unit_a_2 = line_a.dot(base_vec)
                unit_b_1 = (b_1 - a_1).dot(base_vec)
                unit_b_2 = (b_2 - a_1).dot(base_vec)

                current_overlap = max(0.0, min(unit_a_2, max(unit_b_1,unit_b_2)) - max(unit_a_1, min(unit_b_1, unit_b_2)))
                if current_overlap > EPS:
                    total_overlap = total_overlap + current_overlap

    return total_overlap

# identify the first contact point aligned with the other tile, there should exists contact point along clockwise direction after the point
def get_first_touch_point(tile_1, tile_2):
    # always clockwise orientation
    assert not tile_1.tile_poly.exterior.is_ccw
    assert not tile_2.tile_poly.exterior.is_ccw
    assert polygon_align_length(tile_1, tile_2) > EPS
    assert intersection_area(tile_1, tile_2) < EPS

    ##########################
    # checking the whether two triangle align
    trinagle_1_points = list(tile_1.tile_poly.exterior.coords)
    trinagle_2_points = list(tile_2.tile_poly.exterior.coords)

    for i in range(tile_1.get_edge_num()):
        for j in range(tile_2.get_edge_num()):
            # 4 points for checking
            a_1 = np.array([trinagle_1_points[i][0], trinagle_1_points[i][1]])
            a_2 = np.array([trinagle_1_points[i+1][0], trinagle_1_points[i+1][1]])
            line_a = a_2 - a_1

            b_1 = np.array([trinagle_2_points[j][0], trinagle_2_points[j][1]])
            b_2 = np.array([trinagle_2_points[j+1][0], trinagle_2_points[j+1][1]])
            line_b = b_2 - b_1

            # the two vectors are parallel
            if abs(abs(normalize(line_a).dot(normalize(line_b))) - 1.0) < EPS \
               and (abs(abs(normalize(line_a).dot(normalize(b_1 - a_2))) - 1.0) < EPS or # the segment is also colinear with the cross segment
                    abs(abs(normalize(line_a).dot(normalize(b_2 - a_2))) - 1.0) < EPS):
                # the two segments are co-linear and parallel
                base_vec_a = normalize(line_a)
                unit_a_1_basea = 0
                unit_a_2_basea = line_a.dot(base_vec_a)
                unit_b_1_basea = (b_1 - a_1).dot(base_vec_a)
                unit_b_2_basea = (b_2 - a_1).dot(base_vec_a)

                base_vec_b = normalize(line_b)
                unit_b_1_baseb = 0
                unit_a_1_baseb = (a_1 - b_1).dot(base_vec_b)
                unit_a_2_baseb = (a_2 - b_1).dot(base_vec_b)

                current_overlap = max( 0.0, min(unit_a_2_basea, max(unit_b_1_basea,unit_b_2_basea) ) - max(unit_a_1_basea, min(unit_b_1_basea, unit_b_2_basea) ) )
                if current_overlap > EPS:
                    point_a = max(unit_a_1_basea, min(unit_b_1_basea, unit_b_2_basea) ) # point on edge i of tile_1
                    point_b = max(unit_b_1_baseb, min(unit_a_1_baseb, unit_a_2_baseb) ) # point on edge j of tile_2
                    return i, point_a, j, point_b

def polygon_align_type(tile_1: Tile, tile_2: Tile):
    i, point_a, j, point_b = get_first_touch_point(tile_1, tile_2)
    return tile_1.get_align_point(i, point_a), tile_2.get_align_point(j, point_b)
