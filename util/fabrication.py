from util.debugger import MyDebugger
from interfaces.qt_plot import Plotter
from inputs import config
import tiling
import tiling.brick_layout
import os
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.ops import triangulate
import os


def clockwise_orientation(contour: np.ndarray) -> np.ndarray:
    """
    change a contour to counterclockwise direction
    """
    # using the shoelace formula and code adopted from https://stackoverflow.com/questions/14505565/
    # Wikipedia: det[x_i,x_i+1;y_i,y_i+1]
    shoelace = sum(
        [contour[i - 1, 0] * contour[i, 1] - contour[i, 0] * contour[i - 1, 1] for i in range(contour.shape[0])])
    if shoelace < 0:
        return contour
    else:
        return contour[::-1]

def generate_2d_obj(file_path, points):
    filename = file_path
    with open(filename, 'w') as file:
        for point in points:
            x, y = point
            print(f'v {x} {y} 0', file=file)
        print('o Spine', file=file)
        print('g Segment1', file=file)
        print('l', end=' ', file=file)
        for i in range(len(points)):
            print(i + 1, end=' ', file=file)
        print(1, file=file)


def read_2d_obj(filename):
    """
    Given a file path, read the 2d obj file and return an np.ndarray
    :param:file_path: the file path of the 2d obj file
    :return: return an ndarray
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        points = []
        for line in lines:
            str_list = line.split()
            if str_list[0] == 'v':
                point = Point(float(str_list[1]), float(str_list[2]))
                points.append(point)
    result = [(point.x, point.y) for point in points]
    result = np.array(result)
    result = clockwise_orientation(result)
    return result

def generate_3d_meshes(debugger, obj_folder, thickness):
    output_folder = obj_folder # let the output to be the same as input folder
    for filename in os.listdir(obj_folder):
        if filename.endswith(".obj"):
            generate_3d_mesh(debugger, output_filename = os.path.join(output_folder, f"3D_{filename}"),
                                     input_folder= os.path.join(obj_folder, filename), thickness=0.2)

def generate_3d_mesh(debugger: MyDebugger, output_filename: str, input_folder: str, thickness: float):
    """
    generate a 3D mesh of the given contour with the given thickness
    :param debugger: the debugger to provide directory for obj to be stored
    :param output_filename: filename (excluding the extension)
    :param contour: the contour to create 3d object with
    :param thickness: the thickness of 3d object mesh
    :return: None
    """
    filename_path = os.path.abspath(input_folder)
    assert os.path.isfile(filename_path)
    contour = read_2d_obj(input_folder)

    if output_filename[-4:] != '.obj':
        output_filename = output_filename + '.obj'
    destination = debugger.file_path(output_filename)
    with open(destination, 'w') as obj_file:
        point_to_vertex = {}
        for index, point in enumerate(contour):
            point_to_vertex[tuple(point)] = (index * 2 + 1, index * 2 + 2)
            print(f'v {point[0]} {point[1]} 0', file=obj_file)
            print(f'v {point[0]} {point[1]} {thickness}', file=obj_file)

        contour_poly = Polygon(contour)
        triangles = triangulate(contour_poly)
        for triangle in triangles:
            if len(triangles) > 1:
                triangle_bound = LineString(triangle.exterior)
                if not triangle_bound.within(contour_poly):
                    continue
            *points, _ = triangle.exterior.coords
            face_1, face_2 = zip(*[point_to_vertex[point] for point in points])
            for face in (face_1[::-1], face_2):
                print('f ' + ' '.join([str(i) for i in face]), file=obj_file)
        for index, point in enumerate(contour):
            lower_point, upper_point = point_to_vertex[tuple(point)]
            lower_prev, upper_prev = point_to_vertex[tuple(contour[index - 1])]
            print('f ' + ' '.join([str(point) for point in (upper_prev, lower_point, upper_point)]), file=obj_file)
            print('f ' + ' '.join([str(point) for point in (upper_prev, lower_prev, lower_point)]), file=obj_file)



if __name__ == "__main__":
    MyDebugger.pre_fix = os.path.join(MyDebugger.pre_fix, "debug")
    debugger = MyDebugger("brick_layout_test", fix_rand_seed=0)
    plotter = Plotter()
    # data_env = config.environment
    # data_env.load_complete_graph(1)
    #
    # node_feature, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features, gt, re_index = tiling.TileFactory.gen_one_train_data(
    #     plotter, data_env.complete_graph, low=2, high=10)
    # brick_layout = tiling.brick_layout.BrickLayout(debugger, data_env.complete_graph, node_feature, collide_edge_index, collide_edge_features,
    #                            align_edge_index,align_edge_features, gt, re_index)
    #
    # # brick_layout.target = target
    # brick_layout.show_complete_graph(plotter, f"complete_graph.png")
    # brick_layout.show_ground_truth(plotter,   f"GT.png")
    # brick_layout.predict = brick_layout.ground_truth
    #
    # brick_layout.save_predict_as_objs("tile")
    # # generate 3D mesh with axle hole
    generate_3d_meshes(debugger, obj_folder=r"G:\learn_assemble_data\debug\2019-12-16_16-41-01_network_eval_33_30-60-90-ring4-5000-epoch300\result\train\data_19\objs\tree_search_predict_top_4_0_objs",thickness= 0.3)

    # brick_layout.show_super_graph(plotter,    f"supper_graph.png")
    # brick_layout.show_super_contour(plotter,  f"super_contour.png")