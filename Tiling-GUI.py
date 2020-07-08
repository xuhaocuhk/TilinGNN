import os
import tiling.tile_factory as factory
from tiling.brick_layout import BrickLayout
from shapely.geometry import Polygon, LineString
from solver.ml_solver.ml_solver import ML_Solver
from util.debugger import MyDebugger
from interfaces.qt_plot import Plotter
from interfaces.qt_plot import get_scale_translation_polygons
import numpy as np
import sys, math
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtGui import QCursor, QPainter, QPolygon, QPen, QColor, QKeyEvent, QPixmap, QBrush
from PyQt5.QtWidgets import QWidget, QLCDNumber, QSlider, \
    QVBoxLayout, QApplication, QPushButton, QFileDialog, QInputDialog, QMainWindow
from util.data_util import write_brick_layout_data, load_bricklayout
from inputs import config
from interfaces import figure_config as fig_conf
from copy import deepcopy
from inputs.shape_factory import export_contour_as_text
import torch
import traceback
import itertools
import pickle
from shapely.affinity import translate, scale, rotate
from graph_networks.networks.TilinGNN import TilinGNN

SIMPLIFY_RATIO = 5e-3
scale_ratio = 0.02
rotation_angle_ratio = 3
translation_ratio = 0.05

solution_color = "figure_blue"

class DrawerWidgets(QMainWindow):
    def __init__(self, graph, debugger : MyDebugger, plotter : Plotter, solver, app : QApplication):
        super(QMainWindow, self).__init__()
        self.setMinimumSize(QSize(800, 800))
        self.complete_graph = graph
        self.debugger = debugger
        self.plotter = plotter
        self.current_polygon_exterior = []
        self.current_polygon_interiors = []
        self.scale = None
        self.translation = None
        self.current_best_solution = None
        self.initUI()
        self.app = app

        ########################## PENS #############################
        self.grid_pen = QPen(QColor(120, 120, 120, 255), 1)

        self.external_contour_pen = QPen(QColor(220,20,60, 255), 3)
        self.external_point_pen = QPen(QColor(220,20,60, 255), 8)
        self.internal_contour_pen = QPen(QColor(34,139,34, 255), 3)
        self.internal_point_pen = QPen(QColor(34,139,34, 255), 8)

        self.selected_point_pen = QPen(QColor(0, 0, 255, 255), 8)


        ######################## MOUSE ##############################
        self.mouse_mode = 'EDIT'
        self.current_selected_tuple = None

        ######################## SOLVING ##############################
        self.current_brick_layout = None
        self.current_solution_layout = None
        self.solver = solver
        self.brick_layout_cnt = 0
        self.solution_cnt = 0
        self.window_width = None
        self.window_height = None

        self.pixmap = QPixmap(self.size())
        self.pixmap.fill(Qt.transparent)
        self.need_draw_grid = True


        ######################## RESULT PENS ##########################
        self.types = ('lightgreen', 'green', 'lightblue', 'blue', "blue_trans",
                 'pink_red', 'purple', 'violet', 'green_blue', 'orange', 'yellow', 'light_gray',
                 'pink_blue', 'white', 'white_border', 'light_gray_border', 'figure_blue')
        self.pens = {
            type: QPen(QColor(*getattr(fig_conf, type)['edge'])) for type in self.types
        }
        self.brushes = {
            type: QBrush(QColor(*getattr(fig_conf, type)['face'])) for type in self.types
        }

        for pen in self.pens.values():
            pen.setWidth(fig_conf.edge_width)


        ###################### Scaling ###############################
        self.polygon_scale = 1.0
        self.polygon_rotation_angle = 0.0
        self.polygon_translation_x_delta = 0.0
        self.polygon_translation_y_delta = 0.0

        self.graph_x_min, self.graph_x_max, self.graph_y_min, self.graph_y_max = factory.get_graph_bound(self.complete_graph)

    def initUI(self):
        self.setWindowTitle('Tiling Interface')
        self.setWindowFlags(self.windowFlags())
        self.show()

######################### MOUSE ########################################
    def mousePressEvent(self, QMouseEvent):
        if self.mouse_mode == 'DRAW':
            self.add_Point(QMouseEvent.pos().x(), QMouseEvent.pos().y())
        elif self.mouse_mode == 'EDIT':
            self.current_selected_tuple = self.selectPoint(QMouseEvent.pos().x(), QMouseEvent.pos().y())
        else:
            print("Something go wrong in mouse click!")

    def mouseMoveEvent(self, QMouseEvent):
        if self.current_selected_tuple is not None:
            self.updatePointPos(QMouseEvent.pos().x(), QMouseEvent.pos().y(), self.current_selected_tuple)

    def mouseReleaseEvent(self, QMouseEvent):
        self.current_selected_tuple = None
        self.repaint()

######################### KEY PRESS ########################################
    def keyPressEvent(self, event):
        # print(event)
        if type(event) == QKeyEvent:
            # here accept the event and do something
            if event.key() == ord('E'):
                if self.current_polygon_exterior is not None:
                    self.compute_Brick_Layout()
            elif event.key() == ord('S'):
                if self.current_brick_layout is not None:
                    self.current_solution_layout = self.solve_Brick_Layout(self.current_brick_layout)
            elif event.key() == ord('R'):
                self.current_polygon_exterior.clear()
                self.current_polygon_interiors.clear()
                self.current_brick_layout = None
                self.repaint()
            elif event.key() == ord('O'):
                self.open_silhouette_files()
            elif event.key() == ord('L'):
                self.load_brick_layout()
            elif event.key() == ord('M'):
                if self.mouse_mode == 'DRAW':
                    self.mouse_mode = 'EDIT'
                elif self.mouse_mode == 'EDIT':
                    self.mouse_mode = 'DRAW'
            elif event.key() == ord('Z'):
                self.affline_transform(scale_rate = scale_ratio)
            elif event.key() == ord('X'):
                self.affline_transform(scale_rate = - scale_ratio)
            elif event.key() == ord('='):
                self.affline_transform(rotation_angle = rotation_angle_ratio)
            elif event.key() == ord('-'):
                self.affline_transform(rotation_angle = - rotation_angle_ratio)
            elif event.key() == Qt.Key_Up:
                self.affline_transform(translation_y_rate = - translation_ratio)
            elif event.key() == Qt.Key_Down:
                self.affline_transform(translation_y_rate = translation_ratio)
            elif event.key() == Qt.Key_Left:
                self.affline_transform(translation_x_rate = - translation_ratio)
            elif event.key() == Qt.Key_Right:
                self.affline_transform(translation_x_rate = translation_ratio)

            event.accept()
        else:
            event.ignore()

    def affline_transform(self, scale_rate = None, translation_x_rate = None, translation_y_rate = None, rotation_angle = None):

        scale_rate = 1.0 if scale_rate is None else 1 + scale_rate
        translation_x_rate = 0.0 if translation_x_rate is None else translation_x_rate
        translation_y_rate = 0.0 if translation_y_rate is None else translation_y_rate
        rotation_angle = 0.0 if rotation_angle is None else rotation_angle
        self.current_polygon_exterior = DrawerWidgets.transform_shape(self.current_polygon_exterior, scale_rate, rotation_angle, translation_x_rate, translation_y_rate)
        self.current_polygon_interiors = [DrawerWidgets.transform_shape(interior, scale_rate, rotation_angle, translation_x_rate, translation_y_rate,
                                                                        LineString(self.current_polygon_exterior).centroid) for interior in self.current_polygon_interiors]
        self.repaint()


    @staticmethod
    def transform_shape(shape, scale_rate, rotation_rate, translation_rate_x, translation_rate_y, center_point = 'centroid'):


        shape = LineString(shape)
        print(f'before : {np.array(shape.coords)}')
        shape = scale(shape, scale_rate, scale_rate, origin = center_point)
        shape = rotate(shape, rotation_rate, origin = center_point)
        shape = translate(shape, translation_rate_x, translation_rate_y)
        print(f'after : {np.array(shape.coords)}')

        output_coords = list(np.array(shape.coords))
        return output_coords

######################### PAINT ########################################
    def paintEvent(self, event):
        painter = QPainter(self)

        # complete graph ratio
        scale, translation = self._get_scale_translation()

        if self.need_draw_grid:
            self.draw_grid(scale, translation)

        painter.drawPixmap(QPoint(), self.pixmap)

        ############ EXTERIOR #################################
        painter.setPen(self.external_contour_pen)
        exterior_coords_on_screen = [QPoint(*tuple(p * scale + translation)) for p in self.current_polygon_exterior]
        painter.drawPolyline(QPolygon(exterior_coords_on_screen))

        ############ INTERIOR #################################
        painter.setPen(self.internal_contour_pen)
        interior_coords_list = []
        for current_polygon_interior in self.current_polygon_interiors:
            interior_coords_on_screen = [QPoint(*tuple(p * scale + translation)) for p in current_polygon_interior]
            painter.drawPolyline(QPolygon(interior_coords_on_screen))
            interior_coords_list.append(interior_coords_on_screen)

        ####################### Points of each point ##########
        # Draw selected Point in different color
        if self.current_selected_tuple is not None and self.current_selected_tuple[0] == 'exterior':
            exterior_coords_on_screen = self.draw_exterior_selected_point(exterior_coords_on_screen, painter)

        if self.current_selected_tuple is not None and self.current_selected_tuple[0] == 'interior':
            interior_coords_list = self.draw_interior_selected_point(interior_coords_list, painter)

        painter.setPen(self.external_point_pen)
        painter.drawPoints(QPolygon(exterior_coords_on_screen))

        painter.setPen(self.internal_point_pen)
        for interior_coord in interior_coords_list:
            painter.drawPoints(QPolygon(interior_coord))

        ###################### Each tile in the solution ###############
        if self.current_best_solution is not None:
            self.draw_solution(painter)

    def draw_grid(self, scale, translation):
        grid_painter = QPainter(self.pixmap)
        grid_painter.setRenderHint(QPainter.Antialiasing)
        grid_painter.setRenderHint(QPainter.SmoothPixmapTransform)
        ############ Draw grids #############################
        grid_painter.setPen(self.grid_pen)
        polygons = [Plotter.create_polygon(np.array(t.tile_poly.exterior.coords) * scale + translation) for t in
                    self.complete_graph.tiles]
        for p in polygons:
            grid_painter.drawPolygon(p)

        self.need_draw_grid = False

    def draw_solution(self, painter :QPainter):
        tiles = self.current_best_solution.get_selected_tiles()

        painter.setPen(self.pens[solution_color])
        painter.setBrush(self.brushes[solution_color])
        scale, translation = self._get_scale_translation(False)
        polygons = [Plotter.create_polygon(np.array(t.exterior.coords) * scale + translation) for t in
                    tiles]
        for p in polygons:
            painter.drawPolygon(p)

    def draw_exterior_selected_point(self, exterior_coords_on_screen, painter):
        selected_index = self.current_selected_tuple[1]
        ### draw the end point in selected color if neccessary
        painter.setPen(self.selected_point_pen)
        if selected_index == 0 or selected_index == len(self.current_polygon_exterior) - 1:
            selected_point = exterior_coords_on_screen.pop(len(self.current_polygon_exterior) - 1)
            painter.drawPoint(selected_point)
            selected_point = exterior_coords_on_screen.pop(0)
            painter.drawPoint(selected_point)
        else:
            selected_point = exterior_coords_on_screen.pop(selected_index)
            painter.drawPoint(selected_point)

        return exterior_coords_on_screen

    def resizeEvent(self, event):
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.transparent)
        self.pixmap = pixmap
        self.need_draw_grid = True
        p = QPainter(pixmap)
        p.drawPixmap(QPoint(), self.pixmap)
        QMainWindow.resizeEvent(self, event)
        self.scale, self.translation = self._get_scale_translation(True)

    def draw_interior_selected_point(self, interior_coords_list, painter):
        interior_index, point_index = self.current_selected_tuple[1]
        ### draw the end point in selected color if neccessary
        painter.setPen(self.selected_point_pen)
        total_len = len(interior_coords_list[interior_index])
        if point_index == 0 or point_index == total_len - 1:
            selected_point = interior_coords_list[interior_index].pop(total_len - 1)
            painter.drawPoint(selected_point)
            selected_point = interior_coords_list[interior_index].pop(0)
            painter.drawPoint(selected_point)
        else:
            selected_point = interior_coords_list[interior_index].pop(point_index)
            painter.drawPoint(selected_point)

        return interior_coords_list

    ######################### OTHERS ########################################
    def open_silhouette_files(self, margin_padding_ratio = None, rotate_angle = None, x_delta = None, y_delta = None):
        filename, filetype = QFileDialog.getOpenFileName(self, 'Open File', '/home/edwardhui/data/silhouette/selected_v2')


        if margin_padding_ratio is None:
            margin_padding_ratio ,ok = QInputDialog.getDouble(self,"Enter a padding ratio","Enter a padding ratio", decimals = 2)
        if rotate_angle is None:
            rotate_angle, ok = QInputDialog.getDouble(self, "Enter a rotation angle", "Enter a angle")
        if x_delta is None:
            x_delta, ok = QInputDialog.getDouble(self, "Enter a x axis delta", "Enter a delta")
        if y_delta is None:
            y_delta, ok = QInputDialog.getDouble(self, "Enter a y axis delta", "Enter a delta")

        if margin_padding_ratio is not None and \
            rotate_angle is not None and \
            x_delta is not None and \
            y_delta is not None:
            ok = True


        tile_bound = self.complete_graph.tiles[0].tile_poly.bounds
        tile_delta = min((tile_bound[2] - tile_bound[0]), (tile_bound[3] - tile_bound[1]))
        x_delta = x_delta * tile_delta
        y_delta = y_delta * tile_delta

        if os.path.exists(filename) and ok:
            exterior_contour, interior_contours = factory.load_polygons(filename = filename)

            base_diameter, target_polygon = factory.shape_transform(self.complete_graph, exterior_contour, interior_contours,
                                                                    margin_padding_ratio, rotate_angle, x_delta,
                                                                    y_delta)

            # simplify the output
            target_polygon = target_polygon.simplify(base_diameter * SIMPLIFY_RATIO)

            self.current_polygon_exterior = list(np.array(target_polygon.exterior.coords))
            self.current_polygon_interiors = list(map(lambda interior : list(np.array(interior.coords)), target_polygon.interiors))
            self.repaint()
        else:
            print(f"{filename} is not silhouette file.")

    def _get_scale_translation(self, need_recaluate = False):
        if self.scale is None or self.translation is None or need_recaluate:
            scale, translation = get_scale_translation_polygons(
                 [np.array(t.tile_poly.exterior.coords) for t in self.complete_graph.tiles],
                 self
            )
            translation = np.array(translation)
            self.scale, self.translation = scale, translation
            return scale, translation
        else:
            return self.scale, self.translation

    def selectPoint(self, mouseX, mouseY):
        output_index = None
        mouse_world_pos = self.getMousePosWorld(mouseX, mouseY)

        ### EXTERIORS
        exterior_point_min_index = None
        distance_for_exterior = []
        if len(self.current_polygon_exterior) > 0:
            distance_for_exterior = list(map(lambda point : np.linalg.norm(mouse_world_pos-point), self.current_polygon_exterior))
            exterior_point_min_index = int(np.argmin(distance_for_exterior))

        ### INTERIORS
        interior_point_minimum = None
        distance_for_all_interiors = []
        if len(self.current_polygon_interiors) > 0:
            for idx, interior_coords in enumerate(self.current_polygon_interiors):
                distance_for_interior = list(map(lambda point : np.linalg.norm(mouse_world_pos-point), interior_coords))
                interior_point_min_index = int(np.argmin(distance_for_interior))
                distance_for_all_interiors.append((distance_for_interior[interior_point_min_index], idx, interior_point_min_index))

            interior_point_minimum = sorted(distance_for_all_interiors, key = lambda tup: tup[0])[0]

        ### check whether internal or external
        if exterior_point_min_index is not None:
            assert len(distance_for_exterior) > 0
            ## return exterior
            if interior_point_minimum is not None:
                if distance_for_exterior[exterior_point_min_index] < interior_point_minimum[0]:
                    output_index = ('exterior', exterior_point_min_index)
                else:
                    output_index = ('interior', (interior_point_minimum[1], interior_point_minimum[2]))
            else:
                output_index = ('exterior', exterior_point_min_index)

        return output_index

    def getMousePosWorld(self, mouseX, mouseY):
        scale, translation = self._get_scale_translation()
        mouse_pos = np.array([mouseX, mouseY])
        mouse_world_pos = (mouse_pos - translation) / scale
        return mouse_world_pos

    def updatePointPos(self, mouseX, mouseY, update_tuple):
        mouse_world_pos = self.getMousePosWorld(mouseX, mouseY)

        update_type, update_index = update_tuple

        if update_type == 'exterior':
            self.current_polygon_exterior[update_index] = mouse_world_pos

            ## update the index for head and end as well
            if update_index == 0 or update_index == len(self.current_polygon_exterior) - 1:
                self.current_polygon_exterior[len(self.current_polygon_exterior) - 1 - update_index] = mouse_world_pos
        elif update_type == 'interior':
            total_len = len(self.current_polygon_interiors[update_index[0]])
            self.current_polygon_interiors[update_index[0]][update_index[1]] = mouse_world_pos

            ## update the index for head and end as well
            if update_index[1] == 0 or update_index[1] == total_len - 1:
                self.current_polygon_interiors[update_index[0]][total_len - 1 - update_index[1]] = mouse_world_pos
        else:
            print("Something goes wrong!")

        # redraw
        self.repaint()

    def add_Point(self, x, y):
        mouse_pos = np.array([x , y])
        # add the points in world space
        scale, translation = self._get_scale_translation()

        ### Remove last element if needed
        if len(self.current_polygon_exterior) >= 3:
            self.current_polygon_exterior.pop()

        self.current_polygon_exterior.append((mouse_pos - translation) / scale)

        ### add the first point necessary
        if len(self.current_polygon_exterior) >= 3:
            self.current_polygon_exterior.append(self.current_polygon_exterior[0])

        self.repaint()

    def compute_Brick_Layout(self):
        target_polygon = Polygon(self.current_polygon_exterior, holes=self.current_polygon_interiors)
        file_prefix = f'target_shpape_{self.brick_layout_cnt}'


        try:
            node_feature, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features, re_index = \
                factory.create_brick_layout_from_polygon(self.complete_graph, target_polygon)
        except:
            print(traceback.format_exc())
            return

        #### Saving contour
        export_contour_as_text(self.debugger.file_path(f'contour_{self.brick_layout_cnt}.txt'), (self.current_polygon_exterior, self.current_polygon_interiors))


        self.current_brick_layout = BrickLayout(self.complete_graph, node_feature, collide_edge_index,
                                                collide_edge_features,
                                                align_edge_index, align_edge_features, re_index, target_polygon = target_polygon)

        factory.save_all_layout_info(file_prefix, self.current_brick_layout, self.debugger.file_path('./'), with_features = False)

        # visual_brick_layout_graph(brick_layout = self.current_brick_layout, save_path = self.debugger.file_path('./graph.png'))
        self.brick_layout_cnt += 1
        self.solution_cnt = 0
        self.current_brick_layout.show_candidate_tiles(self.plotter, self.debugger,
                                                       f'{self.brick_layout_cnt}_super_graph.png',
                                                       style="blue_trans")
        self.plotter.draw_contours(self.debugger.file_path(
            f'generated_shape_{self.brick_layout_cnt}.png'),
            [('green', np.array(target_polygon.exterior.coords))])

    def solve_Brick_Layout(self, brick_layout, trial_times = None):

        if trial_times is None:
            trial_times, ok = QInputDialog.getInt(self, "Enter number of trial time", "Enter number of trial time")
        else:
            ok = True

        if ok:
            current_max_score = 0.0
            for i in range(trial_times):

                # all dirs
                save_path = debugger.file_path('./')

                self.solution_cnt += 1

                result_brick_layout, score = self.solver.solve(brick_layout)

                if score != 0:
                    if current_max_score < score:
                        current_max_score = score
                        self.current_best_solution = deepcopy(result_brick_layout)
                        self.repaint()
                        self.app.processEvents()
                        print(f"current_max_score : {current_max_score}")

                    result_brick_layout.show_predict(plotter, self.debugger, os.path.join(save_path,
                                                                           f'{score}_{self.brick_layout_cnt}_{self.solution_cnt}_predict.png'))
                    result_brick_layout.show_super_contour(plotter, self.debugger, os.path.join(save_path,
                                                                                 f'{score}_{self.brick_layout_cnt}_{self.solution_cnt}_super_contour.png'))
                    result_brick_layout.show_candidate_tiles(plotter, self.debugger, os.path.join(save_path,
                                                                               f'{score}_{self.brick_layout_cnt}_{self.solution_cnt}_super_graph.png'))

                    file_prefix = f'{score}_{self.brick_layout_cnt}_{self.solution_cnt}'
                    factory.save_all_layout_info(file_prefix = file_prefix, result_brick_layout = result_brick_layout, save_path = save_path, with_features = False)

            return result_brick_layout
        else:
            return None

    def load_brick_layout(self):
        filename, filetype = QFileDialog.getOpenFileName(self, 'Open File')
        if os.path.exists(filename):
            self.current_brick_layout = load_bricklayout(filename, self.complete_graph)
            polygon = self.current_brick_layout.get_super_contour_poly()
            self.current_polygon_exterior = list(np.array(polygon.exterior.coords))
            self.current_polygon_interiors = list(map(lambda interior : list(np.array(interior)), polygon.interiors))
            self.repaint()
            print(f"{filename} loaded!")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Instructions here
Keys:
E : cropping the superset according to current shape
S : solve the tiling problem by TilinGNN
R : clear the draw shape and solution
O : open a shape file in txt format
L : load a precomputed layout file
M : change mode (Edit mode <-> Draw mode)
Z/X : scale up/ down
-/= : rotate clockwise/anti-clockwise
UP/DOWN/LEFT/RIGHT : translate up/down/left/right

Mouse:
Edit mode:
left click : add a new point to the location
Draw mode:
left drag an existing point : change the location of that point

Procedure for solving drawing shapes:
1. Change to drawing mode by Key M
2. Draw the shape that you like
3. Transform by the shapes (Z/X, -/=, UP/DOWN/LEFT/RIGHT) or edit the shapes in Edit mode
4. Crop superset by Key E
5. Solve the problem by Key S (with number of trials)

Procedure for solving silhouette shapes:
1. Load the txt file in the ./silhouette folder by Key O
2. Transform by the shapes (Z/X, -/=, UP/DOWN/LEFT/RIGHT) or edit the shapes in Edit mode
3. Crop superset by Key E
4. Solve the problem by Key S (with number of trials)
'''

if __name__ == "__main__":
    MyDebugger.pre_fix = os.path.join(MyDebugger.pre_fix, "debug")
    debugger = MyDebugger("UI_interface", fix_rand_seed=0, save_print_to_file = False)
    plotter = Plotter()
    data_env = config.environment
    data_env.load_complete_graph(config.complete_graph_size)

    network = TilinGNN(adj_edge_features_dim=data_env.complete_graph.total_feature_dim,
                       network_depth=config.network_depth, network_width=config.network_width).to(device)
    solver = ML_Solver(debugger, device, data_env.complete_graph, network, num_prob_maps = 1)
    solver.load_saved_network(config.network_path)

    app = QApplication(sys.argv)
    draw = DrawerWidgets(data_env.complete_graph, debugger, plotter, solver, app)
    sys.exit(app.exec_())

