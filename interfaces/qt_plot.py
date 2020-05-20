"""
Plotting library using Qt
"""
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import numpy as np
import interfaces.figure_config as fig_conf
from typing import Iterable, Tuple, Union
from util.debugger import MyDebugger


class Plotter:
    types = ('lightgreen', 'green', 'lightblue', 'blue', "blue_trans",
             'pink_red', 'purple', 'violet', 'green_blue', 'orange', 'yellow', 'light_yellow', 'light_gray',
             'pink_blue', 'white', 'white_border', 'light_gray_border')
    pens = {
        type: QtGui.QPen(QtGui.QColor(*getattr(fig_conf, type)['edge'])) for type in types
    }
    brushes = {
        type: QtGui.QBrush(QtGui.QColor(*getattr(fig_conf, type)['face'])) for type in types
    }

    for pen in pens.values():
        pen.setWidth(fig_conf.edge_width)


    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = PlotterWindow()
        self.window.resize(*fig_conf.figure_size)
        self.pen_width = fig_conf.edge_width
        # self.app.exec_()

    @staticmethod
    def create_polygon(contour: np.ndarray) -> QtGui.QPolygonF:
        """
        Create a QPolygonF object from the given contour
        :param contour: numpy contour
        :return: Qt Polygon
        """
        points = [QtCore.QPointF(x, y) for x, y in contour]
        # logger.debug(f'Creating polygon with points {points}')
        return QtGui.QPolygonF(points)

    def scaled_polygon(self, contour: np.ndarray) -> QtGui.QPolygonF:
        # logging.debug(f'Scaling polygon {contour}')
        return self.create_polygon((contour + self.translation) * self.scaling)

    def draw_contours(self, file_path: str, contours: Iterable[ Tuple[ Union[str, Tuple[Tuple[float, ...], Tuple[Tuple[float, ...]]]],
                                                                       np.ndarray]]):
        """
        draw the given contours and save to an image file
        :param file_path: the path to the image file to save
        :param contours: (color option, contour) of the contours to draw
        :return: None
        """
        if len(contours) == 0:
            print("Nothing to plot.")
            return
        else:
            print(f'saving file {file_path}...')
        # get scale and translate
        scale, translate = get_scale_translation_polygons([contour for _, contour in contours], self.window)

        # set up attributes
        self.window.polygons = [self.create_polygon(contour * scale + translate) for _, contour in contours]

        ### get brushes
        self.window.brushes = [self.brushes[config] if isinstance(config, str)
                                else QtGui.QBrush(QtGui.QColor(*config[0]))
                                for config, _ in contours]

        ### get pen and change the width
        self.window.pens = [self.pens[config] if isinstance(config, str)
                            else QtGui.QPen(QtGui.QColor(*config[1]))
                            for config, _ in contours]
        for pen in self.window.pens:
            pen.setWidth(fig_conf.edge_width)
        self.window.setStyleSheet('background-color: white;')
        self._save_canvas(file_path)

    def _save_canvas(self, file_path: str):
        self.window.repaint()
        self.window.grab().save(file_path)

    def __del__(self):
        self.window.close()
        self.app.quit()
        del self.window
        del self.app


class PlotterWindow(QtWidgets.QWidget):
    def __init__(self):
        self.polygons = []
        self.pens = []
        self.brushes = []
        super().__init__()

    def paintEvent(self, event: QtGui.QPaintEvent):
        assert len(self.pens) == len(self.brushes) and len(self.brushes) == len(self.polygons)
        painter = QtGui.QPainter(self)
        for pen, brush, polygon in zip(self.pens, self.brushes, self.polygons):
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawPolygon(polygon)

def get_scale_translation_polygons(polygon_list, widget : QtWidgets.QWidget):
    polygons = list(polygon_list)

    x_max, x_min, y_max, y_min = get_polygon_bound(polygons)

    #### calculate scale and translate
    scale = min(widget.width() / (x_max - x_min), widget.height() / (y_max - y_min))
    scale = scale * (1 - fig_conf.white_padding)
    translate = (widget.width() / 2 - scale * (x_min + x_max) / 2, \
                 widget.height() / 2 - scale * (y_min + y_max) / 2)

    return scale, translate


def get_polygon_bound(polygons):
    ##### get spatial range of all tiles
    x_min = np.min([min(polygon[:, 0]) for polygon in polygons])
    x_max = np.max([max(polygon[:, 0]) for polygon in polygons])
    y_min = np.min([min(polygon[:, 1]) for polygon in polygons])
    y_max = np.max([max(polygon[:, 1]) for polygon in polygons])
    return x_max, x_min, y_max, y_min


if __name__ == '__main__':
    MyDebugger.pre_fix = os.path.join(MyDebugger.pre_fix, "debug")
    debugger = MyDebugger(["xuhao", "khh"])
    plotter = Plotter()
    for i in range(10):
        triangle = np.array([[0,0], [0,1], [1,0]])
        triangle2 = np.array([[0, 0], [0, -1], [1, 0]])
        plotter.draw_contours(debugger.file_path('layout.png'), [('blue', triangle), ('lightgreen', triangle2)])