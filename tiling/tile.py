from shapely.geometry import Polygon
import numpy as np
from util.algo_util import _distance

ESP = 1e-5

class Tile:
    def __init__(self, tile_poly: Polygon, id: int):
        self.tile_poly = tile_poly
        self.id = id

    def __eq__(self, other):
        if abs(self.tile_poly.area - other.tile_poly.area) > ESP:
            return False

        if self.tile_poly.almost_equals(other.tile_poly, decimal=5):
            return True
        else:
            return abs(self.tile_poly.intersection(other.tile_poly).area - self.tile_poly.area) < ESP

    def get_plot_attribute(self, style = "blue"):
        return (style, np.array(list(self.tile_poly.exterior.coords)))

    def get_edge(self, edge_idx):
        contour_segments = list(self.tile_poly.exterior.coords)
        contour_edge_p0, contour_edge_p1 = contour_segments[edge_idx], contour_segments[edge_idx + 1]
        return np.array(contour_edge_p0), np.array(contour_edge_p1)

    def get_edge_length(self, edge_idx):
        p0, p1 = self.get_edge(edge_idx)
        return _distance(p0, p1)

    def get_edge_num(self):
        contour_segments = list(self.tile_poly.exterior.coords)
        return len(contour_segments)-1

    def area(self):
        assert self.tile_poly.area > ESP
        return self.tile_poly.area

    def get_perimeter(self):
        return np.sum([self.get_edge_length(i) for i in range(self.get_edge_num())])

    def get_align_point(self, edge_idx: int, edge_shift: float):
        perimeter = self.get_perimeter()
        accumelate_dist = 0.0
        for i in range(edge_idx):
            accumelate_dist = accumelate_dist + self.get_edge_length(i)
        accumelate_dist = accumelate_dist + edge_shift
        return accumelate_dist / perimeter

if __name__ == '__main__':
    triangle = np.array([[0,0], [0,0.9], [1,0]])
    triangle2 = np.array([[0, 0], [0, 0.9], [1, 0]])
    triangle2 = np.array([[0, 0], [0, 0.9], [1, 0]])
    poly1 = Polygon(triangle)
    poly2 = Polygon(triangle2)
    tiles = [Tile(poly1) , Tile(poly2)]


    print(new_tiles)