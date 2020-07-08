import os
import glob
from util.shape_processor import getSVGShapeAsNp
from tiling.tile import Tile
from tiling.tile_graph import TileGraph
from shapely.geometry import Polygon
import shapely

class Environment(object):

    # Using a folder to create graph and tile count
    # EXAMPLE IN ml_solver.py
    # getting the path:
    # assume the path contain a folder named tiles
    def __init__(self, base_path, symmetry_tiles = True):
        self.base_path = base_path
        tiles_dir = os.path.join(base_path, 'tiles')

        assert os.path.isdir(base_path), "Please set the working directory as the root directory of this project."
        assert os.path.isdir(tiles_dir)

        self.tiles_name = glob.glob(os.path.join(os.path.join(tiles_dir), "*.txt"))
        self.proto_tiles = [Tile(Polygon(getSVGShapeAsNp(tile_name)), id = i) for i, tile_name in enumerate(self.tiles_name)]
        if symmetry_tiles:
            symm_tiles = []
            for tile in self.proto_tiles:
                symm_tile_poly = shapely.affinity.affine_transform(tile.tile_poly, (-1, 0, 0, 1, 0, 0))  # notice: the shape become Counter-CLOCKWISE now
                symm_tile_poly = shapely.geometry.polygon.orient(symm_tile_poly, sign=-1.0)
                symm_tiles.append(Tile(symm_tile_poly, id= tile.id + len(self.tiles_name)))
            self.proto_tiles.extend(symm_tiles)
        self.complete_graph = None

        assert len(self.proto_tiles) > 0

        # tile count and graph can be got accordingly
        self.tile_count = len(self.proto_tiles)


    def load_complete_graph(self, ring_num: int):
        complete_graph_file = os.path.join(self.base_path, f'complete_graph_ring{ring_num}.pkl')
        assert os.path.exists(complete_graph_file)
        self.complete_graph = TileGraph(self.tile_count)
        self.complete_graph.load_graph_state(complete_graph_file)
        assert self.complete_graph.tile_type_count == self.tile_count

        num_of_nodes, num_of_adj_edges, num_of_collide_edges = self.complete_graph._get_graph_statistics()
        print(f"num_of_nodes : {num_of_nodes}")
        print(f"num_of_adj_edges : {num_of_adj_edges}")
        print(f"num_of_collide_edges : {num_of_collide_edges}")