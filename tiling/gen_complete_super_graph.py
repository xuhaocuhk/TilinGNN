import os
from util.debugger import MyDebugger
from interfaces.qt_plot import Plotter
from tiling.tile_graph import TileGraph, find_candidate_tile_locations, get_all_tiles
import tiling.tile_factory as factory
from inputs import config
from tiling.brick_layout import BrickLayout

def show_all_onering_neibors():
    # Driver code
    MyDebugger.pre_fix = os.path.join(MyDebugger.pre_fix, "debug")
    debugger = MyDebugger(f"gen_tile_graph_{config.env_name}", fix_rand_seed=0)
    plotter = Plotter()
    data_env = config.environment

    result_tiles = []
    align_tags = []
    for base_tile in data_env.proto_tiles:
        for align_tile in data_env.proto_tiles:
            neighbour_tiles, align_tag = get_all_tiles(base_tile, align_tile, integer_align=True)
            for i in range(len(neighbour_tiles)):
                plotter.draw_contours(debugger.file_path(f"{align_tag[i]}.png"),
                                      [base_tile.get_plot_attribute(),neighbour_tiles[i].get_plot_attribute()])



if __name__ == '__main__':
    # Driver code
    MyDebugger.pre_fix = os.path.join(MyDebugger.pre_fix, "debug")
    debugger = MyDebugger(f"gen_tile_graph_{config.env_name}", fix_rand_seed=0, save_print_to_file=False)
    plotter = Plotter()
    data_env = config.environment

    for num_rings in range(1, 3):
        file_name = "complete_graph_ring{}.pkl".format(num_rings)

        # generate complete brick_layouts
        tiles = find_candidate_tile_locations(num_rings=num_rings, base_tile=data_env.proto_tiles[0], align_tiles=data_env.proto_tiles)
        
        print(f"{len(tiles)}  tiles created!")
        graph = TileGraph(data_env.tile_count, tiles = tiles, one_hot = True, proto_tiles= data_env.proto_tiles)

        if not os.path.isdir(data_env.base_path):
            os.mkdir(data_env.base_path)
        graph.save_current_state(os.path.join(data_env.base_path, file_name))

        # load complete brick_layouts
        graph = TileGraph(data_env.tile_count)
        graph.load_graph_state(os.path.join(data_env.base_path, file_name))

        graph.show_complete_super_graph(plotter, debugger, f"super_graph_{num_rings}")

        print(f"done_{num_rings}")
