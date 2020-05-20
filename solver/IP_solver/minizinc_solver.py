from minizinc import Solver, Model, Instance
from util.debugger import MyDebugger
from interfaces.qt_plot import Plotter
from tiling.brick_layout import BrickLayout
from util.shape_processor import getSVGShapeAsNp
from tiling.tile_graph import TileGraph
import os
import time
from solver.base_solver import BaseSolver
from solver.IP_solver.data_util import convert_layout_to_dzn_features
from inputs import config
import numpy as np

class MinizincSolver(BaseSolver):

    def __init__(self, model_file, solver_type, debugger):
        super(MinizincSolver, self).__init__()
        self.debugger = debugger
        self.solver_type = solver_type
        self.solver = Solver.lookup(solver_type)
        self.model_file = model_file

    def solve(self, brick_layout : BrickLayout, verbose = True, intermediate_solutions = False, timeout = None, recompute_graph = False):
        if verbose:
            print(f"start solving by {self.solver_type} ...")
        start_time = time.time()
        model = Model()
        model.add_file(self.model_file)
        instance = Instance(self.solver, model)

        # add data to the solver
        instance = self.add_data_to_model(brick_layout, instance, recompute_graph= recompute_graph)

        result = instance.solve(intermediate_solutions = intermediate_solutions,
                                timeout = timeout)

        if result.status.has_solution():
            if intermediate_solutions:
                selected_nodes = [1 if selected else 0 for selected in result.solution[-1].node]
            else:
                selected_nodes = [1 if selected else 0 for selected in result['node']]
        else:
            selected_nodes = np.zeros(brick_layout.node_feature.shape[0])
            print("No solutions are found")

        if verbose:
            print(f"solve finished in {time.time() - start_time}")
        brick_layout.predict = selected_nodes
        return brick_layout, time.time() - start_time

    def add_data_to_model(self, brick_layout, instance, recompute_graph = False):

        # get the features needed
        num_edge_collision, num_edge_adjacent, collision_edges, adjacent_edge, align_edge_lengths = convert_layout_to_dzn_features(
            brick_layout, recompute_graph = recompute_graph)

        instance["nums_node"]           = len(brick_layout.node_feature)
        instance["nums_edge_collision"] = int(num_edge_collision)
        instance["nums_edge_adjacent"]  = int(num_edge_adjacent)
        instance["from_adjacent"]       = [int(edge + 1) for edge in adjacent_edge[0, :]]   if len(adjacent_edge.shape) > 1 else []
        instance["to_adjacent"]         = [int(edge + 1) for edge in adjacent_edge[1, :]]   if len(adjacent_edge.shape) > 1 else []
        instance["from_collision"]      = [int(edge + 1) for edge in collision_edges[0, :]] if len(collision_edges.shape) > 1 else []
        instance["to_collision"]        = [int(edge + 1) for edge in collision_edges[1, :]] if len(collision_edges.shape) > 1 else []
        instance["node_area"]           = [ float( brick_layout.complete_graph.tiles[brick_layout.inverse_index[i]].tile_poly.area )
                                           for i in range(len(brick_layout.node_feature))]
        instance["node_perimeter"]      = [ float( brick_layout.complete_graph.tiles[brick_layout.inverse_index[i]].get_perimeter() )
                                           for i in range(len(brick_layout.node_feature))]
        instance["align_length"]        = [float(l) for l in align_edge_lengths]
        instance["contour_area"]        = float(brick_layout.get_super_contour_poly().area)


        return instance

if __name__ == '__main__':
    # TODO: add an example of how to use IP solver here
    psss
