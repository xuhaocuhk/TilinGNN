from minizinc import Solver, Model, Instance
from util.debugger import MyDebugger
from interfaces.qt_plot import Plotter
from tiling.brick_layout import BrickLayout
from util.shape_processor import getSVGShapeAsNp
from tiling.tile_graph import TileGraph
import os
import time
from solver.base_solver import BaseSolver
from solver.minizinc_solver.data_util import convert_layout_to_dzn_features
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
    MyDebugger.pre_fix = os.path.join(MyDebugger.pre_fix, "debug")
    debugger = MyDebugger("completely_unsupervised", fix_rand_seed=2)
    plotter = Plotter()
    data_env = config.environment
    data_env.load_complete_graph(5)

    for num_piece in range(2, 100):
        for case_num in range(5):
            node_feature, edge_index, edge_feature, gt, re_index = factory.gen_one_train_data(plotter, graph, low=num_piece, high=num_piece+1)
            brick_layout = BrickLayout(debugger, graph, node_feature, edge_index, edge_feature, gt, re_index)

            # initialize the solver
            solver = MinizincSolver(model_file = './solve_contour_multiTile.mzn',
                                    solver_type = 'coin-bc',
                                    debugger = debugger)

            # Solve the problem
            result_brick_layout, exec_time = solver.solve(brick_layout)
            brick_layout.show_complete_graph(plotter, f"{num_piece}_{case_num}_complete_graph.png")
            brick_layout.show_tiles(plotter, f"{num_piece}_{case_num}_GT.png", draw_ground_truth = True)
            brick_layout.show_tiles(plotter, f"{num_piece}_{case_num}_predict_{exec_time}.png", draw_ground_truth =  False)
            brick_layout.show_candidate_tiles(plotter, f"{num_piece}_{case_num}_supper_graph.png")
            brick_layout.show_super_contour(plotter, f"{num_piece}_{case_num}_n{len(node_feature)}_super_contour.png")

