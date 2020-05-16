from tiling.brick_layout import BrickLayout
import time
from solver.base_solver import BaseSolver
from solver.minizinc_solver.data_util import convert_layout_to_dzn_features
import gurobipy
from gurobipy import GRB
import numpy as np

class GurobiSolver(BaseSolver):

    def __init__(self, debugger):
        super(GurobiSolver, self).__init__()
        self.debugger = debugger

    def solve(self, brick_layout : BrickLayout, verbose = True, intermediate_solutions = False, timeout = None, recompute_graph = False):
        if verbose:
            print(f"start solving by gurobi native ...")
        start_time = time.time()
        model = gurobipy.Model()

        # add data to the solver
        model = self.add_constraint_for_model(brick_layout, model, recompute_graph= recompute_graph)

        model.optimize()

        if verbose:
            print(f"solve finished in {time.time() - start_time}")
        for idx, var in enumerate(model.getVars()):
            brick_layout.predict[idx] = model.getVars()[idx].X

        return brick_layout, time.time() - start_time

    def add_constraint_for_model(self, brick_layout, model, recompute_graph = False):

        # get the features needed
        num_edge_collision, num_edge_adjacent, collision_edges, adjacent_edge, align_edge_lengths = convert_layout_to_dzn_features(
            brick_layout, recompute_graph = recompute_graph)
        node_area = [ float( brick_layout.complete_graph.tiles[brick_layout.inverse_index[i]].tile_poly.area )
                                           for i in range(len(brick_layout.node_feature))]
        nodes = model.addMVar(len(brick_layout.node_feature), vtype = GRB.BINARY, name = 'nodes')
        for i in range(collision_edges.shape[1]):
            model.addConstr(nodes[collision_edges[0,i]] + nodes[collision_edges[1,i]] <= 1, f'c{i}')

        model.setObjective(sum([nodes[i] * node_area[i] for i in range(len(brick_layout.node_feature))]), GRB.MAXIMIZE)

        return model

if __name__ == '__main__':
    pass

