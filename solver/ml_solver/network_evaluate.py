from util.debugger import MyDebugger
from interfaces.qt_plot import Plotter
import torch
from inputs import config
import os
from solver.ml_solver.ml_solver import ML_Solver
from tiling.tile_factory import solve_silhouette_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    MyDebugger.pre_fix = os.path.join(MyDebugger.pre_fix, "debug")
    debugger = MyDebugger(f"evaluation_{config.env_name}_{os.path.basename(config.silhouette_path)}", fix_rand_seed=config.rand_seed,
                          save_print_to_file=True)
    plotter = Plotter()
    environment = config.environment
    environment.load_complete_graph(config.complete_graph_size)

    solver = ML_Solver(debugger, device, environment.complete_graph, None, num_prob_maps = config.num_output_probs_maps)

    solver.load_saved_network(config.network_path, evaluation=False)

    average_score = solve_silhouette_dir(config.silhouette_path,
                         environment.complete_graph, solver, debugger = debugger, plotter = plotter)

    print("Evaluation finished!!!", flush = True)
    print(f"Average score:{average_score}", flush=True)