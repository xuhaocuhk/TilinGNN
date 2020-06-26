from util.debugger import MyDebugger
from interfaces.qt_plot import Plotter
import os
from inputs import config
from solver.ml_solver.trainer import Trainer
import torch
from solver.ml_solver.ml_solver import ML_Solver
import numpy as np
from graph_networks.networks.TilinGNN import TilinGNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    MyDebugger.pre_fix = os.path.join(MyDebugger.pre_fix, "debug")
    debugger = MyDebugger(f"training_{config.experiment_id}", fix_rand_seed = config.rand_seed, save_print_to_file=False)
    plotter = Plotter()
    data_env = config.environment
    data_env.load_complete_graph(config.complete_graph_size)

    #### Network
    network = TilinGNN(adj_edge_features_dim=data_env.complete_graph.total_feature_dim, network_depth= config.network_depth, network_width=config.network_width).to(device)

    ## solver
    ml_solver = ML_Solver(debugger, device, data_env.complete_graph, network, num_prob_maps=1)

    #### Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    if not config.new_training:
        optimizer_stae_dict = torch.load("put your optimizer path here")
        optimizer.load_state_dict(optimizer_stae_dict)

    if not config.new_training:
        ml_solver.load_saved_network(config.load_trained_model_path)

    trainer = Trainer(debugger, plotter, device, ml_solver.network, data_path=config.dataset_path)

    trainer.train(ml_solver          = ml_solver,
                  optimizer          = optimizer,
                  batch_size         = config.batch_size,
                  training_epoch     = config.training_epoch,
                  save_model_per_epoch= config.save_model_per_epoch)

