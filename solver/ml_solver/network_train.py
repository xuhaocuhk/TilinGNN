from util.debugger import MyDebugger
from interfaces.qt_plot import Plotter
import os
from inputs import config
from solver.ml_solver.trainer import Trainer
import torch
from solver.ml_solver.ml_solver import ML_Solver
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    ###### GET thee network needed ###############
    mod = __import__(config.EDWNET_PREFIX + config.EDWNET_NAME, fromlist =['EDWNet'])
    EDWNet = getattr(mod, 'EDWNet')

    MyDebugger.pre_fix = os.path.join(os.path.join(MyDebugger.pre_fix, "training"), "running")
    debugger = MyDebugger(f"training_{config.debug_id}_{config.EDWNET_NAME}_{config.LOSSES_NAME}", fix_rand_seed = config.rand_seed, save_print_to_file=False)
    plotter = Plotter()
    data_env = config.environment
    data_env.load_complete_graph(config.complete_graph_size)

    EDWNet = EDWNet(adj_edge_features_dim=data_env.complete_graph.total_feature_dim,
                        col_edge_feature_dim=1,
                        output_dim=config.num_output_probs_maps).to(device) if config.new_training else None

    ml_solver = ML_Solver(debugger, device, data_env.complete_graph, EDWNet,
                          num_prob_maps=config.num_output_probs_maps)

    if not config.new_training:
        ml_solver.load_saved_network(
            config.load_trained_model_path,
            evaluation=False)

    trainer = Trainer(debugger, plotter, device, ml_solver.network, data_path=config.dataset_path)

    trainer.train(ml_solver          = ml_solver,
                  lr                 = config.learning_rate,
                  batch_size         = config.batch_size,
                  training_epoch     = config.training_epoch,
                  model_saving_epoch = config.model_saving_epoch,
                  new_training       = config.new_training,
                  optimizer_path     = config.optimizer_path)

