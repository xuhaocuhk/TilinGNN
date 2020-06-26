from tiling.brick_layout import BrickLayout
import os
import torch
import numpy as np
from solver.ml_solver.data_util import GraphDataset
from util.data_util import write_brick_layout_data, load_brick_layout_data
from util.algo_util import append_text_to_file
import tiling.tile_factory as factory
from torch_geometric.data import DataLoader
import time
import glob
import asyncio, concurrent.futures
import multiprocessing as mp
import inputs.config as config
import traceback
from graph_networks.network_utils import get_network_prediction
from util.shape_processor import load_polygons
from solver.ml_solver.losses import Losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self, debugger, plotter, device, network, data_path):
        self.debugger = debugger
        self.plotter = plotter
        self.device = device
        self.model_save_path = self.debugger.file_path(self.debugger.file_path("model"))
        self.data_path = data_path
        self.training_path = os.path.join(data_path, 'train')
        self.testing_path = os.path.join(data_path, 'test')
        self.network = network

        # creation of directory for result
        os.mkdir(self.debugger.file_path('model'))
        os.mkdir(self.debugger.file_path('result'))



    def create_data(self, complete_graph, low=0.4, high=0.8, max_vertices=10, testing_ratio=0.2, number_of_data=20000):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        if not os.path.exists(self.training_path):
            os.makedirs(self.training_path)
        if not os.path.exists(self.testing_path):
            os.makedirs(self.testing_path)
        # create training data
        self._create_data(self.plotter, complete_graph, self.training_path, number_of_data, low, high, max_vertices)
        # create testing data
        self._create_data(self.plotter, complete_graph, self.testing_path, int(number_of_data * testing_ratio), low, high, max_vertices)


    def train(self,
              ml_solver,
              optimizer,
              batch_size=32,
              training_epoch=10000,
              save_model_per_epoch=5):

        dataset_train = GraphDataset(root=self.training_path)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataset_test = GraphDataset(root=self.testing_path)
        loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

        print("Training Start!!!", flush=True)
        min_test_loss = float("inf")
        for i in range(training_epoch):
            self.network.train()
            for batch in loader_train:
                ## get prediction
                data = batch.to(self.device)
                probs = get_network_prediction(network = self.network,
                                               x = data.x,
                                               adj_e_index=data.edge_index,
                                               adj_e_features=data.edge_features,
                                               col_e_idx=data.collide_edge_index,
                                               col_e_features=None)

                try:
                    optimizer.zero_grad()
                    train_loss, *_ = Losses.calculate_unsupervised_loss(probs, data.x, data.collide_edge_index,
                                                                                    adj_edges_index=data.edge_index,
                                                                                    adj_edge_features=data.edge_features)
                    train_loss.backward()
                    optimizer.step()
                except:
                    print(traceback.format_exc())
                    continue

            # self.network.train()
            torch.cuda.empty_cache()
            loss_train, *_ = Losses.cal_avg_loss(self.network, loader_train)
            print(f"epoch {i}: training loss: {loss_train}", flush=True)
            loss_test, avg_collision_probs, avg_filled_area, avg_align_length  = Losses.cal_avg_loss(self.network, loader_test)
            print(f"epoch {i}: testing loss: {loss_test}", flush=True)


            ############# result debugging #############
            if (loss_test < min_test_loss or i % save_model_per_epoch == 0):
                if loss_test < min_test_loss:
                    min_test_loss = loss_test
                torch.cuda.empty_cache()
                ############# network testing #############
                # self.network.train()


                torch.save(self.network.state_dict(), os.path.join(self.model_save_path, f'model_{i}_{loss_test}.pth'))
                torch.save(optimizer.state_dict(), os.path.join(self.model_save_path, f'optimizer_{i}_{loss_test}.pth'))
                print(f"model saved at epoch {i}")

                ml_solver.load_saved_network(os.path.join(self.model_save_path, f'model_{i}_{loss_test}.pth'))

                torch.cuda.empty_cache()

                train_sample_data = np.random.randint(low=0, high=len(dataset_train), size=config.debug_data_num)
                test_sample_data = np.random.randint(low=0,  high=len(dataset_test),  size=config.debug_data_num)

                ###### evaluate with training mode
                ml_solver.save_debug_info(self.plotter, train_sample_data, os.path.join(self.training_path, "raw"),
                                          os.path.join("result", os.path.join(f"epoch_{i}", 'training')))
                ml_solver.save_debug_info(self.plotter, test_sample_data, os.path.join(self.testing_path, "raw"),
                                          os.path.join("result", os.path.join(f"epoch_{i}", 'testing')))
                torch.cuda.empty_cache()

        print("Training Done!!!")

    def _create_data(self, plotter, graph, data_path, number_of_data, low, high, max_vertices,
                     workers = 16):
        start_time = time.time()
        print(f"generating data to {data_path}")
        if not os.path.isdir(os.path.join(data_path, 'raw')):
            os.mkdir(os.path.join(data_path, 'raw'))

        loop = asyncio.get_event_loop()

        # put poisitonal args in order of _create_one_data

        pool = mp.Pool(workers)
        pool.map(Trainer._create_one_data, [ (graph, data_path, low, high, max_vertices, i) for i in range(number_of_data) ])

        print("Time used: %s" % (time.time() - start_time))

    @staticmethod
    def _create_one_data(data):
        graph, data_path, low, high, max_vertices, index = data
        node_feature, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features,\
        re_index = factory.generate_random_inputs(graph, max_vertices, low=low, high=high)
        # write the file to a pkl file
        assert len(collide_edge_index) > 0
        assert len(align_edge_index) > 0

        write_brick_layout_data(save_path='raw/data_{}.pkl'.format(index),
                                node_features=node_feature,
                                collide_edge_index=collide_edge_index,
                                collide_edge_features=collide_edge_features,
                                align_edge_index=align_edge_index,
                                align_edge_features=align_edge_features,
                                re_index=re_index,
                                prefix=data_path,
                                predict=None,
                                predict_order=None,
                                predict_probs=None
                                )

        if index % 10 == 0:
            print(f"{index} data generated")
