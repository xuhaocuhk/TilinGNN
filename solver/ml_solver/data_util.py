import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import glob
import os
import torch
from inputs import config
from util.data_util import load_brick_layout_data, to_torch_tensor

class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        files_list = [os.path.basename(f) for f in glob.glob(os.path.join(self.raw_dir, "*.pkl"))]
        return files_list

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(i) for i in range(len(self.raw_file_names))]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        pass

    def process(self):
        print("processing the data...")
        for i, raw_path in enumerate(self.raw_paths):
            _, x, collide_edge_index, collide_edge_features, align_edge_index, align_edge_features, *_ = load_brick_layout_data(raw_path)

            x, align_edge_index, align_edge_features, collide_edge_index, collide_edge_features = \
                to_torch_tensor(torch.device('cpu'), x, align_edge_index, align_edge_features, collide_edge_index, collide_edge_features)

            ## dummy target
            y = torch.from_numpy(np.array([])).float().to(torch.device('cpu'))

            data = Data(x=x,
                        y=y,
                        collide_edge_index = collide_edge_index,
                        collide_edge_features = collide_edge_features,
                        edge_index=align_edge_index,
                        edge_features=align_edge_features)


            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))


    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
