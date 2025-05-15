import os
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse

from Utils import compute_KNN_graph
from Utils import buildPvalueGraph_FC,buildPvalueGraph_EC
from scipy.io import loadmat
from tqdm import tqdm

class ConnectivityData(InMemoryDataset):
    """ Dataset for the connectivity data."""

    def __init__(self,
                 root):
        super(ConnectivityData, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_paths = sorted(list(Path(self.raw_dir).glob("*.txt")))
        return [str(file_path.name) for file_path in file_paths]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def set_new_indices(self):
        self.__indices__ = list(range(self.len()))

    def process(self):
        PvalueGraph = buildPvalueGraph_FC()

        labels = loadmat(r"../../").get('')[0]
        data_list = []
        data = loadmat(r"../../").get('').T
        nrois = 116
        for i in tqdm(range(len(data))):
            connectivity = np.zeros((nrois, nrois), dtype=float)
            id = 0
            for row in range(116):
                for col in range(row + 1, 116):
                    connectivity[row][col] = data[i][id]
                    id = id + 1

            for row in range(116):
                for col in range(0, row):
                    connectivity[row][col] = connectivity[col][row]

            x = torch.from_numpy(connectivity).float()
            y = labels[i]
            y = torch.tensor([y]).long()
            adj = compute_KNN_graph(connectivity, PvalueGraph)
            adj = torch.from_numpy(adj).float()
            edge_index, edge_attr = dense_to_sparse(adj)
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


    # EC matrix construction

    # def process(self):
    #     PvalueGraph = buildPvalueGraph_EC()
    #
    #     labels = loadmat(r"../../").get('')[0]
    #     data_list = []
    #     data = loadmat(r"../../").get('')
    #     nrois = 116
    #     for i in tqdm(range(len(data))):
    #         connectivity = np.zeros((nrois, nrois), dtype=float)
    #         id = 0
    #         for row in range(116):
    #             for col in range(row + 1, 116):
    #                 connectivity[row][col] = data[i][id]
    #                 id = id + 1
    #
    #         for col in range(116):
    #             for row in range(col + 1, 116):
    #                 connectivity[row][col] = data[i][id]
    #                 id = id + 1
    #
    #         for row in range(116):
    #             connectivity[row][row] = 1
    #
    #         x = torch.from_numpy(connectivity).float()
    #         y = labels[i]
    #         y = torch.tensor([y]).long()
    #         adj = compute_KNN_graph(connectivity, PvalueGraph)
    #         adj = torch.from_numpy(adj).float()
    #         edge_index, edge_attr = dense_to_sparse(adj)
    #         data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    #
    #     data, slices = self.collate(data_list)
    #     torch.save((data, slices), self.processed_paths[0])
