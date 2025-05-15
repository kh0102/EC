
import numpy as np
from torch.utils.data import DataLoader


class Mdd_EC(DataLoader):
    def __init__(self, Mdd_EC_npz_file_path):
        super(Mdd_EC).__init__()
        self.ecs = np.load(Mdd_EC_npz_file_path)
        self.num_sub = len(self.ecs)
        self.labels = np.zeros((self.num_sub), dtype=int)

        index = 0
        for i in range(832):
            self.labels[index] = 1
            index = index+1
        for i in range(779):
            self.labels[index] = 0
            index = index+1

        self.labels = np.eye(2)[self.labels]

    def __len__(self):
        return len(self.ecs)

    def __getitem__(self, index):
        return self.ecs[index], self.labels[index]

    def __getallitems__(self):
        return self.ecs, self.labels


class Mdd_FC(DataLoader):
    def __init__(self, Mdd_FC_npz_file_path):
        super(Mdd_FC).__init__()
        self.fcs = np.load(Mdd_FC_npz_file_path)
        self.num_sub = len(self.fcs)
        self.labels = np.zeros((self.num_sub), dtype=int)

        index = 0
        for i in range(832):
            self.labels[index] = 1
            index = index+1

        for i in range(779):
            self.labels[index] = 0
            index = index+1

        self.labels = np.eye(2)[self.labels]

    def __len__(self):
        return len(self.fcs)

    def __getitem__(self, index):
        return self.fcs[index], self.labels[index]

    def __getallitems__(self):
        return self.fcs, self.labels


