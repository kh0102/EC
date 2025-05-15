import numpy as np
from torch.utils.data import DataLoader
from scipy.io import loadmat


class MddData(DataLoader):

    def __init__(self):
        super(MddData).__init__()
        self.nrois = 116
        self.total_subjects = 1611
        self.fc = np.load("../../")
        self.ec = np.load("../../")
        self.labels = loadmat("../../").get('label')[0]
        self.fc = np.expand_dims(self.fc, 1)
        self.ec = np.expand_dims(self.ec,1)

    def __len__(self):
        return self.total_subjects

    def __getitem__(self, index):
        return self.fc[index],self.ec[index], self.labels[index]

    def __getallitems__(self):
        return self.fc,self.ec,self.labels


if __name__ == '__main__':
    dataset = MddData()
