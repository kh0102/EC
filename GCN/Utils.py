import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import loadmat
from scipy import stats
from tqdm import tqdm

# def compute_KNN_graph(matrix, k_degree=5):
#     """ Calculate the adjacency matrix from the connectivity matrix."""
#
#     matrix = np.abs(matrix)
#     idx = np.argsort(-matrix)[:, 0:k_degree]
#     matrix.sort()
#     matrix = matrix[:, ::-1]
#     matrix = matrix[:, 0:k_degree]
#
#     A = adjacency(matrix, idx).astype(np.float32)
#
#     return A


def adjacency(dist, idx):

    m, k = dist.shape
    assert m, k == idx.shape
    dist = np.abs(dist)
    assert dist.min() >= 0

    # Weight matrix.
    I = np.arange(0, m).repeat(k)
    J = idx.reshape(m * k)
    V = dist.reshape(m * k)
    W = coo_matrix((V, (I, J)), shape=(m, m))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    return W.todense()

def compute_KNN_graph(matrix, PvalueGraph, k_degree=10):
    """ Calculate the adjacency matrix from the connectivity matrix."""

    # PvalueGraph = np.abs(PvalueGraph)
    idx = np.argsort(PvalueGraph)[:, 0:k_degree]
    #matrix.sort()
    #matrix = matrix[:, ::-1]
    #matrix = matrix[:, 0:k_degree]
    matrix_new = np.zeros((116, k_degree), dtype=float)
    for i in range(116):
        for j in range(k_degree):
            matrix_new[i][j] = matrix[i][idx[i][j]]

    A = adjacency(matrix_new, idx).astype(np.float32)

    return A

def buildPvalueGraph_FC():

    labels = loadmat(r"../../").get('label')[0]
    data = loadmat(r"../.").get('').T
    nrois = 116

    Parray = []

    for j in tqdm(range(6670)):
        feat = data[:, j]
        # stats.levene(feat, target)
        data1 = []
        data0 = []
        for i in range(1611):
            if(labels[i] == 1):
                data1.append(feat[i])
            else:
                data0.append(feat[i])

        lev = stats.levene(data1, data0)
        if (lev.pvalue < 0.05):
            ttest = stats.ttest_ind(data1, data0, equal_var=False)
        else:
            ttest = stats.ttest_ind(data1, data0, equal_var=True)

        Parray.append(ttest.pvalue)

    connectivity = np.zeros((nrois, nrois), dtype=float)
    id = 0
    for row in range(116):
        for col in range(row + 1, 116):
            connectivity[row][col] = Parray[id]
            id = id + 1

    for row in range(116):
        for col in range(0, row):
            connectivity[row][col] = connectivity[col][row]

    for row in range(116):
        connectivity[row][row] = 10

    return connectivity

def buildPvalueGraph_EC():

    labels = loadmat(r"../..").get('')[0]
    data = loadmat(r"../../").get('')
    nrois = 116

    Parray = []


    for j in tqdm(range(6670*2)):
        feat = data[:, j]
        # stats.levene(feat, target)
        data1 = []
        data0 = []
        for i in range(1611):
            if(labels[i] == 1):
                data1.append(feat[i])
            else:
                data0.append(feat[i])

        lev = stats.levene(data1, data0)
        if (lev.pvalue < 0.05):
            ttest = stats.ttest_ind(data1, data0, equal_var=False)
        else:
            ttest = stats.ttest_ind(data1, data0, equal_var=True)

        Parray.append(ttest.pvalue)

    connectivity = np.zeros((nrois, nrois), dtype=float)
    id = 0
    for row in range(116):
        for col in range(row + 1, 116):
            connectivity[row][col] = Parray[id]
            id = id + 1

    for col in range(116):
        for row in range(col + 1, 116):
            connectivity[row][col] = Parray[id]
            id = id + 1

    for row in range(116):
        connectivity[row][row] = 10

    return connectivity
