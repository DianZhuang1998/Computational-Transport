import numpy as np

def cluster_OrachardBouman(S, w, minVar):
    # Initially, all measurements are one cluster
    C1 = {'X': S, 'w': w}
    C1 = calc(C1)
    nodes = [C1]

    while max([node['lambda'] for node in nodes]) > minVar:
        nodes = split(nodes)
    print(len(nodes))
    mu = np.zeros((3, len(nodes)))
    Sigma = np.zeros((3, 3, len(nodes)))

    for i, node in enumerate(nodes):
        mu[:, i] = node['q']
        Sigma[:, :, i] = node['R']

    return mu, Sigma

# Calculates cluster statistics
def calc(C):
    W = np.sum(C['w'])
    # Weighted mean
    C['q'] = np.sum(np.tile(C['w'], (1, C['X'].shape[1])) * C['X'], axis=0) / W
    # Weighted covariance
    t = (C['X'] - np.tile(C['q'], (C['X'].shape[0], 1))) * np.tile(np.sqrt(C['w']), (1, C['X'].shape[1]))
    C['R'] = np.dot(t.T, t) / W + 1e-5 * np.eye(3)

    C['wtse'] = np.sum((C['X'] - np.tile(C['q'], (C['X'].shape[0], 1))) ** 2)
    

    D, V = np.linalg.eig(C['R'])
    
    C['e'] = V[:, 0]
    C['lambda'] = D[0]

    return C

# Splits maximal eigenvalue node in direction of maximal variance
def split(nodes):
    x, i = max([(node['lambda'], idx) for idx, node in enumerate(nodes)])
    Ci = nodes[i]
    idx = np.dot(Ci['X'], Ci['e']) <= np.dot(Ci['q'], Ci['e'])
    Ca = {'X': Ci['X'][idx, :], 'w': Ci['w'][idx]}
    Cb = {'X': Ci['X'][~idx, :], 'w': Ci['w'][~idx]}
    Ca = calc(Ca)
    Cb = calc(Cb)
    nodes.pop(i) # Remove the i'th node and replace it with its children
    nodes.extend([Ca, Cb])
    return nodes


#%%
from scipy.io import loadmat

data = loadmat('data.mat')
S = data['S']
w = data['w']
minVar = data['minVar']

mu, sigma = cluster_OrachardBouman(S, w, minVar)