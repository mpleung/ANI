import numpy as np, networkx as nx, math
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix, identity

def make_Zs(Y,ind1,ind0,pscores1,pscores0,subsample=False):
    """Generates vector of Z_i's, used to construct HT estimator.

    Parameters
    ----------
    Y : numpy float array
        n-dimensional outcome vector.
    ind1 : numpy boolean array
        n-dimensional vector of indicators for first exposure mapping.
    ind0 : numpy boolean array
        n-dimensional vector of indicators for second exposure mapping.
    pscores1 : numpy float array
        n-dimensional vector of probabilities of first exposure mapping for each unit.
    pscores0 : numpy float array
        n-dimensional vector of probabilities of second exposure mapping for each unit
    subsample : numpy boolean array
        When set to an object that's not a numpy array, the function will define subsample to be an n-dimensional array of ones, meaning it is assumed that all n units are included in the population. Otherwise, it must be an boolean array of the same dimension as Z where True components indicate population inclusion.

    Returns
    -------
    n-dimensional numpy float array, where entries corresponding to the True entries of subsample are equal to the desired Z's, and entries corresponding to False subsample entries are set to -1000.
    """
    if type(subsample) != np.ndarray: subsample = np.ones(Y.size, dtype=bool)
    i1 = ind1[subsample]
    i0 = ind0[subsample]
    ps1 = pscores1[subsample]
    ps0 = pscores0[subsample]
    weight1 = i1.copy().astype('float')
    weight0 = i0.copy().astype('float')
    weight1[weight1 == 1] = i1[weight1 == 1] / ps1[weight1 == 1]
    weight0[weight0 == 1] = i0[weight0 == 1] / ps0[weight0 == 1]
    Z = np.ones(Y.size) * (-1000) # filler entries that won't be used
    Z[subsample] = Y[subsample] * (weight1 - weight0)
    return Z

def network_SE(Zs, A, subsample=False, K=0, exp_nbhd=True, disp=False, b=-1):
    """Network-dependence robust standard errors.

    Returns our standard errors for the sample mean of each array in Zs.

    Parameters
    ----------
    Zs : a list of numpy float arrays
        Each array is n-dimensional.
    A : NetworkX undirected graph
        Graph on n nodes. NOTE: Assumes nodes are labeled 0 through n-1, so that the data for node i is given by the ith component of each array in Zs.
    subsample : numpy boolean array
        When set to an object that's not a numpy array, the function will define subsample to be an n-dimensional array of ones, meaning it is assumed that all n units are included in the population. Otherwise, it must be an boolean array of the same dimension as each array in Zs where True components indicate population inclusion.
    K : integer
        K used to define the K-neighborhood exposure mapping. 
    exp_nbhd : boolean
        Boolean for whether neighborhood growth is exponential (True) or polynomial (False). Used to determine recommended bandwidth. 
    b : float
        User-specified bandwidth. If a negative value is specified, function will compute our recommended bandwidth choice.
    disp : boolean
        Boolean for whether to also return more than just the SE (see below).

    Returns
    -------
    SE : float
        List of network-dependence robust standard error, one for each array of Zs.
    APL : float
        Average path length of A.
    b : int
        Bandwidth.
    PSD_failure : list of booleans
        True if substitute PSD variance estimator needed to be used for that component of Zs.
    """
    if type(Zs) == np.ndarray: 
        is_list = False
        Z_list = [Zs] # handle case where Zs is just an array
    else:
        is_list = True
        Z_list = Zs
    if type(subsample) != np.ndarray: 
        subsample = np.ones(Z_list[0].size, dtype=bool) # handle case where subsample is False

    n = subsample.sum()
    SEs = []
    PSD_failures = []
    if b == 0:
        for Z in Z_list:
            SEs.append(Z[subsample].std() / math.sqrt(subsample.sum())) # iid SE
            APL = 0
            PSD_failures.append(False)
    else:
        # compute path distances
        G = nx.to_scipy_sparse_matrix(A, nodelist=range(A.number_of_nodes()), format='csr')
        dist_matrix = dijkstra(csgraph=G, directed=False, unweighted=True)
        Gcc = [A.subgraph(c).copy() for c in sorted(nx.connected_components(A), key=len, reverse=True)]
        giant = [i for i in Gcc[0]] # set of nodes in giant component
        APL = dist_matrix[np.ix_(giant,giant)].sum() / len(giant) / (len(giant)-1) # average path length

        # default bandwidth
        if b < 0: 
            b = round(APL/2) if exp_nbhd else round(APL**(1/3)) # rec bandwidth
            b = max(2*K,b)

        weights = dist_matrix <= b # weight matrix
        for Z in Z_list:
            Zc = Z[subsample] - Z[subsample].mean() # demeaned data

            # default variance estimator (not guaranteed PSD)
            var_est = Zc.dot(weights[np.ix_(subsample,subsample)].dot(Zc[:,None])) / n 

            # PSD variance estimator from the older draft (Leung, 2019)
            if var_est <= 0:
                PSD_failures.append(True)
                if b < 0: b = round(APL/4) if exp_nbhd else round(APL**(1/3)) # rec bandwidth
                b = max(K,b)
                b_neighbors = dist_matrix <= b
                row_sums = np.squeeze(b_neighbors.dot(np.ones(Z.size)[:,None]))
                b_norm = b_neighbors / np.sqrt(row_sums)[:,None]
                weights = b_norm.dot(b_norm.T)
                var_est = Zc.dot(weights[np.ix_(subsample,subsample)].dot(Zc[:,None])) / n
            else:
                PSD_failures.append(False)

            SEs.append(math.sqrt(var_est / n))

    if disp:
        if is_list:
            return SEs, APL, b, PSD_failures
        else:
            return SEs[0], APL, b, PSD_failures
    else:
        if is_list:
            return SEs
        else:
            return SEs[0]

