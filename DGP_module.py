import numpy as np, networkx as nx, math
from scipy import spatial
from scipy.special import gamma as GammaF

def ball_vol(d,r):
    """Computes the volume of a d-dimensional ball of radius r. Used to construct RGG.

    Parameters
    ----------
    d : int 
        Dimension of space.
    r : float
        RGG parameter.
    """
    return math.pi**(d/2) * r**d / GammaF(d/2+1)

def gen_RGG(positions, r):
    """Generates an RGG.

    Parameters
    ----------
    positions : numpy array
        n x d array of d-dimensional positions, one for each of the n nodes.
    r : float
        RGG parameter.

    Returns
    -------
    RGG as NetworkX graph
    """
    kdtree = spatial.cKDTree(positions)
    pairs = kdtree.query_pairs(r) # default is Euclidean norm
    RGG = nx.empty_graph(n=positions.shape[0], create_using=nx.Graph())
    RGG.add_edges_from(list(pairs))
    return RGG

def linear_in_means(D, A_norm, LIM_inv, errors, theta):
    """Generates outcomes from the linear-in-means model.

    Parameters
    ----------
    D : numpy array
        n-dimensional vector of treatment indicators.
    A_norm : scipy sparse matrix (csr format)
        Row-normalized adjacency matrix.
    LIM_inv : scipy sparse matrix
        Leontief inverse.
    errors : numpy array
        n-dimensional array of error terms
    theta : numpy array
        Vector of structural parameters: intercept, endogenous peer effect, exogenous peer effect, treatment effect.

    Returns
    -------
    n-dimensional array of outcomes
    """
    Y = LIM_inv.dot( (theta[0] + theta[2]*np.squeeze(np.asarray(A_norm.dot(D[:,None]))) + theta[3]*D + errors)[:,None] )
    return np.squeeze(np.asarray(Y))

def threshold_model(D, A_norm, errors, theta):
    """Generates outcomes from the complex contagion model.

    Parameters
    ----------
    D : numpy array
        n-dimensional vector of treatment indicators.
    A_norm : scipy sparse matrix (csr format)
        Row-normalized adjacency matrix.
    errors : numpy array
        n-dimensional array of error terms
    theta : numpy array
        Vector of structural parameters: intercept, endogenous peer effect, exogenous peer effect, treatment effect.

    Returns
    -------
    n-dimensional array of outcomes
    """
    if theta[1] < 0:
        raise ValueError('Must have theta[1] >= 0.')

    U_exo_eps = theta[0] + theta[2]*np.squeeze(np.asarray(A_norm.dot(D[:,None]))) + theta[3]*D + errors

    # set initial outcome to 1 iff the agent will always choose outcome 1
    Y = (U_exo_eps > 0).astype('float')

    stable = False
    while stable == False:
        peer_avg = np.squeeze(np.asarray(A_norm.dot(Y[:,None])))
        Y_new = (U_exo_eps + theta[1]*peer_avg > 0).astype('float') # best response
        if (Y_new == Y).sum() == D.size:
            stable = True
        else:
            Y = Y_new

    return Y_new

