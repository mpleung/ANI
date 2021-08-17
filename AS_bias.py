import numpy as np, pandas as pd, networkx as nx, multiprocessing as mp, sys, traceback
from scipy.sparse.csgraph import dijkstra
from data_module import *
from DGP_module import *

processes = 16
B = 5000 
num_schools = [1,2]
AC_FX = True
if AC_FX:
    print('autocorrelated exposure effects\n')
else:
    print('independent exposure effects\n')

##### Task per node #####

def one_sim(b, deg_seq, eligibles, network_model):
    """
    Task to be parallelized: one simulation draw. 
    """
    n = deg_seq.size
    seed = int(n*(B*10) + b)
    np.random.seed(seed=seed)

    # simulate network data
    if network_model == 'configuration':
        A = nx.configuration_model(deg_seq, seed=seed)
        A = nx.Graph(A) # remove multi-edges
        A.remove_edges_from(nx.selfloop_edges(A)) # remove self-loops
        errors = np.random.normal(size=n)
    elif network_model == 'RGG':
        positions = np.random.uniform(size=(n,2))
        A = gen_RGG(positions, (deg_seq.mean()/ball_vol(2,1)/n)**(1/2))
        errors = np.random.normal(size=n) + (positions[:,0] - 0.5) 
    A_mat = nx.to_scipy_sparse_matrix(A, nodelist=range(n), format='csc')
    friends_eligible = np.squeeze(np.asarray(A_mat.dot(eligibles[:,None])))
    pop = (friends_eligible > 0) 

    # simulate potential outcomes
    Y0 = np.random.normal(size=n) # potential outcomes with no treated neighbors
    deg_seq = np.squeeze(A_mat.dot(np.ones(n)[:,None]))
    r,c = A_mat.nonzero()
    rD_sp = csr_matrix(((1.0/np.maximum(deg_seq,1))[r], (r,c)), shape=(A_mat.shape))
    A_norm = A_mat.multiply(rD_sp) # row-normalized adjacency matrix
    Y0 += np.squeeze(A_mat.dot(Y0[:,None]))
    beta = np.random.normal(1,1,size=n)
    if AC_FX: 
        deg_seq = np.squeeze(A_mat.dot(np.ones(n)[:,None]))
        r,c = A_mat.nonzero()
        rD_sp = csr_matrix(((1.0/np.maximum(deg_seq,1))[r], (r,c)), shape=(A_mat.shape))
        A_norm = A_mat.multiply(rD_sp) 
        beta += np.squeeze(A_mat.dot(beta[:,None]))
    Y1 = beta + Y0 # potential outcomes with some treated neighbor
    Y0 = Y0[pop]
    Y1 = Y1[pop]
    
    tmp = A_mat.multiply(eligibles)
    tmp2 = np.squeeze(np.array(tmp.sum(axis=1)))
    tmp3 = tmp.dot(tmp)
    Eij = ((tmp3 > 0).multiply(tmp3 == tmp2)).astype('int') # n x n matrix of 1\{\mathcal{E}_{ij}\}
    Eij.setdiag(0) # zero out diagonals (source of warning message, ignore)
    Eij = Eij.todense()[np.ix_(pop,pop)]
    AS_bias = np.power(Y1-Y0,2).mean() \
            + ( np.power(Y1,2).dot(Eij.dot(np.ones(pop.sum())[:,None])) \
            + np.ones(pop.sum()).dot(Eij.dot(np.power(Y0,2)[:,None])) \
            - 2 * Y1.dot(Eij.dot(Y0[:,None])) ) / pop.sum() 

    dist_matrix = dijkstra(csgraph=A_mat, directed=False, unweighted=True)
    G_my = dist_matrix <= 2
    G_my = G_my[np.ix_(pop,pop)]
    ATE = (Y1-Y0).mean()
    my_bias = (Y1-Y0).var() + (Y1-Y0-ATE).dot(G_my.dot((Y1-Y0-ATE)[:,None])) / pop.sum()

    return [AS_bias, my_bias, pop.sum()]

##### Main #####

_,D,A,_,_,IDs = assemble_data()
deg_seq = np.array([i[1] for i in A.out_degree])
A = A.to_undirected()
eligibles = (D >= 0)

for network_model in ['configuration','RGG']:
    print(network_model + ' model')

    for i,ns in enumerate(num_schools):
        # select schools
        if ns == 1:
            students = (IDs[:,1] == 24)
        elif ns == 2:
            students = (IDs[:,1] == 24) + (IDs[:,1] == 22)
        else:
            students = (IDs[:,1] == 24) + (IDs[:,1] == 22) + (IDs[:,1] == 60) + (IDs[:,1] == 56)
        print('  {} schools, n = {}.'.format(ns,students.sum()))
        sys.stdout.flush()

        if deg_seq[students].sum() % 2 != 0: 
            deg_seq_pop = deg_seq[students].copy()
            deg_seq_pop[0] += 1 # need even total degree for configuration model
        else:
            deg_seq_pop = deg_seq[students]

        # run simulations
        def one_sim_wrapper(b):
            try:
                return one_sim(b, deg_seq_pop, eligibles[students], network_model)
            except:
                print('%s: %s' % (b, traceback.format_exc()))
                sys.stdout.flush()
        pool = mp.Pool(processes=processes, maxtasksperchild=1)
        parallel_output = pool.imap(one_sim_wrapper, range(B), chunksize=25)
        pool.close()
        pool.join()
        results = np.array([r for r in parallel_output]).mean(axis=0)

        print('  AS bias = {}. our bias = {}. n = {}'.format(results[0],results[1],results[2]))

