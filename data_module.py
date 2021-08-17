import numpy as np, networkx as nx, pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import hypergeom

def node_stats(Y, D):
    """Computes unit-level summary statistics.

    Parameters
    ----------
    Y : numpy array
        Vector of observations.
    D : numpy array
        Vector of treatment indicators.
    """
    table = pd.DataFrame([[Y.mean(),Y.std()], [D.mean(),D.std()]])
    table.index = ['Outcome', 'Exposure Mapping']
    table.columns = ['Mean', 'SD']
    print('\n\n\\begin{table}[ht]')
    print('\centering')
    print('\caption{Summary Statistics}')
    print('\\begin{threeparttable}')
    print(table.to_latex(float_format = lambda x: '%.2f' % x, header=True, escape=False))
    print('\\begin{tablenotes}[para,flushleft]')
    print("\\footnotesize $n= {}$.".format(Y.size))
    print('\end{tablenotes}')
    print('\end{threeparttable}')
    print('\end{table}')

def network_stats(A, IDs, school_IDs):
    """Computes network summary statistics, averaged across schools.

    Parameters
    ----------
    A : NetworkX graph
    IDs : numpy array
        Student identifiers unit.
    school_IDs : numpy array
        School identifiers for each unit.
    """
    numnodes = np.zeros(len(school_IDs))
    numedges = np.zeros(len(school_IDs))
    clustering = np.zeros(len(school_IDs))
    diam = np.zeros(len(school_IDs))
    APL = np.zeros(len(school_IDs))
    giant_size = np.zeros(len(school_IDs))
    ccount = np.zeros(len(school_IDs))
    num_isos = np.zeros(len(school_IDs))
    max_deg = np.zeros(len(school_IDs))
    avg_deg = np.zeros(len(school_IDs))

    for i,cluster in enumerate(school_IDs):
        if len(school_IDs) > 1:
            G = A.subgraph(np.arange(IDs.shape[0])[IDs[:,1]==cluster])
        else:
            G = A

        numnodes[i] = G.number_of_nodes()
        numedges[i] = G.number_of_edges()

        clustering[i] = nx.average_clustering(G)

        Gcc = [A.subgraph(c).copy() for c in sorted(nx.connected_components(A), key=len, reverse=True)]
        giant = Gcc[0].to_undirected()
        diam[i] = nx.diameter(giant)
        APL[i] = nx.average_shortest_path_length(giant)
        giant_size[i] = len(giant) # largest component size
        ccount[i] = len(Gcc)

        deg_seq = np.array([i[1] for i in G.degree]) # degree sequence
        num_isos[i] = (deg_seq == 0).sum()
        max_deg[i] = deg_seq.max()
        avg_deg[i] = deg_seq.mean()
        
    table = pd.DataFrame(np.array([numnodes.mean(), numedges.mean(), avg_deg.mean(), max_deg.mean(), num_isos.mean(), giant_size.mean(), diam.mean(), APL.mean(), ccount.mean(), clustering.mean()])[:,None])
    table.index = ['\# Units', '\# Links', 'Average Degree', 'Max Degree', '\# Isolates', 'Giant Size', 'Diameter', 'Average Path Length', '\# Components', 'Clustering']

    print('\n\n\\begin{table}[ht]')
    print('\centering')
    print('\caption{Network Summary Statistics}')
    print('\\begin{threeparttable}')
    print(table.to_latex(float_format = lambda x: '%.2f' % x, header=False, escape=False))
    print('\\begin{tablenotes}[para,flushleft]')
    print('\\footnotesize Displays averages over the {} largest treated schools.'.format(len(school_IDs)))
    print('\end{tablenotes}')
    print('\end{threeparttable}')
    print('\end{table}')

def assemble_data():
    """Clean and assemble data for empirical application.

    Returns
    -------
    Y : numpy array
        n-dimensional vector of outcomes.
    D : numpy array
        n-dimensional vector of treatment indicators.
    A : NetworkX graph
        network data.
    A_norm : scipy sparse matrix
        n x n row-normalized adjacency matrix.
    pscores0 : numpy array
        n-dimensional vector of propensity scores for probability of not being treated conditional on network.
    IDs : numpy array
        Student and school identifiers for each observation.
    """
    missing_values = ['--blank--','--impossible--','--shifted--','--nom--','--void--','-55','-66','-77','-88','-95','-96','-97','-98','-99','999',' ']
    cols = ['UID','ID','SCHID','STRB','WRISTOW2','TREAT','SCHTREAT']+['ST'+str(i) for i in range(1,11)]
    data = pd.read_csv('37070-0001-Data.tsv', sep='\t', usecols=cols, na_values = missing_values)
    data.fillna(-99, inplace=True)
    data = data[data.SCHTREAT == 1] # restrict to treated schools only
    data = data[data.STRB >= 0]
    data = data[data.ID >= 0]
    data = data[data.SCHID >= 0]
    data = data[data.WRISTOW2 >= 0]
    data.at[271,'ID'] = 284 # fix typo

    data.sort_values('SCHID',inplace=True)
    data = data[(data.SCHID == 24) | (data.SCHID == 22) | (data.SCHID == 60) | (data.SCHID == 56) | (data.SCHID == 58)] # restrict sample to five largest schools

    # construct graph
    adjlist = np.hstack([data[['ID','SCHID']+['ST'+str(i) for i in range(1,11)]].values]).astype('int') 
    school_ids = np.unique(adjlist[:,1]) 
    A = nx.DiGraph()
    for i in data.UID:
        A.add_node(i)
    for sch in school_ids:
        # add edges separately for each school
        school_adjlist = adjlist[adjlist[:,1] == sch,:]
        for i in range(school_adjlist.shape[0]):
            A.add_node(sch*100000+school_adjlist[i,0]) # incorporate school ID into unit ID to get UID
            for col in range(2,12):
                if school_adjlist[i,col] in school_adjlist[:,0]:
                    A.add_edge( sch*100000+school_adjlist[i,0], sch*100000+school_adjlist[i,col] )

    Y = data.WRISTOW2.values
    D = data.TREAT.values
    D[D==0] = (-1)*np.ones(Y.size)[D==0]  # recode ineligible to -1 
    D[D==2] = np.zeros(Y.size)[D==2]   # recode control to 0

    # construct normalized adjacency matrix
    A_mat = nx.to_scipy_sparse_matrix(A, nodelist=np.squeeze(data[['UID']].values).tolist())
    out_degrees = np.squeeze(A_mat.dot(np.ones(A_mat.shape[0])[:,None]))
    r,c = A_mat.nonzero() 
    rD_sp = csr_matrix(((1.0/np.maximum(out_degrees,1))[r], (r,c)), shape=(A_mat.shape))
    A_norm = A_mat.multiply(rD_sp)

    # construct propensity score for having zero treated friends
    num_friends_blks = np.vstack([np.squeeze(np.asarray( A_mat.dot((data.STRB==k).to_numpy()[:,None]) )) for k in range(1,5)]) # each row is a vector giving the number of friends of each student assigned to student block k
    pscores0 = hypergeom(16, 8, num_friends_blks).pmf(0).prod(axis=0) # blocks of 16 students, half treated

    # relabel nodes
    IDs = data[['UID','SCHID']].values
    mapping = dict(zip(A,range(IDs.shape[0])))
    A = nx.relabel_nodes(A, mapping)

    return Y,D,A,A_norm,pscores0,IDs

