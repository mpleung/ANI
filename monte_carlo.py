import numpy as np, pandas as pd, networkx as nx, multiprocessing as mp, sys, traceback
from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix, identity
from scipy.stats import norm, binom
from data_module import *
from DGP_module import *
from inference_module import *

##### User parameters #####

processes = 16                           # number of parallel processes 
network_model = 'RGG'                    # options: config, RGG

B = 10000                                # number of simulations
alpha = 0.05                             # significance level of t-test
num_schools = [1,2,4]                    # number of schools to include in sample
p = 0.5                                  # treatment probability
theta_LIM = np.array([-1,0.8,1,1])       # structural parameters for linear-in-means: intercept, 
                                         #      endogenous, exogenous, and treatment effect
theta_TSI = np.array([-1,1.5,1,1])       # structural parameters for threshold model: intercept, 
                                         #      endogenous, exogenous, and treatment effect
save_csv = True                          # save output in CSV
estimands_only = False                   # only simulate estimands
manual_estimands = False                 # use previously simulated estimands (hard-coded below)
half_sims = 0                            # 1 = only 1st half of sims, 2 = only 2nd half, any other number = run all.

print('theta: {},{}'.format(theta_LIM,theta_TSI))
exp_nbhd = network_model != 'RGG' # exponential (True) or polynomial (False) neighborhood growth rates

##### Task per node #####

def one_sim(b, deg_seq, eligibles, estimates_only, estimand_LIM, estimand_TSI, oracle_SE_LIM, oracle_SE_TSI):
    """
    Task to be parallelized: one simulation draw. Set estimates_only to True if you only want to return estimators.
    """
    n = deg_seq.size
    c = 2 if estimates_only else 1
    seed = int(n*c*(B*10) + b)
    np.random.seed(seed=seed)
    if b%100 == 0: 
        print('  b = {}'.format(b))
        sys.stdout.flush()

    # simulate data
    if network_model == 'config':
        A = nx.configuration_model(deg_seq, seed=seed)
        A = nx.Graph(A) # remove multi-edges
        A.remove_edges_from(nx.selfloop_edges(A)) # remove self-loops
        errors = np.random.normal(size=n)
    elif network_model == 'RGG':
        positions = np.random.uniform(size=(n,2))
        A = gen_RGG(positions, (deg_seq.mean()/ball_vol(2,1)/n)**(1/2))
        errors = np.random.normal(size=n) + (positions[:,0] - 0.5) 
    else:
        raise ValueError('Not a valid choice of network model.')
    A_mat = nx.to_scipy_sparse_matrix(A, nodelist=range(n), format='csc')
    deg_seq_sim = np.squeeze(A_mat.dot(np.ones(n)[:,None]))
    r,c = A_mat.nonzero() 
    rD_sp = csr_matrix(((1.0/np.maximum(deg_seq_sim,1))[r], (r,c)), shape=(A_mat.shape))
    A_norm = A_mat.multiply(rD_sp) # row-normalized adjacency matrix
    friends_eligible = np.squeeze(np.asarray(A_mat.dot(eligibles[:,None])))

    D = np.zeros(n)
    D[eligibles] = np.random.binomial(1,p,eligibles.sum()) # assign treatments to eligibles
    LIM_inv = inv( identity(n,format='csc') - theta_LIM[1]*A_norm ) # (I - beta * \tilde{A})^{-1}; used to simulate linear in means model; csc better for inverse
    Y_LIM = linear_in_means(D, A_norm, LIM_inv, errors, theta_LIM)
    Y_TSI = threshold_model(D, A_norm, errors, theta_TSI)
    friends_treated = np.squeeze(np.asarray(A_mat.dot(D[:,None]))) # num friends treated

    # estimation
    pop = (friends_eligible > 0) # indicators for inclusion in population, in this case only include units with eligible friends
    pscores0 = binom(friends_eligible,p).pmf(0)
    pscores1 = 1 - binom(friends_eligible,p).pmf(0)
    ind1 = friends_treated > 0 # exposure mapping indicators for spillover effect
    ind0 = 1 - ind1
    Zs_LIM = make_Zs(Y_LIM,ind1,ind0,pscores1,pscores0,pop)
    Zs_TSI = make_Zs(Y_TSI,ind1,ind0,pscores1,pscores0,pop)
    estimate_LIM = Zs_LIM[pop].mean()
    estimate_TSI = Zs_TSI[pop].mean()

    if estimates_only:
        return [estimate_LIM, estimate_TSI]
    else:
        # standard errors
        [SE_LIM,SE_TSI],APL,bandwidth,[PSD_failure_LIM,PSD_failure_TSI] \
                = network_SE([Zs_LIM,Zs_TSI], A, pop, 1, exp_nbhd, True) # network-robust SE
        naive_SE_LIM = Zs_LIM[pop].std() / math.sqrt(pop.sum()) # iid SE
        naive_SE_TSI = Zs_TSI[pop].std() / math.sqrt(pop.sum())

        # t-test
        numerator_LIM = np.abs(estimate_LIM - estimand_LIM)
        numerator_TSI = np.abs(estimate_TSI - estimand_TSI)
        ttest_LIM = numerator_LIM / SE_LIM > norm.ppf(1-alpha/2)
        ttest_TSI = numerator_TSI / SE_TSI > norm.ppf(1-alpha/2)
        naive_ttest_LIM = numerator_LIM / naive_SE_LIM > norm.ppf(1-alpha/2)
        naive_ttest_TSI = numerator_TSI / naive_SE_TSI > norm.ppf(1-alpha/2)
        oracle_ttest_LIM = numerator_LIM / oracle_SE_LIM > norm.ppf(1-alpha/2)
        oracle_ttest_TSI = numerator_TSI / oracle_SE_TSI > norm.ppf(1-alpha/2)

        return [estimate_LIM, estimate_TSI, ttest_LIM, ttest_TSI, oracle_ttest_LIM, oracle_ttest_TSI, naive_ttest_LIM, naive_ttest_TSI, SE_LIM, SE_TSI, oracle_SE_LIM, oracle_SE_TSI, naive_SE_LIM, naive_SE_TSI, ind1[pop].sum(), ind0[pop].sum(), APL, bandwidth, PSD_failure_LIM, PSD_failure_TSI]

##### Containers #####

estimates_LIM = np.zeros(len(num_schools))            # treatment effect estimates for linear-in-means model
estimates_TSI = np.zeros(len(num_schools))            # treatment effect estimates for threshold model
ttests_LIM = np.zeros(len(num_schools))               # t-test for linear-in-means model using our standard errors
ttests_TSI = np.zeros(len(num_schools))               # t-test for threshold model using our standard errors
oracle_ttests_LIM = np.zeros(len(num_schools))        # t-test using true standard errors
oracle_ttests_TSI = np.zeros(len(num_schools))
naive_ttests_LIM = np.zeros(len(num_schools))         # t-test using iid standard errors
naive_ttests_TSI = np.zeros(len(num_schools))
SEs_LIM = np.zeros(len(num_schools))                  # our standard errors 
SEs_TSI = np.zeros(len(num_schools))
oracle_SEs_LIM = np.zeros(len(num_schools))
oracle_SEs_TSI = np.zeros(len(num_schools))
naive_SEs_LIM = np.zeros(len(num_schools))
naive_SEs_TSI = np.zeros(len(num_schools))
eff_SS1 = np.zeros(len(num_schools))                  # number of units assigned to first exposure mapping
eff_SS0 = np.zeros(len(num_schools))
APLs = np.zeros(len(num_schools))                     # average path length
bandwidths = np.zeros(len(num_schools))               # bandwidth
Ns = np.zeros(len(num_schools)).astype('int')         # population sizes
PSD_failures_LIM = np.zeros(len(num_schools))
PSD_failures_TSI = np.zeros(len(num_schools))

##### Main #####

# assemble network data
_,D,A,_,_,IDs = assemble_data()
deg_seq = np.array([i[1] for i in A.out_degree])
A = A.to_undirected()
eligibles = (D >= 0)

for i,ns in enumerate(num_schools):
    # select schools
    if ns == 1:
        students = (IDs[:,1] == 24)
    elif ns == 2:
        students = (IDs[:,1] == 24) + (IDs[:,1] == 22)
    else:
        students = (IDs[:,1] == 24) + (IDs[:,1] == 22) + (IDs[:,1] == 60) + (IDs[:,1] == 56)
    print('n = {}'.format(students.sum()))
    Ns[i] = students.sum()

    if deg_seq[students].sum() % 2 != 0: 
        deg_seq_pop = deg_seq[students].copy()
        deg_seq_pop[0] += 1 # need even total degree for configuration model
    else:
        deg_seq_pop = deg_seq[students]

    if manual_estimands:
        # HARD CODE simulated estimands and oracle SEs
        if ns == 4 and network_model == 'config': 
            estimands = np.array([0.3059139,0.0805116]) 
            oracle_SEs = np.array([0.69383978,0.06043479])
        else:
            estimands = np.array([0,0])
            oracle_SEs = np.array([1,1])
    else:
        # simulate estimands and oracle standard errors

        def one_sim_wrapper(b):
            try:
                return one_sim(b, deg_seq_pop, eligibles[students], True, 0, 0, 0, 0)
            except:
                print('%s: %s' % (b, traceback.format_exc()))
                sys.stdout.flush()
        
        sims_range = range(B,2*B)
        if half_sims == 1: 
            sims_range = range(B,B+int(B/2))
        elif half_sims == 2: 
            sims_range = range(B+int(B/2),2*B)
        pool = mp.Pool(processes=processes, maxtasksperchild=1)
        parallel_output = pool.imap(one_sim_wrapper, sims_range, chunksize=25) 
        pool.close()
        pool.join()
        results = np.array([r for r in parallel_output])

        if half_sims in [1,2]:
            gd = '_1' if half_sims==1 else '_2'
            table = pd.DataFrame(results)
            table.to_csv('half_sims_oracle_' + str(ns) + gd + '.csv', float_format='%.10f', index=False, header=False)

        estimands = results.mean(axis=0)
        oracle_SEs = results.std(axis=0)

    print('Estimands: {}'.format(estimands)) # use these to HARD CODE estimands above
    print('Oracle SEs: {}'.format(oracle_SEs))
    sys.stdout.flush()

    if estimands_only:
        results = np.zeros(26)
    else:
        # simulate main results

        def one_sim_wrapper(b):
            try:
                return one_sim(b, deg_seq_pop, eligibles[students], False, estimands[0], estimands[1], oracle_SEs[0], oracle_SEs[1])
            except:
                print('%s: %s' % (b, traceback.format_exc()))
                sys.stdout.flush()
        
        sims_range = range(B)
        if half_sims == 1: 
            sims_range = range(int(B/2))
        elif half_sims == 2: 
            sims_range = range(int(B/2),B)
        pool = mp.Pool(processes=processes, maxtasksperchild=1)
        parallel_output = pool.imap(one_sim_wrapper, sims_range, chunksize=25)
        pool.close()
        pool.join()
        results = np.array([r for r in parallel_output])

        if half_sims in [1,2]:
            gd = '_1' if half_sims==1 else '_2'
            table = pd.DataFrame(results)
            table.to_csv('half_sims_main_' + str(ns) + gd + '.csv', float_format='%.10f', index=False, header=False)

        results = results.mean(axis=0)

        if half_sims == 2:
            results1 = pd.read_csv('half_sims_main_4_1.csv', header=None)
            results2 = pd.read_csv('half_sims_main_4_2.csv', header=None)
            results = np.vstack([results1.values, results2.values]).mean(axis=0)

    # store results
    estimates_LIM[i] = results[0]
    estimates_TSI[i] = results[1]
    ttests_LIM[i] = results[2]
    ttests_TSI[i] = results[3]
    oracle_ttests_LIM[i] = results[4]
    oracle_ttests_TSI[i] = results[5]
    naive_ttests_LIM[i] = results[6]
    naive_ttests_TSI[i] = results[7]
    SEs_LIM[i] = results[8]
    SEs_TSI[i] = results[9]
    oracle_SEs_LIM[i] = results[10]
    oracle_SEs_TSI[i] = results[11]
    naive_SEs_LIM[i] = results[12]
    naive_SEs_TSI[i] = results[13]
    eff_SS1[i] = results[14]
    eff_SS0[i] = results[15]
    APLs[i] = results[16]
    bandwidths[i] = results[17]
    PSD_failures_LIM[i] = results[18]*B
    PSD_failures_TSI[i] = results[19]*B

##### Output #####

print('\nLinear-in-Means')
print('Failures: {}'.format(PSD_failures_LIM))
table = pd.DataFrame(np.vstack([Ns, eff_SS1, eff_SS0, estimates_LIM, ttests_LIM, oracle_ttests_LIM, naive_ttests_LIM, SEs_LIM, oracle_SEs_LIM, naive_SEs_LIM, bandwidths, APLs]).T)
table.index = num_schools
table.columns = ['$n$', "$\hat{n}(t)$", "$\hat{n}(t')$", 'Estimate', 'Rej', 'Oracle Rej', 'Naive Rej', 'SE', 'Oracle SE', 'Naive SE', '$b_n$', 'APL']
print(table.to_latex(float_format = lambda x: '%.4f' % x, header=True, escape=False))

print('Threshold Model')
print('Failures: {}'.format(PSD_failures_TSI))
table = pd.DataFrame(np.vstack([Ns, eff_SS1, eff_SS0, estimates_TSI, ttests_TSI, oracle_ttests_TSI, naive_ttests_TSI, SEs_TSI, oracle_SEs_TSI, naive_SEs_TSI, bandwidths, APLs]).T)
table.index = num_schools
table.columns = ['$n$', "$\hat{n}(t)$", "$\hat{n}(t')$", 'Estimate', 'Rej', 'Oracle Rej', 'Naive Rej', 'SE', 'Oracle SE', 'Naive SE', '$b_n$', 'APL']
print(table.to_latex(float_format = lambda x: '%.4f' % x, header=True, escape=False))

if save_csv:
    table = pd.DataFrame(np.vstack([ np.vstack([Ns, eff_SS1, eff_SS0, estimates_LIM, ttests_LIM, oracle_ttests_LIM, naive_ttests_LIM, SEs_LIM, oracle_SEs_LIM, naive_SEs_LIM, bandwidths, APLs]).T, np.vstack([Ns, eff_SS1, eff_SS0, estimates_TSI, ttests_TSI, oracle_ttests_TSI, naive_ttests_TSI, SEs_TSI, oracle_SEs_TSI, naive_SEs_TSI, bandwidths, APLs]).T]))
    table.index = num_schools + num_schools
    table.columns = ['$n$', "$\hat{n}(t)$", "$\hat{n}(t')$", 'Estimate', 'Rej', 'Oracle Rej', 'Naive Rej', 'SE', 'Oracle SE', 'Naive SE', '$b_n$', 'APL']
    table.to_csv('results_table_monte_carlo_' + network_model + '.csv', float_format='%.6f')

