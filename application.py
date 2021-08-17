import numpy as np, networkx as nx, math
from inference_module import *
from data_module import *

Y,D,A,A_norm,pscores0,IDs = assemble_data()
A = A.to_undirected()
bandwidths = range(4)
school_IDs = [24, 56, 22, 60, 58]

# summary statistics
network_stats(A, IDs, school_IDs)

##### Treatment effect #####

print('Treatment effect')

# units eligible for treatment
pop = (D >= 0)

# summary statistics
node_stats(Y[pop], D[pop])
treat_pop_size = Y[pop].size

# results
n = Y.size 
Z = make_Zs(Y,D,1-D,0.5*np.ones(n),0.5*np.ones(n),pop)
Z1 = make_Zs(Y,D,np.zeros(n),0.5*np.ones(n),0.5*np.ones(n),pop)
Z0 = -make_Zs(Y,np.zeros(n),1-D,0.5*np.ones(n),0.5*np.ones(n),pop)
estimate_treat = np.array([Z[pop].mean(),Z1[pop].mean(),Z0[pop].mean()])
SE_treat = np.array([network_SE(Z, A, pop, 1, True, False, b) for b in bandwidths])

##### Spillover effect #####

print('\n\nSpillover effect')

# restrict to units with a friend eligible for treatment
pop = (np.squeeze(np.asarray(A_norm.dot((D>=0)[:,None]))) > 0)
has_treated_friends = (np.squeeze(np.asarray(A_norm.dot((D==1)[:,None]))) > 0) # > 0 treated friends

# summary statistics
node_stats(Y[pop], has_treated_friends[pop])
spill_pop_size = Y[pop].size

# results
n = Y.size 
Z = make_Zs(Y,has_treated_friends,1-has_treated_friends,1-pscores0,pscores0,pop)
Z1 = make_Zs(Y,has_treated_friends,np.zeros(n),1-pscores0,pscores0,pop)
Z0 = -make_Zs(Y,np.zeros(n),1-has_treated_friends,1-pscores0,pscores0,pop)
estimate_spill = np.array([Z[pop].mean(),Z1[pop].mean(),Z0[pop].mean()])
SE_spill = np.array([network_SE(Z, A, pop, 1, True, False, b) for b in bandwidths])

# mu(0) results
SE_mu0 = np.array([network_SE(Z0, A, pop, 1, True, False, b) for b in bandwidths])

##### Print results #####

table = pd.DataFrame(np.vstack([np.hstack([estimate_treat, SE_treat]), np.hstack([estimate_spill, SE_spill])]).T)
table.index = ['$\hat\\tau(1,0)$', '$\hat\\mu(1)$', '$\hat\\mu(0)$'] + ['$b_n = ' + str(i) + '$' for i in bandwidths]
table.columns = ['Treatment', 'Spillover']
print('\n\n\\begin{table}[ht]')
print('\centering')
print('\caption{Causal Effect Estimates and Standard Errors}')
print('\\begin{threeparttable}')
print(table.to_latex(float_format = lambda x: '%.4f' % x, header=True, escape=False))
print('\\begin{tablenotes}[para,flushleft]')
print("\\footnotesize Columns display results for the treatment ($n={}$) and spillover ($n={}$) effects. Rows ``$b_n=k$'' report standard errors for the indicated values of the bandwidth.".format(treat_pop_size,spill_pop_size))
print('\end{tablenotes}')
print('\end{threeparttable}')
print('\end{table}')

print('\n\n\hat\mu(0) SEs')
print(SE_mu0)
