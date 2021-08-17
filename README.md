This repository contains replication files for "Causal Inference Under Approximate Neighborhood Interference". The files were coded for Python 3 and require the following dependencies: numpy, scipy, networkx, and pandas.

Contents:
* AS\_bias.py: simulate the bias of our and Aronow and Samii's variance estimators.
* DGP\_module.py: functions for simulating networks and outcome models.
* application.py: empirical application.
* data\_module.py: functions used to assemble data and compute summary statistics.
* inference\_module.py: functions for implementing our estimators and standard errors.
* monte\_carlo.py: monte carlo results. Run this once with user parameter network\_model='RGG' and once with network\_model='config'. 

To run the code, download the Paluck et al. (2016) data from [ICPSR](https://www.icpsr.umich.edu/web/civicleads/studies/37070), extract the contents, and then place the data file '37070-0001-Data.tsv' into this folder.

To run monte\_carlo.py on a cluster with a 24 hour walltime limit, you can follow these steps:
* To replicate the results, which use 10k simulations, you need to likely need to use at least 16 parallel processors.
* Each time you wish to run the file with a given choice of network\_model (see above), you will need to do two separate runs, one with num\_schools = [1,2] and one with num\_schools = [4].
* For simulations with network\_model='config' and num\_schools=[4], you will have to do even more work because of how computationally intensive this is. First, run the file with estimands\_only=True. This simulates the estimands and oracle SEs but not the rejection rates. Depending on your cluster's computational power, you may have to further split this into two runs, one with half\_sims=1 and one with half\_sims=2. Hard code the results output by the half\_sims=2 run (see line 188-9 of the file) into lines 154-5 in monte\_carlo.py. (The file already includes the hard-coded values I obtained.) Then run the simulations again but with estimands\_only=False and manual\_estimands=True to get the main results. Again, you may need to split this into two runs using half\_sims.
