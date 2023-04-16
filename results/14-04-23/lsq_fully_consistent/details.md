Updated implementation of TS-AL so that it is consistent with the *appendix* of the
original paper (which differs slightly from the main body). Also used Sobol indices for
sampling initial values.

``` 
Running Experiment with Flags: --batch_size=1
--noconservative_penalty_decrease
--epsilon=0.01
--fully_consistent
--num_bo_iterations=45
--num_experiments=100
--num_initial_samples=5
--num_rff_features=1000
--problem=LSQ
--sampling_strategy=sobol
--save_lagrange
--save_path=results/14-04-23/lsq_fully_consistent/data/run_
--noupdate_lagrange_via_kkt
--nohelp

```