Final Adam on LSQ problem. Now using zero mean function.

``` 
Running Experiment with Flags: --acquisition_fn_optimiser=adam
--batch_size=1
--noconservative_penalty_decrease
--epsilon=0.01
--fully_consistent
--noknown_objective
--num_acquisition_optimiser_start_points=5000
--num_bo_iterations=40
--num_experiments=100
--num_initial_samples=5
--num_rff_features=1000
--problem=LSQ
--sampling_strategy=sobol
--save_lagrange
--save_path=results/final_ts_results/lsq/adam/data/run_
--noupdate_lagrange_via_kkt
```