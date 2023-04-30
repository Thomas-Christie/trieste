Final Adam on GSBP problem with aggressive penalty reduction. Now using zero mean function.

``` 
Running Experiment with Flags: --acquisition_fn_optimiser=adam
--batch_size=1
--noconservative_penalty_decrease
--epsilon=0.01
--fully_consistent
--noknown_objective
--num_acquisition_optimiser_start_points=5000
--num_bo_iterations=140
--num_experiments=50
--num_initial_samples=10
--num_rff_features=1000
--problem=GSBP
--sampling_strategy=sobol
--save_lagrange
--save_path=results/final_ts_results/gsbp/adam_aggressive/data/run_
--noupdate_lagrange_via_kkt
```