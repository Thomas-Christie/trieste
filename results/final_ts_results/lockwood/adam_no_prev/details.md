Ran TS-AL with Adam optimiser on Lockwood problem, not including previous point in random samples for
deciding starting point for Adam optimiser.

``` 
Running Experiment with Flags: --acquisition_fn_optimiser=adam
--batch_size=1
--noconservative_penalty_decrease
--epsilon=0.01
--fully_consistent
--known_objective
--num_acquisition_optimiser_start_points=6000
--num_bo_iterations=370
--num_experiments=20
--num_initial_samples=30
--num_rff_features=1000
--problem=LOCKWOOD
--sampling_strategy=sobol
--save_lagrange
--save_path=results/final_ts_results/lockwood/adam_no_prev/data/run_
--noupdate_lagrange_via_kkt
```