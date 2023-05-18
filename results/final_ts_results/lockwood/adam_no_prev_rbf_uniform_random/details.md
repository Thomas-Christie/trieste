Ran TS-AL + Adam on Lockwood with uniform random sampling for initialisation.

``` 
Running Experiment with Flags: --acquisition_fn_optimiser=adam
--batch_size=1
--noconservative_penalty_decrease
--epsilon=0.01
--fully_consistent
--kernel_name=squared_exponential
--known_objective
--num_acquisition_optimiser_start_points=6000
--num_bo_iterations=370
--num_experiments=30
--num_initial_samples=30
--num_rff_features=1000
--problem=LOCKWOOD
--sampling_strategy=uniform_random
--save_lagrange
--save_path=results/final_ts_results/lockwood/adam_no_prev_rbf_uniform_random/data/run_
--noupdate_lagrange_via_kkt
```