Ran TS-AL on Lockwood with no L-BFGS-B, only taking the best of 6000 sobol samples to
optimise acquisition function.

``` 
Running Experiment with Flags: --batch_size=1
--noconservative_penalty_decrease
--epsilon=0.01
--fully_consistent
--known_objective
--num_bo_iterations=480
--num_experiments=30
--num_initial_samples=30
--num_rff_features=1000
--nooptimise_acquisition_fn
--problem=LOCKWOOD
--sampling_strategy=sobol
--save_lagrange
--save_path=results/25-04-23/lockwood_no_optim/data/run_
--noupdate_lagrange_via_kkt
```

