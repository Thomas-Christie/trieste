Running TS-AL on Lockwood problem. Not that for the optimiser I set `num_initial_samples=6000` and
`num_optimization_runs=60` in line with the recommendations in the Trieste source code.

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
--problem=LOCKWOOD
--sampling_strategy=sobol
--save_lagrange
--save_path=results/21-04-23/lockwood_slack/data/run_
--noupdate_lagrange_via_kkt
```