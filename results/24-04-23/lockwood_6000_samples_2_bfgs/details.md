Ran again with `num_initial_samples=6000` and `num_optimisation_runs=2`.
On run 6 the initial penalty is `1.432118060810256e-16` which made it very slow.
Presumably this is due to massively high gradients.

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
--save_path=results/24-04-23/lockwood_6000_samples_2_bfgs/data/run_
--noupdate_lagrange_via_kkt
```