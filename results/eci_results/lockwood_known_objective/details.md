Running ECI on Lockwood, treating the objective as known by modelling it with a GP with a linear
mean and kernel with extremely low variance.

NOTE - ALso modified `bayesian_optimizer.py` to prevent it from optimising the parameters of the 
objective function surrogate model.

``` 
Running Experiment with Flags: --kernel_name=squared_exponential
--num_acquisition_optimiser_start_points=6000
--num_bo_iterations=370
--num_experiments=30
--num_initial_samples=30
--num_rff_features=1000
--sampling_strategy=uniform_random
--save_path=results/eci_results/lockwood_known_objective/data/run_
```