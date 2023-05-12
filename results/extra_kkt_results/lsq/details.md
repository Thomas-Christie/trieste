Final KKT experiments. Now using same model as with TS-AL and only optimising best point found by optimiser.
This run uses the mean, rather than Thompson samples, as in the paper. 

``` 
Running Experiment with Flags: --alpha_lower_bound=0.01
--ei_epsilon=0.001
--epsilon=0.01
--nofeasible_region_only
--initial_alpha=0.2
--kernel_name=squared_exponential
--num_acquisition_optimiser_start_points=5000
--num_bo_iterations=40
--num_experiments=100
--num_initial_samples=5
--num_rff_features=1000
--problem=LSQ
--sampling_strategy=sobol
--save_path=results/extra_kkt_results/lsq/data/run_
--nothompson_sampling
```