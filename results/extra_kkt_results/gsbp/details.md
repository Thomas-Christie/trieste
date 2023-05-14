Ran original KKT-EGO on GSBP (i.e. no Thompson sampling).

``` 
Running Experiment with Flags: --alpha_lower_bound=0.01
--ei_epsilon=0.001
--epsilon=0.01
--nofeasible_region_only
--initial_alpha=0.2
--kernel_name=squared_exponential
--num_acquisition_optimiser_start_points=5000
--num_bo_iterations=140
--num_experiments=20
--num_initial_samples=10
--num_rff_features=1000
--problem=GSBP
--sampling_strategy=sobol
--save_path=results/extra_kkt_results/gsbp/data/run_
--nothompson_sampling
```