Ran KKT-EGO on GSBP again, but once the best acquisition function point is returned, added a check to
ensure that it has a value > 0. If it is equal to zero this implies no binding constraints were found,
so alpha should be reduced in order to find a point with binding constraints. Don't believe this will make
a difference.

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
--save_path=results/extra_kkt_results/gsbp_refined/data/run_
--nothompson_sampling
```