Ran Vanilla SCBO with plain random sampling rather than Sobol sampling to check for any
differences in performance. 

```
--acquisition_fn_optimiser=random
--batch_size=1
--kernel_name=matern52
--num_acquisition_optimiser_start_points=10000
--num_bo_iterations=190
--num_experiments=30
--num_initial_samples=10
--num_rff_features=1000
--problem=ACKLEY10
--sampling_strategy=halton
--save_path=final_scbo_results/ackley_10/vanilla_random/data/run_
--notrust_region
```