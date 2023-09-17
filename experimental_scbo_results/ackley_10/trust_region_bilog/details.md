Trying out SCBO with bilog transformation applied to constraints. Note that dividing
constants were removed from the constraint functions.

```
--acquisition_fn_optimiser=sobol
--batch_size=1
--kernel_name=matern52
--num_acquisition_optimiser_start_points=10000
--num_bo_iterations=190
--num_experiments=30
--num_initial_samples=10
--num_rff_features=1000
--problem=ACKLEY10
--sampling_strategy=halton
--save_path=experimental_scbo_results/ackley_10/trust_region_bilog/data/run_
--trust_region
```