Running with 500 RFF (otherwise same as `trust_region_default_noise`)

```
--acquisition_fn_optimiser=random
--batch_size=50
--kernel_name=matern52
--num_acquisition_optimiser_start_points=5000
--num_bo_iterations=19
--num_experiments=30
--num_initial_samples=50
--num_rff_features=500
--problem=LUNAR10
--sampling_strategy=halton
--save_path=final_scbo_results/lunar_10/trust_region_500_rff/data/run_
--trust_region
```