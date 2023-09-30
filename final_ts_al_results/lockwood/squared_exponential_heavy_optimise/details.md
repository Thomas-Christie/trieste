IMPORTANT: switched to using 200 iterations of Adam again.

```
--acquisition_fn_optimiser=adam
--batch_size=1
--epsilon=0.01
--kernel_name=squared_exponential
--noknown_objective
--num_acquisition_optimiser_start_points=6000
--num_bo_iterations=370
--num_experiments=30
--num_initial_samples=30
--num_rff_features=1000
--problem=LOCKWOOD
--sampling_strategy=halton
--save_path=final_ts_al_results/lockwood/squared_exponential_heavy_optimise/data/run_
--trust_region
```