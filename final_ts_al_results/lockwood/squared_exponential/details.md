Running with squared exponential and 6000 initial starting points for optimiser again.
Also 150 iterations for Adam.

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
--save_path=final_ts_al_results/lockwood/squared_exponential/data/run_
--trust_region
```