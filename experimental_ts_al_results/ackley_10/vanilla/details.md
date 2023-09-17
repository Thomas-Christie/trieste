Running Ackley10 without trust region. Adam was run for 200 iterations with `clipnorm=200`.

```
Running Experiment with Flags:
--acquisition_fn_optimiser=adam
--batch_size=1
--epsilon=0.01
--kernel_name=matern52
--noknown_objective
--num_acquisition_optimiser_start_points=10000
--num_bo_iterations=190
--num_experiments=30
--num_initial_samples=10
--num_rff_features=1000
--problem=ACKLEY10
--sampling_strategy=halton
--nosave_lagrange
--save_path=final_ts_al_results/ackley_10/vanilla/data/run_
--notrust_region
```