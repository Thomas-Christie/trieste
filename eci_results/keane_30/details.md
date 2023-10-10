Note - no `run_8_data.pkl` as this random seed caused computational issues (with a large
dataset in the final few iterations of BO). This was the only seed to fail, however, so
ended up just running an additional experiment with a different seed.

```
--batch_size=50
--kernel_name=matern52
--mc_sample_size=50
--num_acquisition_optimiser_start_points=5000
--num_bo_iterations=38
--num_experiments=30
--num_initial_samples=100
--num_rff_features=1000
--sampling_strategy=halton
--save_path=eci_results/keane_30/data/run_
--trust_region
```