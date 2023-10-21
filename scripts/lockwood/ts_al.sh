#!/bin/sh
python ts_al_experiment_runner.py --acquisition_fn_optimiser=adam \
--batch_size=1 \
--epsilon=0.01 \
--kernel_name=matern52 \
--num_acquisition_optimiser_start_points=3000 \
--num_bo_iterations=370 \
--num_experiments=30 \
--num_initial_samples=30 \
--num_rff_features=1000 \
--problem=LOCKWOOD \
--sampling_strategy=halton \
--save_path=final_ts_al_results/lockwood/trust_region/data/run_ \
--trust_region=True