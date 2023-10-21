#!/bin/sh
python ts_al_experiment_runner.py --acquisition_fn_optimiser=adam \
--batch_size=100 \
--epsilon=0.01 \
--kernel_name=matern52 \
--num_acquisition_optimiser_start_points=5000 \
--num_bo_iterations=22 \
--num_experiments=30 \
--num_initial_samples=300 \
--num_rff_features=500 \
--problem=MAZDA \
--sampling_strategy=halton \
--save_path=final_ts_al_results/mazda/data/run_ \
--trust_region=True
