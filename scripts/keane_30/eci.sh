#!/bin/sh
python batched_eci_experiment_runner.py --batch_size=50 \
--problem=KEANE30 \
--kernel_name=matern52 \
--mc_sample_size=50 \
--num_acquisition_optimiser_start_points=5000 \
--num_bo_iterations=38 \
--num_experiments=30 \
--num_initial_samples=100 \
--num_rff_features=1000 \
--sampling_strategy=halton \
--save_path=eci_results/keane_30/data/run_ \
--trust_region=True
