#!/bin/sh
python eci_experiment_runner.py --kernel_name=matern52 \
--num_acquisition_optimiser_start_points=3000 \
--num_bo_iterations=370 \
--num_experiments=30 \
--num_initial_samples=30 \
--num_rff_features=1000 \
--problem=LOCKWOOD \
--sampling_strategy=halton \
--save_path=eci_results/lockwood/data/run_ \
--trust_region=True
