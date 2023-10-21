#!/bin/sh
python eci_experiment_runner.py --kernel_name=matern52 \
--num_acquisition_optimiser_start_points=5000 \
--num_bo_iterations=190 \
--num_experiments=30 \
--num_initial_samples=10 \
--num_rff_features=1000 \
--problem=ACKLEY10 \
--sampling_strategy=halton \
--save_path=eci_results/ackley_10/data/run_ \
--trust_region=True
