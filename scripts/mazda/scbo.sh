#!/bin/sh
python scbo_experiment_runner.py --acquisition_fn_optimiser=random \
--batch_size=100 \
--kernel_name=matern52 \
--num_acquisition_optimiser_start_points=5000 \
--num_bo_iterations=22 \
--num_experiments=10 \
--num_initial_samples=300 \
--num_rff_features=500 \
--problem=MAZDA \
--sampling_strategy=halton \
--save_path=final_scbo_results/mazda/data/run_ \
--trust_region=True
