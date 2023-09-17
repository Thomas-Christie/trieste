import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gpflow.mean_functions
from absl import app, flags
import tensorflow as tf
import math
import numpy as np
import trieste
from trieste.acquisition.optimizer import (
    generate_random_search_optimizer,
    generate_sobol_random_search_optimizer,
)
from trieste.acquisition.function import (
    SCBO,
)
from trieste.acquisition.rule import (
    EfficientGlobalOptimization,
    ConstrainedTURBO,
    ConstrainedTURBOEfficientGlobalOptimization,
)
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.space import Box
from functions import constraints
from functions import objectives
from functions.lockwood.runlock.runlock import lockwood_constraint_observer
from functools import partial
import pickle


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_experiments", 30, "Number of repeats of experiment to run.")
flags.DEFINE_integer(
    "num_bo_iterations",
    190,
    "Number of iterations of Bayesian optimisation to run for.",
)
flags.DEFINE_enum(
    "problem",
    "ACKLEY10",
    ["LSQ", "GSBP", "LOCKWOOD", "ACKLEY10"],
    "Test problem to use.",
)
flags.DEFINE_integer(
    "num_rff_features",
    1000,
    "Number of Random Fourier Features to use when approximating the kernel.",
)
flags.DEFINE_integer(
    "batch_size", 1, "Number of points to sample at each iteration of BO."
)
flags.DEFINE_integer(
    "num_initial_samples",
    10,
    "Number of random samples to fit models before starting BO.",
)
flags.DEFINE_enum(
    "sampling_strategy",
    "halton",
    ["halton", "sobol", "uniform_random"],
    "Random sampling strategy for selecting " "initial points.",
)
flags.DEFINE_enum(
    "acquisition_fn_optimiser",
    "random",
    ["random", "sobol"],
    "Which optimiser to use for optimising the acquisition function.",
)
flags.DEFINE_integer(
    "num_acquisition_optimiser_start_points",
    10000,
    "Number of starting points to randomly sample from"
    "acquisition function when optimising it.",
)
flags.DEFINE_enum(
    "kernel_name",
    "matern52",
    ["matern52", "squared_exponential"],
    "Which kernel to use.",
)
flags.DEFINE_boolean(
    "trust_region", True, "Use a trust region for optimising the acquisition function."
)
flags.DEFINE_string(
    "save_path",
    "experimental_scbo_results/ackley_10/trust_region_randomcxcxzc/data/run_",
    "Prefix of path to save results to.",
)


def create_model(search_space, num_rff_features, kernel_name, data):
    gpr = build_gpr(
        data,
        search_space,
        likelihood_variance=1e-6,
        mean=gpflow.mean_functions.Zero(),
        kernel_name=kernel_name,
    )
    return GaussianProcessRegression(
        gpr, num_rff_features=num_rff_features, use_decoupled_sampler=True
    )


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main(argv):
    print(f"Running Experiment with Flags: {FLAGS.flags_into_string()}")
    for run in range(10, FLAGS.num_experiments):
        print(f"Starting Run: {run}")
        set_seed(run + 42)
        if FLAGS.problem == "LOCKWOOD":
            search_space = Box(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            )
        elif FLAGS.problem == "ACKLEY10":
            search_space = Box(
                tf.zeros(10, dtype=tf.float64), tf.ones(10, dtype=tf.float64)
            )
        else:
            search_space = Box([0.0, 0.0], [1.0, 1.0])

        if FLAGS.problem == "LSQ":
            observer = trieste.objectives.utils.mk_multi_observer(
                OBJECTIVE=objectives.linear_objective,
                INEQUALITY_CONSTRAINT_ONE=constraints.toy_constraint_one,
                INEQUALITY_CONSTRAINT_TWO=constraints.toy_constraint_two,
            )
        elif FLAGS.problem == "ACKLEY10":
            observer = trieste.objectives.utils.mk_multi_observer(
                OBJECTIVE=objectives.ackley_10,
                INEQUALITY_CONSTRAINT_ONE=constraints.ackley_10_constraint_one,
                INEQUALITY_CONSTRAINT_TWO=constraints.ackley_10_constraint_two,
            )
        elif FLAGS.problem == "LOCKWOOD":
            observer = lockwood_constraint_observer

        if FLAGS.sampling_strategy == "uniform_random":
            initial_inputs = search_space.sample(
                FLAGS.num_initial_samples, seed=run + 42
            )
        elif FLAGS.sampling_strategy == "halton":
            initial_inputs = search_space.sample_halton(
                FLAGS.num_initial_samples, seed=run + 42
            )
        elif FLAGS.sampling_strategy == "sobol":
            initial_inputs = search_space.sample_sobol(
                FLAGS.num_initial_samples, skip=42 + run * FLAGS.num_initial_samples
            )
        print(f"Initial Inputs: {initial_inputs}")
        if FLAGS.problem == "LOCKWOOD":
            initial_data = lockwood_constraint_observer(initial_inputs)
        else:
            initial_data = observer(initial_inputs)
        initial_models = trieste.utils.map_values(
            partial(
                create_model, search_space, FLAGS.num_rff_features, FLAGS.kernel_name
            ),
            initial_data,
        )

        scbo = SCBO()

        if FLAGS.acquisition_fn_optimiser == "random":
            optimizer = generate_random_search_optimizer(
                num_samples=FLAGS.num_acquisition_optimiser_start_points
            )
        elif FLAGS.acquisition_fn_optimiser == "sobol":
            optimizer = generate_sobol_random_search_optimizer(
                num_samples=FLAGS.num_acquisition_optimiser_start_points
            )

        if FLAGS.trust_region:
            dimensionality = search_space.dimension
            perturbation_prob = min(1.0, 20.0/(tf.cast(dimensionality, tf.float64)*10))
            constrained_turbo_ego = ConstrainedTURBOEfficientGlobalOptimization(
                scbo,
                optimizer=optimizer,
                batch_size=FLAGS.batch_size,
            )
            rule = ConstrainedTURBO(
                search_space=search_space,
                perturbation_prob=perturbation_prob,
                num_trust_regions=1,
                rule=constrained_turbo_ego,
                equality_constraint_tolerance=1e-3,
                L_min=tf.constant(2.0**-7, dtype=tf.float64),
                L_max=tf.constant(1.6, dtype=tf.float64),
                L_init=tf.constant(0.8, dtype=tf.float64),
                success_tolerance=max(3, math.ceil(dimensionality / 10)),
                failure_tolerance=math.ceil(dimensionality / FLAGS.batch_size),
                local_models=initial_models,
            )
            bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
            data = bo.optimize(
                FLAGS.num_bo_iterations,
                initial_data,
                initial_models,
                rule,
                fit_model=False,
                track_state=False,
            ).try_get_final_datasets()
        else:
            ego = EfficientGlobalOptimization(
                scbo,
                optimizer=optimizer,
                num_query_points=FLAGS.batch_size,
            )
            bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
            data = bo.optimize(
                FLAGS.num_bo_iterations,
                initial_data,
                initial_models,
                ego,
                track_state=False,
            ).try_get_final_datasets()
        with open(FLAGS.save_path + f"{run}_data.pkl", "wb") as fp:
            pickle.dump(data, fp)


if __name__ == "__main__":
    app.run(main)
