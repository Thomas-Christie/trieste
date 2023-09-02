import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gpflow.mean_functions
from absl import app, flags
import tensorflow as tf
import numpy as np
import trieste
from trieste.acquisition.optimizer import (
    generate_continuous_optimizer,
    generate_adam_optimizer,
    generate_random_search_optimizer,
)
from trieste.acquisition.function.thompson_sampling_augmented_lagrangian import (
    ThompsonSamplingAugmentedLagrangian,
)
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.models.gpflow import build_gpr, GaussianProcessRegression

from trieste.space import Box
from functions import constraints
from functions import objectives
from functions.lockwood.runlock.runlock import lockwood_constraint_observer
from functools import partial
import pickle

OBJECTIVE = "OBJECTIVE"
INEQUALITY_CONSTRAINT_ONE = "INEQUALITY_CONSTRAINT_ONE"
INEQUALITY_CONSTRAINT_TWO = "INEQUALITY_CONSTRAINT_TWO"
EQUALITY_CONSTRAINT_ONE = "EQUALITY_CONSTRAINT_ONE"
EQUALITY_CONSTRAINT_TWO = "EQUALITY_CONSTRAINT_TWO"

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_experiments", 30, "Number of repeats of experiment to run.")
flags.DEFINE_integer(
    "num_bo_iterations",
    190,
    "Number of iterations of Bayesian optimisation to run for.",
)
flags.DEFINE_float(
    "epsilon",
    0.01,
    "Bound within which equality constraints are considered to be satisfied.",
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
    "sobol",
    ["sobol", "uniform_random"],
    "Random sampling strategy for selecting " "initial points.",
)
flags.DEFINE_enum(
    "acquisition_fn_optimiser",
    "l-bfgs-b",
    ["random", "sobol", "l-bfgs-b", "adam"],
    "Which optimiser to use for optimising the acquisition function.",
)
flags.DEFINE_integer(
    "num_acquisition_optimiser_start_points",
    5000,
    "Number of starting points to randomly sample from"
    "acquisition function when optimising it.",
)
flags.DEFINE_boolean(
    "known_objective",
    False,
    "Whether to use a known objective function or model it with a surrogate.",
)
flags.DEFINE_enum(
    "kernel_name",
    "matern52",
    ["matern52", "squared_exponential"],
    "Which kernel to use.",
)
flags.DEFINE_boolean(
    "save_lagrange", False, "Save intermediate values of Lagrange multipliers."
)
flags.DEFINE_string(
    "save_path",
    "results/final_ts_results/lockwood/dsdano_opt_rbf_uniform_random/data/run_",
    "Prefix of path to save results to.",
)


def create_model(search_space, num_rff_features, kernel_name, data):
    gpr = build_gpr(
        data, search_space, likelihood_variance=1e-6, kernel_name=kernel_name
    )
    return GaussianProcessRegression(
        gpr, num_rff_features=num_rff_features, use_decoupled_sampler=True
    )


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main(argv):
    print(f"Running Experiment with Flags: {FLAGS.flags_into_string()}")
    for run in range(FLAGS.num_experiments):
        print(f"Starting Run: {run}")
        set_seed(run + 42)
        if FLAGS.problem == "LOCKWOOD":
            search_space = Box(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
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
        elif FLAGS.problem == "GSBP":
            observer = trieste.objectives.utils.mk_multi_observer(
                OBJECTIVE=objectives.goldstein_price,
                INEQUALITY_CONSTRAINT_ONE=constraints.toy_constraint_one,
                EQUALITY_CONSTRAINT_ONE=constraints.centered_branin,
                EQUALITY_CONSTRAINT_TWO=constraints.parr_constraint,
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

        if FLAGS.known_objective:
            if FLAGS.problem == "LOCKWOOD":
                known_objective = objectives.lockwood_objective_trajectory
        else:
            known_objective = None

        # Initialise inequality constraint Lagrange multipliers
        if (
            FLAGS.problem == "LSQ"
            or FLAGS.problem == "ACKLEY10"
            or FLAGS.problem == "LOCKWOOD"
        ):
            inequality_lambda = {
                INEQUALITY_CONSTRAINT_ONE: tf.zeros(1, dtype=tf.float64),
                INEQUALITY_CONSTRAINT_TWO: tf.zeros(1, dtype=tf.float64),
            }
        elif FLAGS.problem == "GSBP":
            inequality_lambda = {
                INEQUALITY_CONSTRAINT_ONE: tf.zeros(1, dtype=tf.float64)
            }

        # Initialise equality constraint Lagrange multipliers
        if FLAGS.problem == "GSBP":
            equality_lambda = {
                EQUALITY_CONSTRAINT_ONE: tf.zeros(1, dtype=tf.float64),
                EQUALITY_CONSTRAINT_TWO: tf.zeros(1, dtype=tf.float64),
            }

        lambda_save_path = None
        if FLAGS.save_path is not None:
            lambda_save_path = FLAGS.save_path + f"{run}"

        if (
            FLAGS.problem == "LSQ"
            or FLAGS.problem == "ACKLEY10"
            or FLAGS.problem == "LOCKWOOD"
        ):
            augmented_lagrangian = ThompsonSamplingAugmentedLagrangian(
                objective_tag=OBJECTIVE,
                known_objective=known_objective,
                inequality_constraint_prefix="INEQUALITY",
                equality_constraint_prefix=None,
                inequality_lambda=inequality_lambda,
                equality_lambda=None,
                batch_size=FLAGS.batch_size,
                penalty=None,
                epsilon=FLAGS.epsilon,
                search_space=search_space,
                plot=False,
                save_lambda=FLAGS.save_lagrange,
                save_path=lambda_save_path,
                num_bo_iters=FLAGS.num_bo_iterations,
            )
        elif FLAGS.problem == "GSBP":
            augmented_lagrangian = ThompsonSamplingAugmentedLagrangian(
                objective_tag=OBJECTIVE,
                known_objective=known_objective,
                inequality_constraint_prefix="INEQUALITY",
                equality_constraint_prefix="EQUALITY",
                inequality_lambda=inequality_lambda,
                equality_lambda=equality_lambda,
                batch_size=FLAGS.batch_size,
                penalty=None,
                epsilon=FLAGS.epsilon,
                search_space=search_space,
                plot=False,
                save_lambda=FLAGS.save_lagrange,
                save_path=lambda_save_path,
                num_bo_iters=FLAGS.num_bo_iterations,
            )

        if FLAGS.acquisition_fn_optimiser == "l-bfgs-b":
            optimizer = generate_continuous_optimizer(
                num_initial_samples=FLAGS.num_acquisition_optimiser_start_points,
                num_optimization_runs=1,
                optimizer_args={"options": {"maxls": 100}},
            )
        elif FLAGS.acquisition_fn_optimiser == "random":
            optimizer = generate_random_search_optimizer(
                num_samples=FLAGS.num_acquisition_optimiser_start_points
            )
        elif FLAGS.acquisition_fn_optimiser == "adam":
            optimizer = generate_adam_optimizer(
                num_initial_samples=FLAGS.num_acquisition_optimiser_start_points
            )

        rule = EfficientGlobalOptimization(
            augmented_lagrangian, optimizer=optimizer, num_query_points=FLAGS.batch_size
        )
        bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
        data = bo.optimize(
            FLAGS.num_bo_iterations,
            initial_data,
            initial_models,
            rule,
            track_state=False,
        ).try_get_final_datasets()
        with open(FLAGS.save_path + f"{run}_data.pkl", "wb") as fp:
            pickle.dump(data, fp)


if __name__ == "__main__":
    app.run(main)
