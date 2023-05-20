import gpflow.mean_functions
from absl import app, flags
import tensorflow as tf
import numpy as np
import trieste
from trieste.acquisition.optimizer import generate_al_continuous_optimizer, generate_sobol_random_search_optimizer, \
    generate_al_adam_optimizer, generate_random_search_optimizer
from trieste.acquisition.function.new_constrained_thompson_sampling import BatchThompsonSamplingAugmentedLagrangian, FullyConsistentBatchThompsonSamplingAugmentedLagrangian
from trieste.acquisition.rule import ALEfficientGlobalOptimization
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

flags.DEFINE_integer('num_experiments', 30, 'Number of repeats of experiment to run.')
flags.DEFINE_integer('num_bo_iterations', 370, 'Number of iterations of Bayesian optimisation to run for.')
flags.DEFINE_float('epsilon', 0.01, 'Bound within which equality constraints are considered to be satisfied.')
flags.DEFINE_enum('problem', 'LOCKWOOD', ['LSQ', 'GSBP', 'LOCKWOOD'], 'Test problem to use.')
flags.DEFINE_integer('num_rff_features', 1000, 'Number of Random Fourier Features to use when approximating the kernel.')
flags.DEFINE_integer('batch_size', 1, 'Number of points to sample at each iteration of BO.')
flags.DEFINE_integer('num_initial_samples', 30, 'Number of random samples to fit models before starting BO.')
flags.DEFINE_boolean('update_lagrange_via_kkt', False, 'Whether to update Lagrange multipliers using a gradient-based'
                                                       'approach based on KKT conditions.')
flags.DEFINE_boolean('conservative_penalty_decrease', False, 'Whether to reduce the penalty parameter more '
                                                             'conservatively if no valid solutions have been found '
                                                             'yet.')
flags.DEFINE_boolean('fully_consistent', True, 'Whether to update Lagrange multipliers and penalty parameter in a '
                                               'manner which is fully consistent with the appendix of the original '
                                               'paper (which differs from the body of the original paper).')
flags.DEFINE_enum('sampling_strategy', 'uniform_random', ['sobol', 'uniform_random'], 'Random sampling strategy for selecting '
                                                                             'initial points.')
flags.DEFINE_enum('acquisition_fn_optimiser', 'random', ['random', 'sobol', 'l-bfgs-b', 'adam'],
                  'Which optimiser to use for optimising the acquisition function.')
flags.DEFINE_integer('num_acquisition_optimiser_start_points', 6000, 'Number of starting points to randomly sample from'
                                                                     'acquisition function when optimising it.')
flags.DEFINE_boolean('known_objective', True, 'Whether to use a known objective function or model it with a surrogate.')
flags.DEFINE_enum('kernel_name', 'squared_exponential', ['matern52', 'squared_exponential'], 'Which kernel to use.')
flags.DEFINE_boolean('save_lagrange', True, 'Save intermediate values of Lagrange multipliers.')
flags.DEFINE_string('save_path', 'results/final_ts_results/lockwood/no_opt_rbf_uniform_random/data/run_',
                    'Prefix of path to save results to.')


def create_model(search_space, num_rff_features, kernel_name, data):
    gpr = build_gpr(data, search_space, likelihood_variance=1e-6, mean=gpflow.mean_functions.Zero(), kernel_name=kernel_name)
    return GaussianProcessRegression(gpr, num_rff_features=num_rff_features, use_decoupled_sampler=True)


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main(argv):
    print(f"Running Experiment with Flags: {FLAGS.flags_into_string()}")
    for run in range(FLAGS.num_experiments):
        print(f"Starting Run: {run}")
        set_seed(run + 42)
        if FLAGS.problem == "LOCKWOOD":
            search_space = Box([0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        else:
            search_space = Box([0.0, 0.0], [1.0, 1.0])

        if FLAGS.problem == "LSQ":
            observer = trieste.objectives.utils.mk_multi_observer(
                OBJECTIVE=objectives.linear_objective,
                INEQUALITY_CONSTRAINT_ONE=constraints.toy_constraint_one,
                INEQUALITY_CONSTRAINT_TWO=constraints.toy_constraint_two)
        elif FLAGS.problem == "GSBP":
            observer = trieste.objectives.utils.mk_multi_observer(
                OBJECTIVE=objectives.goldstein_price,
                INEQUALITY_CONSTRAINT_ONE=constraints.toy_constraint_one,
                EQUALITY_CONSTRAINT_ONE=constraints.centered_branin,
                EQUALITY_CONSTRAINT_TWO=constraints.parr_constraint)
        elif FLAGS.problem == "LOCKWOOD":
            observer = lockwood_constraint_observer

        if FLAGS.sampling_strategy == 'uniform_random':
            initial_inputs = search_space.sample(FLAGS.num_initial_samples, seed=run+42)
        elif FLAGS.sampling_strategy == 'sobol':
            initial_inputs = search_space.sample_sobol(FLAGS.num_initial_samples, skip=42+run*FLAGS.num_initial_samples)
        print(f"Initial Inputs: {initial_inputs}")
        if FLAGS.problem == "LOCKWOOD":
            initial_data = lockwood_constraint_observer(initial_inputs)
        else:
            initial_data = observer(initial_inputs)
        initial_models = trieste.utils.map_values(partial(create_model, search_space, FLAGS.num_rff_features, FLAGS.kernel_name),
                                                  initial_data)

        if FLAGS.known_objective:
            if FLAGS.problem == "LOCKWOOD":
                known_objective = objectives.lockwood_objective_trajectory
        else:
            known_objective = None

        # Initialise inequality constraint Lagrange multipliers
        if FLAGS.problem == "LSQ":
            if FLAGS.fully_consistent:
                inequality_lambda = {INEQUALITY_CONSTRAINT_ONE: tf.zeros(1, dtype=tf.float64),
                                     INEQUALITY_CONSTRAINT_TWO: tf.zeros(1, dtype=tf.float64)}
            else:
                inequality_lambda = {INEQUALITY_CONSTRAINT_ONE: tf.zeros((1, FLAGS.batch_size, 1), dtype=tf.float64),
                                     INEQUALITY_CONSTRAINT_TWO: tf.zeros((1, FLAGS.batch_size, 1), dtype=tf.float64)}
        elif FLAGS.problem == "GSBP":
            if FLAGS.fully_consistent:
                inequality_lambda = {INEQUALITY_CONSTRAINT_ONE: tf.zeros(1, dtype=tf.float64)}
            else:
                inequality_lambda = {INEQUALITY_CONSTRAINT_ONE: tf.zeros((1, FLAGS.batch_size, 1), dtype=tf.float64)}
        elif FLAGS.problem == "LOCKWOOD":
            if FLAGS.fully_consistent:
                inequality_lambda = {INEQUALITY_CONSTRAINT_ONE: tf.zeros(1, dtype=tf.float64),
                                     INEQUALITY_CONSTRAINT_TWO: tf.zeros(1, dtype=tf.float64)}
            else:
                inequality_lambda = {INEQUALITY_CONSTRAINT_ONE: tf.zeros((1, FLAGS.batch_size, 1), dtype=tf.float64),
                                     INEQUALITY_CONSTRAINT_TWO: tf.zeros((1, FLAGS.batch_size, 1), dtype=tf.float64)}

        # Initialise equality constraint Lagrange multipliers
        if FLAGS.problem == "GSBP":
            if FLAGS.fully_consistent:
                equality_lambda = {EQUALITY_CONSTRAINT_ONE: tf.zeros(1, dtype=tf.float64),
                                   EQUALITY_CONSTRAINT_TWO: tf.zeros(1, dtype=tf.float64)}
            else:
                equality_lambda = {EQUALITY_CONSTRAINT_ONE: tf.zeros((1, FLAGS.batch_size, 1), dtype=tf.float64),
                                   EQUALITY_CONSTRAINT_TWO: tf.zeros((1, FLAGS.batch_size, 1), dtype=tf.float64)}

        lambda_save_path = None
        if FLAGS.save_path is not None:
            lambda_save_path = FLAGS.save_path + f"{run}"

        if FLAGS.problem == "LSQ" or FLAGS.problem == "LOCKWOOD":
            if FLAGS.fully_consistent:
                augmented_lagrangian = FullyConsistentBatchThompsonSamplingAugmentedLagrangian(objective_tag=OBJECTIVE,
                                                                                               known_objective=known_objective,
                                                                                               inequality_constraint_prefix="INEQUALITY",
                                                                                               equality_constraint_prefix=None,
                                                                                               inequality_lambda=inequality_lambda,
                                                                                               equality_lambda=None,
                                                                                               batch_size=FLAGS.batch_size, penalty=None,
                                                                                               conservative_penalty_decrease=FLAGS.conservative_penalty_decrease,
                                                                                               epsilon=FLAGS.epsilon,
                                                                                               update_lagrange_via_kkt=FLAGS.update_lagrange_via_kkt,
                                                                                               search_space=search_space, plot=False,
                                                                                               save_lambda=FLAGS.save_lagrange,
                                                                                               save_path=lambda_save_path,
                                                                                               num_bo_iters=FLAGS.num_bo_iterations)
            else:
                augmented_lagrangian = BatchThompsonSamplingAugmentedLagrangian(objective_tag=OBJECTIVE,
                                                                                inequality_constraint_prefix="INEQUALITY",
                                                                                equality_constraint_prefix=None,
                                                                                inequality_lambda=inequality_lambda,
                                                                                equality_lambda=None,
                                                                                batch_size=FLAGS.batch_size, penalty=None,
                                                                                conservative_penalty_decrease=FLAGS.conservative_penalty_decrease,
                                                                                epsilon=FLAGS.epsilon,
                                                                                update_lagrange_via_kkt=FLAGS.update_lagrange_via_kkt,
                                                                                search_space=search_space, plot=False,
                                                                                save_lambda=FLAGS.save_lagrange,
                                                                                save_path=lambda_save_path,
                                                                                num_bo_iters=FLAGS.num_bo_iterations)
        elif FLAGS.problem == "GSBP":
            if FLAGS.fully_consistent:
                augmented_lagrangian = FullyConsistentBatchThompsonSamplingAugmentedLagrangian(objective_tag=OBJECTIVE,
                                                                                               known_objective=known_objective,
                                                                                               inequality_constraint_prefix="INEQUALITY",
                                                                                               equality_constraint_prefix="EQUALITY",
                                                                                               inequality_lambda=inequality_lambda,
                                                                                               equality_lambda=equality_lambda,
                                                                                               batch_size=FLAGS.batch_size,
                                                                                               penalty=None,
                                                                                               conservative_penalty_decrease=FLAGS.conservative_penalty_decrease,
                                                                                               epsilon=FLAGS.epsilon,
                                                                                               update_lagrange_via_kkt=FLAGS.update_lagrange_via_kkt,
                                                                                               search_space=search_space, plot=False,
                                                                                               save_lambda=FLAGS.save_lagrange,
                                                                                               save_path=lambda_save_path,
                                                                                               num_bo_iters=FLAGS.num_bo_iterations)
            else:
                augmented_lagrangian = BatchThompsonSamplingAugmentedLagrangian(objective_tag=OBJECTIVE,
                                                                                inequality_constraint_prefix="INEQUALITY",
                                                                                equality_constraint_prefix="EQUALITY",
                                                                                inequality_lambda=inequality_lambda,
                                                                                equality_lambda=equality_lambda,
                                                                                batch_size=FLAGS.batch_size, penalty=None,
                                                                                conservative_penalty_decrease=FLAGS.conservative_penalty_decrease,
                                                                                epsilon=FLAGS.epsilon,
                                                                                update_lagrange_via_kkt=FLAGS.update_lagrange_via_kkt,
                                                                                search_space=search_space, plot=False,
                                                                                save_lambda=FLAGS.save_lagrange,
                                                                                save_path=lambda_save_path,
                                                                                num_bo_iters=FLAGS.num_bo_iterations)

        if FLAGS.acquisition_fn_optimiser == 'l-bfgs-b':
            optimizer = generate_al_continuous_optimizer(num_initial_samples=FLAGS.num_acquisition_optimiser_start_points,
                                                         num_optimization_runs=2)
        elif FLAGS.acquisition_fn_optimiser == 'sobol':
            optimizer = generate_sobol_random_search_optimizer(num_samples=FLAGS.num_acquisition_optimiser_start_points)
        elif FLAGS.acquisition_fn_optimiser == 'random':
            optimizer = generate_random_search_optimizer(num_samples=FLAGS.num_acquisition_optimiser_start_points)
        elif FLAGS.acquisition_fn_optimiser == 'adam':
            optimizer = generate_al_adam_optimizer(num_initial_samples=FLAGS.num_acquisition_optimiser_start_points)

        rule = ALEfficientGlobalOptimization(augmented_lagrangian, optimizer=optimizer, num_query_points=FLAGS.batch_size)
        bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
        data = bo.optimize(FLAGS.num_bo_iterations, initial_data, initial_models, rule,
                           track_state=True).try_get_final_datasets()
        with open(FLAGS.save_path + f"{run}_data.pkl", "wb") as fp:
            pickle.dump(data, fp)


if __name__ == "__main__":
    app.run(main)
