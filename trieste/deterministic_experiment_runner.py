from absl import app, flags
import tensorflow as tf
import numpy as np
import trieste
from trieste.acquisition.optimizer import generate_al_continuous_optimizer
from trieste.acquisition.function.new_constrained_thompson_sampling import DeterministicAugmentedLagrangian
from trieste.acquisition.rule import ALEfficientGlobalOptimization
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.space import Box
from functions import constraints
from functions import objectives
from functools import partial

OBJECTIVE = "OBJECTIVE"
INEQUALITY_CONSTRAINT_ONE = "INEQUALITY_CONSTRAINT_ONE"
INEQUALITY_CONSTRAINT_TWO = "INEQUALITY_CONSTRAINT_TWO"
EQUALITY_CONSTRAINT_ONE = "EQUALITY_CONSTRAINT_ONE"
EQUALITY_CONSTRAINT_TWO = "EQUALITY_CONSTRAINT_TWO"

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_experiments', 50, 'Number of repeats of experiment to run.')
flags.DEFINE_integer('num_bo_iterations', 500, 'Number of iterations of Bayesian optimisation to run for.')
flags.DEFINE_float('epsilon', 0.01, 'Bound within which equality constraints are considered to be satisfied.')
flags.DEFINE_enum('problem', 'LSQ', ['LSQ', 'GSBP', 'LOCKWOOD'], 'Test problem to use.')
flags.DEFINE_integer('num_rff_features', 1000,
                     'Number of Random Fourier Features to use when approximating the kernel.')
flags.DEFINE_integer('batch_size', 1, 'Number of points to sample at each iteration of BO.')
flags.DEFINE_integer('num_initial_samples', 10, 'Number of random samples to fit models before starting BO.')
flags.DEFINE_boolean('update_lagrange_via_kkt', False, 'Whether to update Lagrange multipliers using a gradient-based'
                                                       'approach based on KKT conditions.')
flags.DEFINE_boolean('conservative_penalty_decrease', False,
                     'Whether to reduce the penalty parameter more conservatively'
                     'if no valid solutions have been found yet.')
flags.DEFINE_boolean('fully_consistent', True,
                     'Whether to update Lagrange multipliers and penalty parameter in a mannner'
                     'which is fully consistent with the appendix of the original paper (which '
                     'differs from the body of the original paper).')
flags.DEFINE_enum('sampling_strategy', 'sobol', ['sobol', 'uniform_random'], 'Random sampling strategy for selecting '
                                                                             'initial points.')
flags.DEFINE_boolean('save_lagrange', False, 'Save intermediate values of Lagrange multipliers.')


def create_model(search_space, num_rff_features, data):
    gpr = build_gpr(data, search_space, likelihood_variance=1e-7)
    return GaussianProcessRegression(gpr, num_rff_features=num_rff_features)


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main(argv):
    print(f"Running Experiment with Flags: {FLAGS.flags_into_string()}")
    for run in range(2, FLAGS.num_experiments):
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

        if FLAGS.sampling_strategy == 'uniform_random':
            initial_inputs = search_space.sample(FLAGS.num_initial_samples, seed=run + 42)
        elif FLAGS.sampling_strategy == 'sobol':
            initial_inputs = search_space.sample_sobol(FLAGS.num_initial_samples,
                                                       skip=42 + run * FLAGS.num_initial_samples)
        print(f"Initial Inputs: {initial_inputs}")
        initial_data = observer(initial_inputs)
        initial_models = trieste.utils.map_values(partial(create_model, search_space, FLAGS.num_rff_features),
                                                  initial_data)

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

        if FLAGS.problem == "GSBP":
            if FLAGS.fully_consistent:
                equality_lambda = {EQUALITY_CONSTRAINT_ONE: tf.zeros(1, dtype=tf.float64),
                                   EQUALITY_CONSTRAINT_TWO: tf.zeros(1, dtype=tf.float64)}
            else:
                equality_lambda = {EQUALITY_CONSTRAINT_ONE: tf.zeros((1, FLAGS.batch_size, 1), dtype=tf.float64),
                                   EQUALITY_CONSTRAINT_TWO: tf.zeros((1, FLAGS.batch_size, 1), dtype=tf.float64)}


        lambda_save_path = None

        if FLAGS.problem == "LSQ":
            if FLAGS.fully_consistent:
                augmented_lagrangian = DeterministicAugmentedLagrangian(objective_tag=OBJECTIVE,
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
                                                                        num_bo_iters=FLAGS.num_bo_iterations,
                                                                        gsbp=False)
        elif FLAGS.problem == 'GSBP':
            if FLAGS.fully_consistent:
                augmented_lagrangian = DeterministicAugmentedLagrangian(objective_tag=OBJECTIVE,
                                                                        inequality_constraint_prefix="INEQUALITY",
                                                                        equality_constraint_prefix="EQUALITY",
                                                                        inequality_lambda=inequality_lambda,
                                                                        equality_lambda=equality_lambda,
                                                                        batch_size=FLAGS.batch_size,
                                                                        penalty=None,
                                                                        conservative_penalty_decrease=FLAGS.conservative_penalty_decrease,
                                                                        epsilon=FLAGS.epsilon,
                                                                        update_lagrange_via_kkt=FLAGS.update_lagrange_via_kkt,
                                                                        search_space=search_space,
                                                                        plot=False,
                                                                        save_lambda=FLAGS.save_lagrange,
                                                                        save_path=lambda_save_path,
                                                                        num_bo_iters=FLAGS.num_bo_iterations,
                                                                        gsbp=True)

        rule = ALEfficientGlobalOptimization(augmented_lagrangian, optimizer=generate_al_continuous_optimizer(),
                                             num_query_points=FLAGS.batch_size)
        bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
        data = bo.optimize(FLAGS.num_bo_iterations, initial_data, initial_models, rule,
                           track_state=True).try_get_final_datasets()


if __name__ == "__main__":
    app.run(main)
