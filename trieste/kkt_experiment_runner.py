from absl import app, flags
import gpflow
import tensorflow as tf
import numpy as np
import trieste
from trieste.acquisition.optimizer import generate_continuous_optimizer, generate_kkt_continuous_optimizer
from trieste.acquisition.function.kkt_expected_improvement import KKTExpectedImprovement, KKTThompsonSamplingExpectedImprovement
from trieste.acquisition.rule import KKTEfficientGlobalOptimization
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.space import Box
from functions import constraints
from functions import objectives
from functools import partial
import pickle

OBJECTIVE = "OBJECTIVE"
INEQUALITY_CONSTRAINT_ONE = "INEQUALITY_CONSTRAINT_ONE"
INEQUALITY_CONSTRAINT_TWO = "INEQUALITY_CONSTRAINT_TWO"
EQUALITY_CONSTRAINT_ONE = "EQUALITY_CONSTRAINT_ONE"
EQUALITY_CONSTRAINT_TWO = "EQUALITY_CONSTRAINT_TWO"

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_experiments', 100, 'Number of repeats of experiment to run.')
flags.DEFINE_integer('num_bo_iterations', 40, 'Number of iterations of Bayesian optimisation to run for.')
flags.DEFINE_float('epsilon', 0.01, 'Bound within which equality constraints are considered to be satisfied.')
flags.DEFINE_enum('problem', 'LSQ', ['LSQ', 'GSBP'], 'Test problem to use.')
flags.DEFINE_integer('num_rff_features', 1000, 'Number of Random Fourier Features to use when approximating the kernel.')
flags.DEFINE_integer('num_initial_samples', 5, 'Number of random samples to fit models before starting BO.')
flags.DEFINE_float('ei_epsilon', 0.001, 'Fractional improvement over current best valid observed point required to'
                                        'run simulation with newly suggested point returned by acquisition function.')
flags.DEFINE_boolean('feasible_region_only', False, 'Whether to only consider region where all surrogate models for '
                                                   'constraints are satisfied when generating acquisition function.')
flags.DEFINE_boolean('thompson_sampling', False, 'Whether to use Thompson Sampling to model constraints, so that more'
                                                 'distributional information can be utilised in gradient calculations.')
flags.DEFINE_float('initial_alpha', 0.2, 'Initial value of alpha for considering whether constraints are binding.')
flags.DEFINE_enum('sampling_strategy', 'sobol', ['sobol', 'uniform_random'], 'Random sampling strategy for selecting '
                                                                             'initial points.')
flags.DEFINE_integer('num_acquisition_optimiser_start_points', 5000, 'Number of starting points to randomly sample from'
                                                                     'acquisition function when optimising it.')
flags.DEFINE_float('alpha_lower_bound', 0.01, 'Lower bound on alpha for considering whether constraints are binding.')
flags.DEFINE_enum('kernel_name', 'squared_exponential', ['matern52', 'squared_exponential'], 'Which kernel to use.')
flags.DEFINE_string('save_path', 'results/extra_kkt_results/lsq_refined/data/run_', 'Prefix of path to save results to.')


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
            initial_inputs = search_space.sample(FLAGS.num_initial_samples, seed=run+42)
        elif FLAGS.sampling_strategy == 'sobol':
            initial_inputs = search_space.sample_sobol(FLAGS.num_initial_samples, skip=42+run*FLAGS.num_initial_samples)
        print(f"Initial Inputs: {initial_inputs}")
        initial_data = observer(initial_inputs)
        initial_models = trieste.utils.map_values(partial(create_model, search_space, FLAGS.num_rff_features, FLAGS.kernel_name),
                                                  initial_data)

        if FLAGS.problem == "LSQ":
            if FLAGS.thompson_sampling:
                kkt_expected_improvement = KKTThompsonSamplingExpectedImprovement(objective_tag=OBJECTIVE,
                                                                                  inequality_constraint_prefix="INEQUALITY",
                                                                                  equality_constraint_prefix=None,
                                                                                  epsilon=FLAGS.epsilon,
                                                                                  search_space=search_space,
                                                                                  feasible_region_only=FLAGS.feasible_region_only,
                                                                                  plot=False)
            else:
                kkt_expected_improvement = KKTExpectedImprovement(objective_tag=OBJECTIVE,
                                                                  inequality_constraint_prefix="INEQUALITY",
                                                                  equality_constraint_prefix=None,
                                                                  epsilon=FLAGS.epsilon,
                                                                  search_space=search_space,
                                                                  feasible_region_only=FLAGS.feasible_region_only,
                                                                  plot=False)
        elif FLAGS.problem == "GSBP":
            if FLAGS.thompson_sampling:
                kkt_expected_improvement = KKTThompsonSamplingExpectedImprovement(objective_tag=OBJECTIVE,
                                                                                  inequality_constraint_prefix="INEQUALITY",
                                                                                  equality_constraint_prefix="EQUALITY",
                                                                                  epsilon=FLAGS.epsilon,
                                                                                  search_space=search_space,
                                                                                  feasible_region_only=FLAGS.feasible_region_only,
                                                                                  plot=False)
            else:
                kkt_expected_improvement = KKTExpectedImprovement(objective_tag=OBJECTIVE,
                                                                  inequality_constraint_prefix="INEQUALITY",
                                                                  equality_constraint_prefix="EQUALITY",
                                                                  epsilon=FLAGS.epsilon,
                                                                  search_space=search_space,
                                                                  feasible_region_only=FLAGS.feasible_region_only,
                                                                  plot=False)

        optimizer = generate_kkt_continuous_optimizer(num_initial_samples=FLAGS.num_acquisition_optimiser_start_points, num_optimization_runs=1)
        rule = KKTEfficientGlobalOptimization(kkt_expected_improvement, optimizer=optimizer, epsilon=FLAGS.ei_epsilon)
        kkt_bo = trieste.kkt_bayesian_optimizer.KKTBayesianOptimizer(observer, search_space)
        data = kkt_bo.optimize(FLAGS.num_bo_iterations, initial_data, initial_models, rule, track_state=True,
                               initial_alpha=FLAGS.initial_alpha,
                               alpha_lower_bound=FLAGS.alpha_lower_bound).try_get_final_datasets()
        with open(FLAGS.save_path + f"{run}_data.pkl", "wb") as fp:
            pickle.dump(data, fp)


if __name__ == "__main__":
    app.run(main)
