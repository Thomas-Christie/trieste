from absl import app, flags
import tensorflow as tf
import numpy as np
import trieste
from trieste.acquisition.optimizer import generate_continuous_optimizer
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

flags.DEFINE_integer('num_experiments', 50, 'Number of repeats of experiment to run.')
flags.DEFINE_integer('num_bo_iterations', 150, 'Number of iterations of Bayesian optimisation to run for.')
flags.DEFINE_float('epsilon', 0.01, 'Bound within which equality constraints are considered to be satisfied.')
flags.DEFINE_enum('problem', 'GSBP', ['LSQ', 'GSBP'], 'Test problem to use.')
flags.DEFINE_integer('num_rff_features', 1000, 'Number of Random Fourier Features to use when approximating the kernel.')
flags.DEFINE_integer('num_initial_samples', 10, 'Number of random samples to fit models before starting BO.')
flags.DEFINE_float('ei_epsilon', 0.001, 'Fractional improvement over current best valid observed point required to'
                                          'run simulation with newly suggested point returned by acquisition function.')
flags.DEFINE_boolean('feasible_region_only', True, 'Whether to only consider region where all surrogate models for '
                                                   'constraints are satisfied when generating acquisition function.')
flags.DEFINE_boolean('thompson_sampling', True, 'Whether to use Thompson Sampling to model constraints, so that more'
                                                 'distributional information can be utilised in gradient calculations.')
flags.DEFINE_float('initial_alpha', 0.2, 'Initial value of alpha for considering whether constraints are binding.')
flags.DEFINE_float('alpha_lower_bound', 0.01, 'Lower bound on alpha for considering whether constraints are binding.')
flags.DEFINE_string('save_path', 'results/09-04-23/kkt_gsbp_feasible_only_ts/data/run_', 'Prefix of path to save results to.')


def create_model(search_space, num_rff_features, data):
    gpr = build_gpr(data, search_space, likelihood_variance=1e-7)
    return GaussianProcessRegression(gpr, num_rff_features=num_rff_features)


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

        initial_inputs = search_space.sample(FLAGS.num_initial_samples)
        initial_data = observer(initial_inputs)
        initial_models = trieste.utils.map_values(partial(create_model, search_space, FLAGS.num_rff_features),
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

        rule = KKTEfficientGlobalOptimization(kkt_expected_improvement, optimizer=generate_continuous_optimizer(), epsilon=FLAGS.ei_epsilon)
        kkt_bo = trieste.kkt_bayesian_optimizer.KKTBayesianOptimizer(observer, search_space)
        data = kkt_bo.optimize(FLAGS.num_bo_iterations, initial_data, initial_models, rule, track_state=True,
                               initial_alpha=FLAGS.initial_alpha,
                               alpha_lower_bound=FLAGS.alpha_lower_bound).try_get_final_datasets()
        with open(FLAGS.save_path + f"{run}_data.pkl", "wb") as fp:
            pickle.dump(data, fp)


if __name__ == "__main__":
    app.run(main)
