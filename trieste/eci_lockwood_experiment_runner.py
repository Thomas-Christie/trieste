import gpflow.mean_functions
from absl import app, flags
import tensorflow as tf
import numpy as np
import trieste
from trieste.acquisition.optimizer import generate_kkt_continuous_optimizer
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.combination import Product
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.space import Box
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
flags.DEFINE_integer('num_rff_features', 1000, 'Number of Random Fourier Features to use when approximating the kernel.')
flags.DEFINE_integer('num_initial_samples', 30, 'Number of random samples to fit models before starting BO.')
flags.DEFINE_enum('sampling_strategy', 'uniform_random', ['sobol', 'uniform_random'], 'Random sampling strategy for selecting '
                                                                                      'initial points.')
flags.DEFINE_integer('num_acquisition_optimiser_start_points', 6000, 'Number of starting points to randomly sample from'
                                                                     'acquisition function when optimising it.')
flags.DEFINE_enum('kernel_name', 'squared_exponential', ['matern52', 'squared_exponential'], 'Which kernel to use.')
flags.DEFINE_string('save_path', 'results/eci_results/lockwood/data/run_', 'Prefix of path to save results to.')


def create_model(search_space, num_rff_features, kernel_name, data):
    gpr = build_gpr(data, search_space, likelihood_variance=1e-6, mean=gpflow.mean_functions.Zero(), kernel_name=kernel_name)
    return GaussianProcessRegression(gpr, num_rff_features=num_rff_features, use_decoupled_sampler=True)


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main(argv):
    print(f"Running Experiment with Flags: {FLAGS.flags_into_string()}")
    for run in range(20, FLAGS.num_experiments):
        print(f"Starting Run: {run}")
        set_seed(run + 42)
        search_space = Box([0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        observer = lockwood_constraint_observer

        if FLAGS.sampling_strategy == 'uniform_random':
            initial_inputs = search_space.sample(FLAGS.num_initial_samples, seed=run+42)
        elif FLAGS.sampling_strategy == 'sobol':
            initial_inputs = search_space.sample_sobol(FLAGS.num_initial_samples, skip=42+run*FLAGS.num_initial_samples)
        print(f"Initial Inputs: {initial_inputs}")
        initial_data = observer(initial_inputs)
        initial_models = trieste.utils.map_values(partial(create_model, search_space, FLAGS.num_rff_features, FLAGS.kernel_name),
                                                  initial_data)

        optimizer = generate_kkt_continuous_optimizer(num_initial_samples=FLAGS.num_acquisition_optimiser_start_points,
                                                      num_optimization_runs=1)

        pof1 = trieste.acquisition.ProbabilityOfFeasibility(threshold=0)
        pof2 = trieste.acquisition.ProbabilityOfFeasibility(threshold=0)
        pof = Product(pof1.using(INEQUALITY_CONSTRAINT_ONE), pof2.using(INEQUALITY_CONSTRAINT_TWO))
        eci = trieste.acquisition.ExpectedConstrainedImprovement(OBJECTIVE, pof)
        rule = EfficientGlobalOptimization(eci, optimizer=optimizer)
        bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
        data = bo.optimize(FLAGS.num_bo_iterations, initial_data, initial_models, rule,
                           track_state=False).try_get_final_datasets()
        with open(FLAGS.save_path + f"{run}_data.pkl", "wb") as fp:
            pickle.dump(data, fp)


if __name__ == "__main__":
    app.run(main)
