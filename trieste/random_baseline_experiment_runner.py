from absl import app, flags
import tensorflow as tf
import numpy as np
import trieste
from trieste.space import Box
from functions import constraints
from functions import objectives
from functions.lockwood.runlock.runlock import lockwood_constraint_observer
import pickle

OBJECTIVE = "OBJECTIVE"
INEQUALITY_CONSTRAINT_ONE = "INEQUALITY_CONSTRAINT_ONE"
INEQUALITY_CONSTRAINT_TWO = "INEQUALITY_CONSTRAINT_TWO"
EQUALITY_CONSTRAINT_ONE = "EQUALITY_CONSTRAINT_ONE"
EQUALITY_CONSTRAINT_TWO = "EQUALITY_CONSTRAINT_TWO"

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_experiments', 30, 'Number of repeats of experiment to run.')
flags.DEFINE_integer('num_samples', 510, 'Number of random points to sample.')
flags.DEFINE_enum('problem', 'LOCKWOOD', ['LSQ', 'GSBP', 'LOCKWOOD'], 'Test problem to use.')
flags.DEFINE_string('save_path', 'results/20-04-23/lockwood_random/data/', 'Prefix of path to save results to.')


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

        initial_inputs = search_space.sample(FLAGS.num_samples, seed=42 + run)
        data = observer(initial_inputs)

        with open(FLAGS.save_path + f"run_{run}_data.pkl", "wb") as fp:
            pickle.dump(data, fp)


if __name__ == "__main__":
    app.run(main)