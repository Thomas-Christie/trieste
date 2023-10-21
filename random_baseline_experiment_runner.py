from absl import app, flags
import tensorflow as tf
import numpy as np
import trieste
from trieste.space import Box
from functions import constraints
from functions import objectives
from functions.lockwood.runlock.runlock import lockwood_constraint_observer
from functions.mazda.Mazda_CdMOBP.src.mazda_runner import mazda_observer
import pickle

OBJECTIVE = "OBJECTIVE"
INEQUALITY_CONSTRAINT_ONE = "INEQUALITY_CONSTRAINT_ONE"
INEQUALITY_CONSTRAINT_TWO = "INEQUALITY_CONSTRAINT_TWO"
EQUALITY_CONSTRAINT_ONE = "EQUALITY_CONSTRAINT_ONE"
EQUALITY_CONSTRAINT_TWO = "EQUALITY_CONSTRAINT_TWO"

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_experiments', 30, 'Number of repeats of experiment to run.')
flags.DEFINE_integer('num_samples', 400, 'Number of random points to sample.')
flags.DEFINE_enum('problem', 'LOCKWOOD', ['LSQ', 'GSBP', 'LOCKWOOD', 'ACKLEY10', 'KEANE30', 'MAZDA'], 'Test problem to use.')
flags.DEFINE_string('save_path', 'random_baseline_results/lockwood/data/', 'Prefix of path to save results to.')


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
                tf.zeros(6, dtype=tf.float64), tf.ones(6, dtype=tf.float64)
            )
        elif FLAGS.problem == "MAZDA":
            search_space = Box(
                tf.zeros(222, dtype=tf.float64), tf.ones(222, dtype=tf.float64)
            )
        elif FLAGS.problem == "ACKLEY10":
            search_space = Box(
                tf.zeros(10, dtype=tf.float64), tf.ones(10, dtype=tf.float64)
            )
        elif FLAGS.problem == "KEANE30":
            search_space = Box(
                tf.zeros(30, dtype=tf.float64), tf.ones(30, dtype=tf.float64)
            )
        else:
            search_space = Box([0.0, 0.0], [1.0, 1.0])

        if FLAGS.problem == "ACKLEY10":
            observer = trieste.objectives.utils.mk_multi_observer(
                OBJECTIVE=objectives.ackley_10,
                INEQUALITY_CONSTRAINT_ONE=constraints.ackley_10_constraint_one,
                INEQUALITY_CONSTRAINT_TWO=constraints.ackley_10_constraint_two,
            )
        elif FLAGS.problem == "KEANE30":
            observer = trieste.objectives.utils.mk_multi_observer(
                OBJECTIVE=objectives.keane_bump_30,
                INEQUALITY_CONSTRAINT_ONE=constraints.keane_bump_30_constraint_one,
                INEQUALITY_CONSTRAINT_TWO=constraints.keane_bump_30_constraint_two,
            )
        elif FLAGS.problem == "LOCKWOOD":
            observer = lockwood_constraint_observer
        elif FLAGS.problem == "MAZDA":
            observer = mazda_observer

        initial_inputs = search_space.sample(FLAGS.num_samples, seed=42 + run)
        data = observer(initial_inputs)

        with open(FLAGS.save_path + f"run_{run}_data.pkl", "wb") as fp:
            pickle.dump(data, fp)


if __name__ == "__main__":
    app.run(main)