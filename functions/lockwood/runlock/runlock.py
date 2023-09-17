import subprocess
import tensorflow as tf
from trieste.types import TensorType, Tag
from trieste.data import Dataset
from typing import Mapping
import os


def lockwood_constraint_observer(query_points: TensorType) -> Mapping[Tag, Dataset]:
    """
    Takes input values of shape [N, 6] and returns tagged Dataset with objective and constraint
    observations at given query points.
    """
    initial_wd = os.getcwd()

    # Lockwood executable needs to be run from its own directory
    os.chdir("/Users/thomaschristie/Documents/GitHub.nosync/trieste/functions/lockwood/runlock/")
    observations = [[], [], []]
    scaled_query_points = query_points * 20000
    for scaled_query_point in scaled_query_points:
        # Write query point to 'input.tst' file
        with open('python_input.tst', 'w') as fp:
            fp.write(f"{6}\n")
            for dim in scaled_query_point:
                fp.write(f"{dim}\n")
        subprocess.run(["./RunLock python_input.tst python_output.tst 1"], shell=True)

        with open('python_output.tst', 'r') as fp:
            i = 0
            for line in fp:
                # Scale the outputs into sensible ranges - divide objective by 10000 and constraints by 1000
                scaling_factor = 10000 if i == 0 else 1000
                observations[i].append(float(line.split()[0])/scaling_factor)
                i += 1

    tagged_observations = {'OBJECTIVE': Dataset(query_points, tf.Variable(observations[0], dtype=tf.float64)[..., None]),
                           'INEQUALITY_CONSTRAINT_ONE': Dataset(query_points, tf.Variable(observations[1], dtype=tf.float64)[..., None]),
                           'INEQUALITY_CONSTRAINT_TWO': Dataset(query_points, tf.Variable(observations[2], dtype=tf.float64)[..., None])}

    # Revert to initial directory
    os.chdir(initial_wd)
    return tagged_observations