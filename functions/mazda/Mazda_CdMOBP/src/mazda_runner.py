import subprocess
import tensorflow as tf
from trieste.types import TensorType, Tag
from trieste.data import Dataset
from typing import Mapping
import os
import numpy as np


def mazda_observer(query_points: TensorType) -> Mapping[Tag, Dataset]:
    """
    Takes input values of shape [N, 222] and returns tagged Dataset with objective and constraint
    observations at given query points.
    """
    initial_wd = os.getcwd()

    # Run Mazda executable from its own directory
    os.chdir("/Users/thomaschristie/Documents/GitHub.nosync/trieste/functions/mazda/Mazda_CdMOBP/src/")
    query_points_lower_bounds = tf.constant([[0.9, 0.3, 1.1, 0.35, 1.5, 0.5, 1.3, 1.7, 1.1, 0.7, 1.7, 0.5, 1.1, 1.3, 1.1, 1.3, 0.9, 1.1, 1.5, 0.9, 0.9, 0.5, 0.5, 0.9, 0.3, 0.4, 1.3, 0.6, 0.6, 0.4, 0.9, 0.6, 0.9, 0.7, 0.4, 1.5, 0.6, 0.3, 0.3, 0.6, 1.7, 0.9, 0.5, 1.3, 0.7, 0.7, 0.9, 1.3, 1.1, 1.3, 2.0, 1.7, 1.1, 0.9, 0.3, 0.6, 0.9, 1.1, 1.3, 0.3, 0.3, 1.3, 0.9, 1.1, 1.7, 1.5, 1.1, 0.9, 0.7, 1.1, 0.9, 1.7, 2.0, 0.5, 0.9, 0.3, 1.1, 0.35, 1.5, 0.5, 1.3, 1.7, 1.1, 0.7, 1.7, 0.5, 1.1, 1.3, 1.1, 1.3, 0.9, 1.1, 1.5, 0.9, 0.9, 0.5, 0.5, 0.9, 0.3, 0.4, 1.3, 0.6, 0.6, 0.4, 0.9, 0.6, 0.9, 0.7, 0.4, 1.5, 0.6, 0.3, 0.3, 0.6, 1.7, 0.9, 0.5, 1.3, 0.7, 0.7, 0.9, 1.3, 1.1, 1.3, 2.0, 1.7, 1.1, 0.9, 0.3, 0.6, 0.9, 1.1, 1.3, 0.3, 0.3, 1.3, 0.9, 1.1, 1.7, 1.5, 1.1, 0.9, 0.7, 1.1, 0.9, 1.7, 2.0, 0.5, 0.9, 0.3, 1.1, 0.35, 1.5, 0.5, 1.3, 1.7, 1.1, 0.7, 1.7, 0.5, 1.1, 1.3, 1.1, 1.3, 0.9, 1.1, 1.5, 0.9, 0.9, 0.5, 0.5, 0.9, 0.3, 0.4, 1.3, 0.6, 0.6, 0.4, 0.9, 0.6, 0.9, 0.7, 0.4, 1.5, 0.6, 0.3, 0.3, 0.6, 1.7, 0.9, 0.5, 1.3, 0.7, 0.7, 0.9, 1.3, 1.1, 1.3, 2.0, 1.7, 1.1, 0.9, 0.3, 0.6, 0.9, 1.1, 1.3, 0.3, 0.3, 1.3, 0.9, 1.1, 1.7, 1.5, 1.1, 0.9, 0.7, 1.1, 0.9, 1.7, 2.0, 0.5]], dtype=tf.float64)
    query_points_upper_bounds = tf.constant([[1.5, 0.95, 2.1, 1.0, 2.3, 2.1, 2.1, 2.3, 1.7, 1.5, 2.6, 1.3, 1.7, 1.9, 1.7, 1.9, 2.1, 1.9, 2.3, 1.5, 1.5, 1.1, 1.2, 2.1, 0.9, 1.0, 1.9, 1.3, 1.2, 1.7, 1.9, 1.3, 1.7, 1.3, 1.9, 2.3, 1.2, 0.9, 0.9, 1.2, 2.3, 1.7, 1.1, 1.9, 1.7, 2.1, 2.1, 2.1, 1.9, 2.1, 2.6, 2.3, 1.7, 2.1, 0.9, 1.3, 1.7, 1.9, 1.9, 1.3, 1.1, 1.9, 1.5, 1.7, 2.3, 2.1, 1.6, 1.7, 2.1, 1.9, 1.7, 2.3, 2.6, 1.3, 1.5, 0.95, 2.1, 1.0, 2.3, 2.1, 2.1, 2.3, 1.7, 1.5, 2.6, 1.3, 1.7, 1.9, 1.7, 1.9, 2.1, 1.9, 2.3, 1.5, 1.5, 1.1, 1.2, 2.1, 0.9, 1.0, 1.9, 1.3, 1.2, 1.7, 1.9, 1.3, 1.7, 1.3, 1.9, 2.3, 1.2, 0.9, 0.9, 1.2, 2.3, 1.7, 1.1, 1.9, 1.7, 2.1, 2.1, 2.1, 1.9, 2.1, 2.6, 2.3, 1.7, 2.1, 0.9, 1.3, 1.7, 1.9, 1.9, 1.3, 1.1, 1.9, 1.5, 1.7, 2.3, 2.1, 1.6, 1.7, 2.1, 1.9, 1.7, 2.3, 2.6, 1.3, 1.5, 0.95, 2.1, 1.0, 2.3, 2.1, 2.1, 2.3, 1.7, 1.5, 2.6, 1.3, 1.7, 1.9, 1.7, 1.9, 2.1, 1.9, 2.3, 1.5, 1.5, 1.1, 1.2, 2.1, 0.9, 1.0, 1.9, 1.3, 1.2, 1.7, 1.9, 1.3, 1.7, 1.3, 1.9, 2.3, 1.2, 0.9, 0.9, 1.2, 2.3, 1.7, 1.1, 1.9, 1.7, 2.1, 2.1, 2.1, 1.9, 2.1, 2.6, 2.3, 1.7, 2.1, 0.9, 1.3, 1.7, 1.9, 1.9, 1.3, 1.1, 1.9, 1.5, 1.7, 2.3, 2.1, 1.6, 1.7, 2.1, 1.9, 1.7, 2.3, 2.6, 1.3]], dtype=tf.float64)
    scaled_query_points = query_points_lower_bounds + (query_points_upper_bounds - query_points_lower_bounds) * query_points
    np_scaled_query_points = scaled_query_points.numpy()
    with open('pop_vars_eval.txt', 'w') as fp:
        for scaled_query_point in np_scaled_query_points:
            fp.write("\t".join(str(x) for x in scaled_query_point))
            fp.write("\n")

    subprocess.run(["./mazda_mop ."], shell=True)

    # Read objective vals
    objective_vals = []
    with open("pop_objs_eval.txt", "r") as fp:
        for line in fp:
            objective_vals.append(float(line.split()[0]))

    # Read constraint vals
    constraint_vals = {f"INEQUALITY_CONSTRAINT_{i}": [] for i in range(1, 55)}
    with open("pop_cons_eval.txt", "r") as fp:
        for line in fp:
            for i, constraint_val in enumerate(line.split()):
                # Multiply by -1.0 as Mazda constraints are satisfied for x >= 0, the opposite of trieste
                constraint_vals[f"INEQUALITY_CONSTRAINT_{i+1}"].append(-1.0 * float(constraint_val))

    tagged_observations = {f"INEQUALITY_CONSTRAINT_{i}": Dataset(query_points, tf.Variable(constraint_vals[f"INEQUALITY_CONSTRAINT_{i}"], dtype=tf.float64)[..., None]) for i in range(1, 55)}
    tagged_observations["OBJECTIVE"] = Dataset(query_points, tf.Variable(objective_vals, dtype=tf.float64)[..., None])

    # Revert to initial directory
    os.chdir(initial_wd)
    return tagged_observations