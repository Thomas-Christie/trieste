import tensorflow as tf
import trieste
import gpflow
from trieste.models.gpflow import GaussianProcessRegression
from trieste.space import Box
from trieste.data import Dataset
from trieste.observer import MultiObserver
from trieste.acquisition.optimizer import generate_continuous_optimizer
from typing import Optional, Mapping, Hashable
import numpy as np
import updated_lagrangians
from functions import constraints
from functions import objectives
from functools import partial


np.random.seed(1793)
tf.random.set_seed(1793)
NUM_INITIAL_SAMPLES = 10
OBJECTIVE = "OBJECTIVE"
INEQUALITY_CONSTRAINT_ONE = "INEQUALITY_CONSTRAINT_ONE"
INEQUALITY_CONSTRAINT_TWO = "INEQUALITY_CONSTRAINT_TWO"


def update_and_optimise_models(models: Mapping[str, GaussianProcessRegression],
                               datasets: Mapping[str, Dataset]):
    # Optimise models based on updated datasets
    for tag, model in models.items():
        data = datasets[tag]
        model.update(data)
        model.optimize(data)

def augmented_lagrangian_optimiser_unknown_functions(search_space: Box,
                                                     inequality_prefix: Optional[str],
                                                     equality_prefix: Optional[str],
                                                     observer: MultiObserver,
                                                     models: Mapping[str, GaussianProcessRegression],
                                                     initial_dataset: Dataset,
                                                     equality_lambd: Optional[tf.Tensor],
                                                     inequality_lambd: Optional[tf.Tensor],
                                                     penalty: tf.Variable,
                                                     epsilon: float,
                                                     num_iterations: int) -> tf.Tensor:
    """
    Optimise an augmented Lagrangian modelled with GPs
    :param objective_observer: Objective function
    :param inequality_observer: Inequality constraint functions
    :param equality_observer: Equality constraint functions
    :param objective_model: GP model representing objective function
    :param inequality_constraint_models: GP models representing inequality constraint functions
    :param equality_constraint_models: GP models representing equality constraint functions
    :param initial_objective_dataset: Initial data sampled from objective function
    :param initial_inequality_constraint_datasets: Initial data sampled from inequality constraint functions
    :param initial_equality_constraint_datasets: Initial data sampled from equality constraint functions
    :param equality_lambd: Array of initial Lagrange multipliers for equality constraints of shape ["num_eq_constraints", 1]
    :param inequality_lambd: Array of initial Lagrange multipliers for inequality constraints of shape ["num_ineq_constraints", 1]
    :param penalty: Initial penalty parameter for violated constraints
    :param epsilon: Bound for equality constraint being satisfied
    :param num_iterations: Number of iterations to run optimisation loop for
    :return: Optimal x value
    """
    datasets = initial_dataset

    update_and_optimise_models(models, datasets)

    objective_trajectory_sampler = models[OBJECTIVE].trajectory_sampler()
    objective_trajectory = objective_trajectory_sampler.get_trajectory()
    inequality_constraint_trajectory_samplers = {}
    inequality_constraint_trajectories = {}
    equality_constraint_trajectory_samplers = {}
    equality_constraint_trajectories = {}

    if inequality_prefix is not None:
        for tag, model in models.items():
            if tag.startswith(inequality_prefix):
                sampler = model.trajectory_sampler()
                trajectory = sampler.get_trajectory()
                inequality_constraint_trajectory_samplers[tag] = sampler
                inequality_constraint_trajectories[tag] = trajectory

    if equality_prefix is not None:
        for tag, model in models.items():
            if tag.startswith(equality_prefix):
                sampler = model.trajectory_sampler()
                trajectory = sampler.get_trajectory()
                equality_constraint_trajectory_samplers[tag] = sampler
                equality_constraint_trajectories[tag] = trajectory

    for i in range(num_iterations):
        augmented_lagrangian = partial(updated_lagrangians.augmented_lagrangian, objective_trajectory, list(inequality_constraint_trajectories.values()),
                                                                              list(equality_constraint_trajectories.values()), equality_lambd, inequality_lambd,
                                                                              penalty)

        continuous_optimiser = generate_continuous_optimizer()
        opt_x = continuous_optimiser(search_space, augmented_lagrangian)

        observed_opt_x = observer(opt_x)
        datasets = {tag: datasets[tag] + observed_opt_x[tag] for tag in observed_opt_x}

        inequality_constraints_satisfied = True
        equality_constraints_satisfied = True

        # TODO: Rename to make clear it uses opt_x?
        if inequality_prefix is not None:
            inequality_constraint_tags = [key for key in datasets.keys() if key.startswith(inequality_prefix)]
            for j in range(len(inequality_lambd)):
                inequality_constraint_val = datasets[inequality_constraint_tags[j]].observations[-1][None, ...]
                slack_val = updated_lagrangians.obtain_single_slack(inequality_constraint_val, inequality_lambd[j], penalty)
                updated_multiplier = inequality_lambd[j][0] + (1 / penalty) * (inequality_constraint_val + slack_val)
                inequality_lambd = tf.tensor_scatter_nd_update(inequality_lambd, [[j]], updated_multiplier)
                # print(f"Inequality Lambda: {inequality_lambd}")

            inequality_constraints_opt_x = tf.stack([datasets[tag].observations[-1][None, ...] for tag in inequality_constraint_tags])
            num_inequality_constraints_satisfied = tf.reduce_sum(tf.cast(inequality_constraints_opt_x <= 0, tf.int8))
            if num_inequality_constraints_satisfied != len(inequality_constraint_tags):
                inequality_constraints_satisfied = False


        if equality_prefix is not None:
            equality_constraint_tags = [key for key in datasets.keys() if key.startswith(equality_prefix)]
            for j in range(len(equality_lambd)):
                equality_constraint_val = datasets[equality_constraint_tags[j]].observations[-1][None, ...]
                updated_multiplier = equality_lambd[j][0] + (1 / penalty) * equality_constraint_val
                equality_lambd = tf.tensor_scatter_nd_update(equality_lambd, [[j]], updated_multiplier)

            equality_constraints_opt_x = tf.stack([datasets[tag].observations[-1][None, ...] for tag in equality_constraint_tags])
            num_equality_constraints_satisfied = tf.reduce_sum(tf.cast(tf.abs(equality_constraints_opt_x) <= epsilon, tf.int8))
            if num_equality_constraints_satisfied != len(equality_constraint_tags):
                equality_constraints_satisfied = False

        if not (equality_constraints_satisfied and inequality_constraints_satisfied):
            print(f"Not Satisfied")
            penalty = penalty / 2
        else:
            print(f"Satisfied")


        print(f"Iteration {i} Objective X: {observed_opt_x[OBJECTIVE].query_points} Value: {observed_opt_x[OBJECTIVE].observations}")

        update_and_optimise_models(models, datasets)
        objective_trajectory = objective_trajectory_sampler.update_trajectory(objective_trajectory)

        for tag, sampler in inequality_constraint_trajectory_samplers.items():
            old_trajectory = inequality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            # print(f"{tag} Updated Trajectory: {updated_trajectory(tf.constant([[0.5, 0.5]], dtype=tf.float64))}")
            # TODO: Does sampler get updated? If so, will need to put new sampler back in dict
            inequality_constraint_trajectories[tag] = updated_trajectory

        for tag, sampler in equality_constraint_trajectory_samplers.items():
            old_trajectory = equality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            # TODO: Does sampler get updated? If so, will need to put new sampler back in dict
            equality_constraint_trajectories[tag] = updated_trajectory
    return opt_x


def create_model(data):
    gpr = gpflow.models.GPR(data.astuple(), kernel=gpflow.kernels.SquaredExponential())
    return GaussianProcessRegression(gpr, num_rff_features=500)


if __name__ == "__main__":
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    # objective_observer = trieste.objectives.utils.mk_observer(objective=objectives.linear_objective, key=OBJECTIVE)
    observer = trieste.objectives.utils.mk_multi_observer(
        OBJECTIVE=objectives.linear_objective,
        INEQUALITY_CONSTRAINT_ONE=constraints.toy_constraint_one,
        INEQUALITY_CONSTRAINT_TWO=constraints.toy_constraint_two)

    initial_inputs = search_space.sample(NUM_INITIAL_SAMPLES)
    initial_data = observer(initial_inputs)

    initial_models = trieste.utils.map_values(create_model, initial_data)

    inequality_lambda = tf.constant([[2.0], [2.0]], dtype=tf.float64)
    initial_penalty = tf.Variable(1.0, dtype=tf.float64)


    augmented_lagrangian_optimiser_unknown_functions(search_space, "INEQUALITY", None, observer, initial_models, initial_data,
                                                     None, inequality_lambda, initial_penalty, 0.001, 100)
