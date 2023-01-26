import tensorflow as tf
import trieste
import gpflow
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.space import Box
from trieste.data import Dataset
from trieste.observer import Observer
from trieste.acquisition.optimizer import generate_continuous_optimizer
from typing import Optional, Mapping
import probabilistic_lagrangians
from functions import constraints
from functions import objectives
from functools import partial

NUM_INITIAL_SAMPLES = 5
INEQUALITY_CONSTRAINT_ONE = "INEQUALITY_CONSTRAINT_ONE"
INEQUALITY_CONSTRAINT_TWO = "INEQUALITY_CONSTRAINT_TWO"


def update_and_optimise_models(objective_model: GaussianProcessRegression,
                               inequality_constraint_models: Optional[Mapping[str, GaussianProcessRegression]],
                               equality_constraint_models: Optional[Mapping[str, GaussianProcessRegression]],
                               objective_dataset: Dataset,
                               inequality_constraint_datasets: Optional[Mapping[str, Dataset]],
                               equality_constraint_datasets: Optional[Mapping[str, Dataset]]):
    # Optimise models based on updated datasets
    objective_model.update(objective_dataset)
    objective_model.optimize(objective_dataset)

    if inequality_constraint_models is not None:
        for tag, model in inequality_constraint_models.items():
            inequality_constraint_dataset = inequality_constraint_datasets[tag]
            model.update(inequality_constraint_dataset)
            model.optimize(inequality_constraint_dataset)

    if equality_constraint_models is not None:
        for tag, model in equality_constraint_models.items():
            equality_constraint_dataset = equality_constraint_datasets[tag]
            model.update(equality_constraint_dataset)
            model.optimize(equality_constraint_dataset)


def augmented_lagrangian_optimiser_unknown_functions(search_space: Box,
                                                     objective_observer: Observer,
                                                     inequality_observer: Optional[Observer],
                                                     equality_observer: Optional[Observer],
                                                     objective_model: GaussianProcessRegression,
                                                     inequality_constraint_models: Optional[Mapping[str, GaussianProcessRegression]],
                                                     equality_constraint_models: Optional[Mapping[str, GaussianProcessRegression]],
                                                     initial_objective_dataset: Dataset,
                                                     initial_inequality_constraint_datasets: Optional[Mapping[str, Dataset]],
                                                     initial_equality_constraint_datasets: Optional[Mapping[str, Dataset]],
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
    best_val = 100
    objective_dataset = initial_objective_dataset
    inequality_constraint_datasets = {}
    equality_constraint_datasets = {}
    if initial_inequality_constraint_datasets is not None:
        inequality_constraint_datasets = initial_inequality_constraint_datasets
    if initial_equality_constraint_datasets is not None:
        equality_constraint_datasets = initial_equality_constraint_datasets

    update_and_optimise_models(objective_model, inequality_constraint_models, equality_constraint_models,
                               objective_dataset, inequality_constraint_datasets, equality_constraint_datasets)
    objective_trajectory_sampler = objective_model.trajectory_sampler()
    objective_trajectory = objective_trajectory_sampler.get_trajectory()
    inequality_constraint_trajectory_samplers = {}
    inequality_constraint_trajectories = {}
    equality_constraint_trajectory_samplers = {}
    equality_constraint_trajectories = {}
    if inequality_constraint_models is not None:
        for tag, model in inequality_constraint_models.items():
            sampler = model.trajectory_sampler()
            trajectory = sampler.get_trajectory()
            inequality_constraint_trajectory_samplers[tag] = sampler
            inequality_constraint_trajectories[tag] = trajectory
    if equality_constraint_models is not None:
        for tag, model in equality_constraint_models.items():
            sampler = model.trajectory_sampler()
            trajectory = sampler.get_trajectory()
            equality_constraint_trajectory_samplers[tag] = sampler
            equality_constraint_trajectories[tag] = trajectory

    for i in range(num_iterations):
        augmented_lagrangian = partial(probabilistic_lagrangians.augmented_lagrangian, objective_trajectory, list(inequality_constraint_trajectories.values()),
                                                                              list(equality_constraint_trajectories.values()), equality_lambd, inequality_lambd,
                                                                              penalty)
        # continuous_optimiser = generate_continuous_optimizer(num_optimization_runs=1, num_initial_samples=1)
        continuous_optimiser = generate_continuous_optimizer()
        opt_x = continuous_optimiser(search_space, augmented_lagrangian)
        print(f"Iteration {i} Opt x: {opt_x}")

        inequality_constraints_satisfied = True
        equality_constraints_satisfied = True

        # TODO: Rename to make clear it uses opt_x?
        objective_values = objective_observer(opt_x)
        inequality_constraint_values = {}
        equality_constraint_values = {}
        if inequality_observer is not None:
            inequality_constraint_values = inequality_observer(opt_x)
            inequality_constraint_tags = [key for key in inequality_constraint_values]
            for j in range(len(inequality_lambd)):
                inequality_constraint_val = inequality_constraint_values[inequality_constraint_tags[j]].observations
                slack_val = probabilistic_lagrangians.obtain_single_slack(inequality_constraint_val, inequality_lambd[j], penalty)
                updated_multiplier = inequality_lambd[j][0] + (1 / penalty) * (inequality_constraint_val + slack_val)
                inequality_lambd = tf.tensor_scatter_nd_update(inequality_lambd, [[j]], updated_multiplier)

            inequality_constraints_opt_x = tf.stack([inequality_constraint_values[tag].observations for tag in inequality_constraint_tags])
            num_inequality_constraints_satisfied = tf.reduce_sum(tf.cast(inequality_constraints_opt_x <= 0, tf.int8))
            if num_inequality_constraints_satisfied != len(inequality_constraint_values):
                inequality_constraints_satisfied = False

        if equality_observer is not None:
            equality_constraint_values = equality_observer(opt_x)
            equality_constraint_tags = [key for key in equality_constraint_values]
            for j in range(len(equality_lambd)):
                equality_constraint_val = equality_constraint_values[equality_constraint_tags[j]].observations
                updated_multiplier = equality_lambd[j][0] + (1 / penalty) * equality_constraint_val
                equality_lambd = tf.tensor_scatter_nd_update(equality_lambd, [[j]], updated_multiplier)

            equality_constraints_opt_x = tf.stack([equality_constraint_values[tag].observations for tag in equality_constraint_tags])
            num_equality_constraints_satisfied = tf.reduce_sum(tf.cast(tf.abs(equality_constraints_opt_x) <= epsilon, tf.int8))
            if num_equality_constraints_satisfied != len(equality_constraint_values):
                equality_constraints_satisfied = False

        if not (equality_constraints_satisfied and inequality_constraints_satisfied):
            print(f"Not Satisfied")
            penalty = penalty / 2
        else:
            print(f"Satisfied")
            if tf.squeeze(objective_values.observations) < best_val:
                best_val = tf.squeeze(objective_values.observations)
                print(f"Best Val: {best_val}")


        # Update models with new data and update sampled trajectories
        objective_dataset = objective_dataset + objective_values
        # inequality_constraint_datasets = inequality_constraint_datasets + inequality_constraint_values
        for tag, dataset in inequality_constraint_datasets.items():
            inequality_constraint_datasets[tag] = dataset + inequality_constraint_values[tag]
        # equality_constraint_datasets = equality_constraint_datasets + equality_constraint_values
        for tag, dataset in equality_constraint_datasets.items():
            equality_constraint_datasets[tag] = dataset + equality_constraint_values[tag]
        update_and_optimise_models(objective_model, inequality_constraint_models, equality_constraint_models, objective_dataset, inequality_constraint_datasets, equality_constraint_datasets)

        objective_trajectory = objective_trajectory_sampler.update_trajectory(objective_trajectory)

        for tag, sampler in inequality_constraint_trajectory_samplers.items():
            # trajectory = sampler.get_trajectory()
            old_trajectory = inequality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            # TODO: Does sampler get updated? If so, will need to put new sampler back in dict
            inequality_constraint_trajectories[tag] = updated_trajectory

        for tag, sampler in equality_constraint_trajectory_samplers.items():
            # trajectory = sampler.get_trajectory()
            old_trajectory = equality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            # TODO: Does sampler get updated? If so, will need to put new sampler back in dict
            equality_constraint_trajectories[tag] = updated_trajectory
    return opt_x


def create_model(data):
    gpr = build_gpr(data, search_space, likelihood_variance=1e-7)
    # gpr = gpflow.models.GPR(data.astuple(), kernel=gpflow.kernels.SquaredExponential())
    return GaussianProcessRegression(gpr, num_rff_features=500)


if __name__ == "__main__":
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    objective_observer = trieste.objectives.utils.mk_observer(objective=objectives.linear_objective)
    inequality_constraint_observer = trieste.objectives.utils.mk_multi_observer(INEQUALITY_CONSTRAINT_ONE=constraints.toy_constraint_one,
                                                                     INEQUALITY_CONSTRAINT_TWO=constraints.toy_constraint_two)

    initial_inputs = search_space.sample(NUM_INITIAL_SAMPLES)
    initial_objective_dataset = objective_observer(initial_inputs)
    initial_inequality_constraint_datasets = inequality_constraint_observer(initial_inputs)

    objective_model = create_model(initial_objective_dataset)
    inequality_constraint_models = trieste.utils.map_values(create_model, initial_inequality_constraint_datasets)

    inequality_lambd = tf.constant([[2.0], [2.0]], dtype=tf.float64)
    initial_penalty = tf.Variable(1.0, dtype=tf.float64)


    augmented_lagrangian_optimiser_unknown_functions(search_space, objective_observer, inequality_constraint_observer, None,
                                                     objective_model, inequality_constraint_models, None, initial_objective_dataset,
                                                     initial_inequality_constraint_datasets, None, None, inequality_lambd, initial_penalty, 0.001, 100)
