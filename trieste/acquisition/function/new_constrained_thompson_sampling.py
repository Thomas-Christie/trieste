from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Optional, cast, Callable

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from matplotlib import ticker
import pickle

from functions import objectives, constraints
from ...data import Dataset
from ...models.interfaces import HasTrajectorySampler
from ...space import SearchSpace
from ...types import Tag, TensorType
from ..interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    ProbabilisticModelType,
    VectorizedAcquisitionFunctionBuilder
)


# Makes use of Lagrange multipliers easier to read, but less vectorised
class ThompsonSamplingAugmentedLagrangian(AcquisitionFunctionBuilder[HasTrajectorySampler]):
    """
    Builder for an augmented Lagrangian acquisition function using Thompson sampling.
    """

    def __init__(
            self,
            objective_tag: Tag,
            inequality_constraint_prefix: Optional[Tag] = None,
            equality_constraint_prefix: Optional[Tag] = None,
            inequality_lambda: Optional[Mapping[Tag, tf.Variable]] = None,
            equality_lambda: Optional[Mapping[Tag, tf.Variable]] = None,
            penalty: tf.Variable = 1,
            epsilon: float = 0.001,
            search_space: Optional[SearchSpace] = None,
            plot: bool = False,
            save_lambda=False,
            save_path: str = None
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param inequality_constraint_prefix: Prefix of tags for inequality constraint data/models/Lagrange multipliers.
        :param equality_constraint_prefix: Prefix for tags for equality constraint data/models/Lagrange multipliers.
        :param inequality_lambda: Initial values for Lagrange multipliers of inequality constraints.
        :param equality_lambda: Initial values for Lagrange multipliers of equality constraints.
        :param penalty: Initial penalty.
        :param epsilon: Bound within which constraints are considered to be satisfied.
        :param search_space: The global search space over which the optimisation is defined. This is
            only used to determine explicit constraints.
        :param plot: Whether to plot the modelled functions at each iteration of BayesOpt.
        :param save_lambda: Whether to keep track of lagrange multipliers at each iteration (for experiments).
        :param save_path: Path to save lagrange multipliers to (method of saving is currently hacky (i.e. need to manually
                          specify num iters of BayesOpt in the code here, but is fine for sake of personal experimentation).
        """
        self._equality_constraint_trajectories = None
        self._equality_constraint_trajectory_samplers = None
        self._inequality_constraint_trajectories = None
        self._inequality_constraint_trajectory_samplers = None
        self._objective_trajectory = None
        self._objective_trajectory_sampler = None
        self._objective_tag = objective_tag
        self._inequality_constraint_prefix = inequality_constraint_prefix
        self._equality_constraint_prefix = equality_constraint_prefix
        self._inequality_lambda = inequality_lambda
        self._equality_lambda = equality_lambda
        self._penalty = penalty
        self._epsilon = epsilon
        self._search_space = search_space
        self._augmented_lagrangian_fn = None
        self._plot = plot
        self._save_lambda = save_lambda
        self._save_path = save_path
        self._iteration = 0
        self._inequality_lambda_tracker = {}  # Store inequality Lagrange multipliers over time
        self._penalty_tracker = [self._penalty]

        for k, v in self._inequality_lambda.items():
            self._inequality_lambda_tracker[k] = [v]

    def __repr__(self) -> str:
        """"""
        return (
            f"ThompsonSamplingAugmentedLagrangian({self._objective_tag!r}, {self._inequality_constraint_prefix!r},"
            f" {self._equality_constraint_prefix!r}, {self._search_space!r})"
        )

    def prepare_acquisition_function(
            self,
            models: Mapping[Tag, HasTrajectorySampler],
            datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param models: The models over each tag.
        :param datasets: The data from the observer.
        :return: The augmented Lagrangian function, using Thompson sampling to approximate all unknown functions.
        :raise KeyError: If `objective_tag` is not found in ``datasets`` and ``models``.
        :raise tf.errors.InvalidArgumentError: If the objective data is empty.
        """
        tf.debugging.Assert(datasets is not None, [tf.constant([])])
        self._iteration += 1

        self._objective_trajectory_sampler = models[self._objective_tag].trajectory_sampler()
        self._objective_trajectory = self._objective_trajectory_sampler.get_trajectory()

        self._inequality_constraint_trajectory_samplers = {}
        self._inequality_constraint_trajectories = {}
        if self._inequality_constraint_prefix is not None:
            for tag, model in models.items():
                if tag.startswith(self._inequality_constraint_prefix):
                    sampler = model.trajectory_sampler()
                    trajectory = sampler.get_trajectory()
                    self._inequality_constraint_trajectory_samplers[tag] = sampler
                    self._inequality_constraint_trajectories[tag] = trajectory

        self._equality_constraint_trajectory_samplers = {}
        self._equality_constraint_trajectories = {}
        if self._equality_constraint_prefix is not None:
            for tag, model in models.items():
                if tag.startswith(self._equality_constraint_prefix):
                    sampler = model.trajectory_sampler()
                    trajectory = sampler.get_trajectory()
                    self._equality_constraint_trajectory_samplers[tag] = sampler
                    self._equality_constraint_trajectories[tag] = trajectory

        self._augmented_lagrangian_fn = self._augmented_lagrangian

        return self._augmented_lagrangian_fn

    def update_acquisition_function(
            self,
            function: AcquisitionFunction,
            models: Mapping[Tag, ProbabilisticModelType],
            datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param models: The models for each tag.
        :param datasets: The data from the observer.
        :return: The augmented Lagrangian function, updating sampled unknown functions (ideally with decoupled sampling).
        """
        tf.debugging.Assert(datasets is not None, [tf.constant([])])
        self._iteration += 1
        datasets = cast(Mapping[Tag, Dataset], datasets)

        objective_dataset = datasets[self._objective_tag]

        tf.debugging.assert_positive(
            len(objective_dataset),
            message="Lagrange multiplier updates are defined with respect to existing points in the"
                    " objective data, but the objective data is empty.",
        )

        # Last point in dataset is most recent estimate of optimal x value
        opt_x = datasets[self._objective_tag].query_points[-1][None, ...]
        tf.debugging.assert_shapes([(opt_x, (1, 2))])

        inequality_constraints_satisfied = True
        equality_constraints_satisfied = True

        opt_objective_x = datasets[self._objective_tag].query_points[-1]
        opt_objective_value = datasets[self._objective_tag].observations[-1]

        # Update Lagrange multipliers for inequality constraints
        if self._inequality_constraint_prefix is not None:
            inequality_constraint_tags = [key for key in datasets.keys() if
                                          key.startswith(self._inequality_constraint_prefix)]
            for tag in inequality_constraint_tags:
                inequality_constraint_val = datasets[tag].observations[-1][None, ...]
                tf.debugging.assert_shapes([(inequality_constraint_val, (1, 1))])
                slack_val = self._obtain_slacks(inequality_constraint_val, self._inequality_lambda[tag], self._penalty)
                updated_multiplier = self._inequality_lambda[tag] + (1 / self._penalty) * (
                            inequality_constraint_val + slack_val)
                self._inequality_lambda[tag] = updated_multiplier
                # If close to zero, we consider it satisfied so penalty doesn't get updated too frequently - otherwise
                # optimiser sometimes fails
                if inequality_constraint_val > self._epsilon:
                    inequality_constraints_satisfied = False

            # inequality_constraints_opt_x = tf.stack([datasets[tag].observations[-1] for tag in inequality_constraint_tags])
            # print(f"Inequality Constraints: {inequality_constraints_opt_x}")

        # Update Lagrange multipliers for equality constraints
        # TODO: Test out code for updating equality constraints
        if self._equality_constraint_prefix is not None:
            equality_constraint_tags = [key for key in datasets.keys() if
                                        key.startswith(self._equality_constraint_prefix)]
            for tag in equality_constraint_tags:
                equality_constraint_val = datasets[tag].observations[-1][None, ...]
                updated_multiplier = self._equality_lambda[tag] + (1 / self._penalty) * equality_constraint_val
                self._equality_lambda[tag] = updated_multiplier
                if tf.abs(equality_constraint_val) > self._epsilon:
                    equality_constraints_satisfied = False

        print(f"Inequality Lambda: {self._inequality_lambda}")
        if not (equality_constraints_satisfied and inequality_constraints_satisfied):
            self._penalty = self._penalty / 2
            print(f"Not Satisfied. Updated Penalty: {self._penalty}")
        else:
            print(f"Satisfied")

        print(f"Objective X: {opt_objective_x} Value: {opt_objective_value}")

        if self._save_lambda:
            # Store inequality lambda values if we're wanting to save lambda values
            for k, v in self._inequality_lambda.items():
                self._inequality_lambda_tracker[k].append(v)
            self._penalty_tracker.append(self._penalty)

            if self._iteration == 40:
                with open(self._save_path + "_inequality_lambda_progression.pkl", "wb") as fp:
                    pickle.dump(self._inequality_lambda_tracker, fp)

                with open(self._save_path + "_penalty_progression.pkl", "wb") as fp:
                    pickle.dump(self._penalty_tracker, fp)

        # Update sampled trajectories
        self._objective_trajectory = self._objective_trajectory_sampler.update_trajectory(self._objective_trajectory)
        for tag, sampler in self._inequality_constraint_trajectory_samplers.items():
            old_trajectory = self._inequality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            self._inequality_constraint_trajectories[tag] = updated_trajectory

        for tag, sampler in self._equality_constraint_trajectory_samplers.items():
            old_trajectory = self._equality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            self._equality_constraint_trajectories[tag] = updated_trajectory

        if self._plot:
            self._plot_models(opt_objective_x)

        self._augmented_lagrangian_fn = self._augmented_lagrangian

        return self._augmented_lagrangian_fn

    def _augmented_lagrangian(
            self,
            x: tf.Tensor) -> tf.Tensor:
        """
        Form augmented Lagrangian from given objective and constraints
        :param x: Array of points at which to evaluate the augmented Lagrangian of shape [N, 1, M] (middle axis is for batching
                  when calling the sampled trajectories)
        :return: Values of augmented Lagrangian at given x values of shape [N, 1]
        """
        objective_vals = self._objective_trajectory(x)
        objective_vals = tf.squeeze(objective_vals, -2)

        # TODO: Test out equality code below
        sum_equality_lambda_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        sum_equality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        if self._equality_constraint_prefix is not None:
            for tag, equality_constraint_trajectory in self._equality_constraint_trajectories.items():
                equality_constraint_vals = equality_constraint_trajectory(x)
                equality_constraint_vals = tf.squeeze(equality_constraint_vals, -2)
                equality_lambda_scaled = self._equality_lambda[tag] * equality_constraint_vals
                equality_penalty_scaled = (1 / (2 * self._penalty)) * tf.square(equality_constraint_vals)
                sum_equality_penalty_scaled += equality_penalty_scaled
                sum_equality_lambda_scaled += equality_lambda_scaled

        # Calculate inequality constraint values using the sampled trajectory, and scale them with Lagrange multipliers
        # and penalty parameter
        sum_inequality_lambda_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        sum_inequality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        if self._inequality_constraint_prefix is not None:
            for tag, inequality_constraint_trajectory in self._inequality_constraint_trajectories.items():
                inequality_constraint_vals = inequality_constraint_trajectory(x)
                inequality_constraint_vals = tf.squeeze(inequality_constraint_vals, -2)
                slack_vals = self._obtain_slacks(inequality_constraint_vals, self._inequality_lambda[tag],
                                                 self._penalty)
                assert (slack_vals.shape == inequality_constraint_vals.shape)
                inequality_plus_slack = inequality_constraint_vals + slack_vals
                inequality_lambda_scaled = self._inequality_lambda[tag] * inequality_plus_slack
                inequality_penalty_scaled = (1 / (2 * self._penalty)) * tf.square(inequality_plus_slack)
                assert (inequality_penalty_scaled.shape == sum_inequality_penalty_scaled.shape)
                assert (inequality_lambda_scaled.shape == sum_inequality_lambda_scaled.shape)
                sum_inequality_penalty_scaled += inequality_penalty_scaled
                sum_inequality_lambda_scaled += inequality_lambda_scaled

        # Return negative of augmented Lagrangian
        return - objective_vals - sum_equality_lambda_scaled - sum_inequality_lambda_scaled - sum_equality_penalty_scaled - sum_inequality_penalty_scaled

    def _obtain_slacks(
            self,
            inequality_constraint_vals: tf.Tensor,
            inequality_lambda: tf.Variable,
            penalty: tf.Variable) -> tf.Tensor:
        """
        Obtain optimal slack values for augmented Lagrangian
        :param inequality_constraint_vals: Inequality constraint values of shape [N, 1]
        :param inequality_lambda: Lagrangian multiplier for given inequality constraint
        :param penalty: Penalty for constraint violation
        :return: Optimal slack values at each x location, of shape [N, 1]
        """
        tf.debugging.assert_rank(inequality_constraint_vals, 2)
        slack_vals = - inequality_lambda * penalty - inequality_constraint_vals
        slack_vals_non_neg = tf.nn.relu(slack_vals)
        tf.debugging.assert_shapes([(slack_vals_non_neg, (..., 1))])
        return slack_vals_non_neg

    def _plot_models(self, prev_query_point):
        """
        Plot visualisation of surrogate models for objective and inequality constraints, as well as the augmented
        Lagrangian.
        """
        x_list = tf.linspace(0, 1, 500)
        y_list = tf.linspace(0, 1, 500)
        xs, ys = tf.meshgrid(x_list, y_list)
        coordinates = tf.expand_dims(tf.stack((tf.reshape(xs, [-1]), tf.reshape(ys, [-1])), axis=1), -2)
        objective_pred = self._objective_trajectory(coordinates)
        lagrangian_pred = - self._augmented_lagrangian(coordinates)
        fig, (ax1, ax2) = plt.subplots(2, len(self._inequality_constraint_trajectories) + 1, figsize=(15, 7))
        objective_plot = ax1[0].contourf(xs, ys, tf.reshape(objective_pred, [y_list.shape[0], x_list.shape[0]]),
                                         levels=500)
        fig.colorbar(objective_plot)
        ax1[0].set_xlabel("OBJECTIVE")
        i = 1
        for tag, trajectory in self._inequality_constraint_trajectories.items():
            inequality_trajectory_pred = trajectory(coordinates)
            inequality_plot = ax1[i].contourf(xs, ys, tf.reshape(inequality_trajectory_pred,
                                                                 [y_list.shape[0], x_list.shape[0]]), levels=500)
            ax1[i].set_xlabel(tag)
            fig.colorbar(inequality_plot)
            i += 1
        lagrangian_plot = ax2[0].contourf(xs, ys, tf.reshape(lagrangian_pred, [y_list.shape[0], x_list.shape[0]]),
                                          levels=np.linspace(-3, 3, 500), extend="both")
        fig.colorbar(lagrangian_plot)
        ax2[0].set_xlabel("AUGMENTED_LAGRANGIAN (CLIPPED)")
        clipped_lagrangian_plot = ax2[1].contourf(xs, ys,
                                                  tf.reshape(lagrangian_pred, [y_list.shape[0], x_list.shape[0]]),
                                                  levels=500)
        fig.colorbar(clipped_lagrangian_plot)
        ax2[1].set_xlabel("AUGMENTED_LAGRANGIAN (UNCLIPPED)")
        ax2[2].text(0.5, 0.5, f"Iteration: {self._iteration} \n Previous Query: {prev_query_point}",
                    horizontalalignment='center', verticalalignment='center')
        ax2[2].axis("off")
        plt.tight_layout()
        plt.show()


class BatchThompsonSamplingAugmentedLagrangian(VectorizedAcquisitionFunctionBuilder[HasTrajectorySampler]):
    """
    Builder for an augmented Lagrangian acquisition function using Thompson sampling.
    """

    def __init__(
        self,
        objective_tag: Tag,
        inequality_constraint_prefix: Optional[Tag] = None,
        equality_constraint_prefix: Optional[Tag] = None,
        inequality_lambda: Optional[Mapping[Tag, TensorType]] = None,
        equality_lambda: Optional[Mapping[Tag, TensorType]] = None,
        batch_size: int = 1,
        penalty: TensorType = None,
        conservative_penalty_decrease: bool = False,
        epsilon: float = 0.01,
        update_lagrange_via_kkt: bool = False,
        search_space: Optional[SearchSpace] = None,
        plot: bool = False,
        save_lambda: bool = False,
        save_path: str = None,
        num_bo_iters: int = None,
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param inequality_constraint_prefix: Prefix of tags for inequality constraint data/models/Lagrange multipliers.
        :param equality_constraint_prefix: Prefix for tags for equality constraint data/models/Lagrange multipliers.
        :param inequality_lambda: Initial values for Lagrange multipliers of inequality constraints.
        :param equality_lambda: Initial values for Lagrange multipliers of equality constraints.
        :param penalty: Initial penalty. If None, it is set to a default value using the _get_initial_penalty method.
        :param conservative_penalty_decrease: Whether to conservatively decrease the penalty parameter if no valid solution
                                              has been found yet.
        :param epsilon: Bound within which equality constraints are considered to be satisfied.
        :param update_lagrange_via_kkt: Whether to update Lagrange multipliers with a gradient-based approach based on
                                        KKT conditions
        :param search_space: The global search space over which the optimisation is defined. This is
            only used to determine explicit constraints.
        :param save_lambda: Whether to keep track of lagrange multipliers at each iteration (for experiments).
        :param save_path: Path to save lagrange multipliers to (method of saving is currently hacky (i.e. need to manually
                          specify num iters of BayesOpt in the code here, but is fine for sake of personal experimentation).
        """
        self._equality_constraint_trajectories = None
        self._equality_constraint_trajectory_samplers = None
        self._inequality_constraint_trajectories = None
        self._inequality_constraint_trajectory_samplers = None
        self._objective_trajectory = None
        self._objective_trajectory_sampler = None
        self._objective_tag = objective_tag
        self._inequality_constraint_prefix = inequality_constraint_prefix
        self._equality_constraint_prefix = equality_constraint_prefix
        if inequality_lambda is None:
            self._inequality_lambda = {}
        else:
            self._inequality_lambda = inequality_lambda  # Values of shape [1, B, 1]
        if equality_lambda is None:
            self._equality_lambda = {}
        else:
            self._equality_lambda = equality_lambda  # Values of shape [1, B, 1]
        self._batch_size = batch_size
        self._penalty = penalty  # [1, B, 1]
        self._epsilon = epsilon
        self._update_lagrange_via_kkt = update_lagrange_via_kkt
        self._search_space = search_space
        self._augmented_lagrangian_fn = None
        self._plot = plot
        self._save_lambda = save_lambda
        self._save_path = save_path
        self._num_bo_iters = num_bo_iters
        self._iteration = 0
        self._inequality_lambda_tracker = {}
        self._equality_lambda_tracker = {}
        self._penalty_tracker = [self._penalty]
        self._best_valid_observation = None  # Kept to None until a valid observation is found.
        self._conservative_penalty_decrease = conservative_penalty_decrease
        if self._inequality_lambda is not None:
            for k, v in self._inequality_lambda.items():
                self._inequality_lambda_tracker[k] = [v]

        if self._equality_lambda is not None:
            for k, v in self._equality_lambda.items():
                self._equality_lambda_tracker[k] = [v]

    def __repr__(self) -> str:
        """"""
        return (
            f"ThompsonSamplingAugmentedLagrangian({self._objective_tag!r}, {self._inequality_constraint_prefix!r},"
            f" {self._equality_constraint_prefix!r}, {self._search_space!r})"
        )

    def prepare_acquisition_function(
        self,
        models: Mapping[Tag, HasTrajectorySampler],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param models: The models over each tag.
        :param datasets: The data from the observer.
        :return: The augmented Lagrangian function, using Thompson sampling to approximate all unknown functions.
        :raise KeyError: If `objective_tag` is not found in ``datasets`` and ``models``.
        :raise tf.errors.InvalidArgumentError: If the objective data is empty.
        """
        tf.debugging.Assert(datasets is not None, [tf.constant([])])
        self._iteration += 1

        # Find the best valid objective value seen so far
        satisfied_mask = tf.constant(value=True, shape=datasets[self._objective_tag].observations.shape)
        for tag, dataset in datasets.items():
            if (self._inequality_constraint_prefix is not None) and (tag.startswith(self._inequality_constraint_prefix)):
                constraint_vals = dataset.observations
                valid_constraint_vals = constraint_vals <= 0
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)
            elif (self._equality_constraint_prefix is not None) and (tag.startswith(self._equality_constraint_prefix)):
                constraint_vals = dataset.observations
                valid_constraint_vals = tf.abs(constraint_vals) <= self._epsilon
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)

        if tf.reduce_sum(tf.cast(satisfied_mask, tf.int32)) != 0:
            objective_vals = datasets[self._objective_tag].observations
            valid_y = tf.boolean_mask(objective_vals, satisfied_mask)
            self._best_valid_observation = tf.math.reduce_min(valid_y)
            print(f"Best Valid Observation: {self._best_valid_observation}")

        if self._penalty is None:
            initial_penalty = self._get_initial_penalty(datasets)
            self._penalty = initial_penalty * tf.ones((1, self._batch_size, 1), dtype=tf.float64)

        self._objective_trajectory_sampler = models[self._objective_tag].trajectory_sampler()
        self._objective_trajectory = self._objective_trajectory_sampler.get_trajectory()

        self._inequality_constraint_trajectory_samplers = {}
        self._inequality_constraint_trajectories = {}
        if self._inequality_constraint_prefix is not None:
            for tag, model in models.items():
                if tag.startswith(self._inequality_constraint_prefix):
                    sampler = model.trajectory_sampler()
                    trajectory = sampler.get_trajectory()
                    self._inequality_constraint_trajectory_samplers[tag] = sampler
                    self._inequality_constraint_trajectories[tag] = trajectory

        self._equality_constraint_trajectory_samplers = {}
        self._equality_constraint_trajectories = {}
        if self._equality_constraint_prefix is not None:
            for tag, model in models.items():
                if tag.startswith(self._equality_constraint_prefix):
                    sampler = model.trajectory_sampler()
                    trajectory = sampler.get_trajectory()
                    self._equality_constraint_trajectory_samplers[tag] = sampler
                    self._equality_constraint_trajectories[tag] = trajectory

        self._augmented_lagrangian_fn = self._augmented_lagrangian

        return self._augmented_lagrangian_fn

    def _get_initial_penalty(self,
                             datasets: Optional[Mapping[Tag, Dataset]] = None):
        """
        :param datasets: The data from the observer.
        :return: Initial penalty to use in Augmented Lagrangian, calculated as done by https://www.mcs.anl.gov/papers/P5401-0915.pdf
                 and https://arxiv.org/pdf/1605.09466.pdf
        """
        all_satisfied = None
        sum_squared = None
        if self._inequality_constraint_prefix is not None:
            inequality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._inequality_constraint_prefix)]
            for tag in inequality_constraint_tags:
                constraint_satisfied = tf.squeeze(datasets[tag].observations) <= 0
                if all_satisfied is None:
                    all_satisfied = constraint_satisfied
                else:
                    all_satisfied = tf.logical_and(all_satisfied, constraint_satisfied)

                # Only consider invalid inequality constraints (i.e. > 0)
                constraint_squared = tf.square(tf.nn.relu(tf.squeeze(datasets[tag].observations)))
                if sum_squared is None:
                    sum_squared = constraint_squared
                else:
                    sum_squared += constraint_squared

        if self._equality_constraint_prefix is not None:
            equality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._equality_constraint_prefix)]
            for tag in equality_constraint_tags:
                constraint_satisfied = tf.abs(tf.squeeze(datasets[tag].observations)) <= self._epsilon
                if all_satisfied is None:
                    all_satisfied = constraint_satisfied
                else:
                    all_satisfied = tf.logical_and(all_satisfied, constraint_satisfied)

                constraint_squared = tf.square(tf.squeeze(datasets[tag].observations))
                if sum_squared is None:
                    sum_squared = constraint_squared
                else:
                    sum_squared += constraint_squared

        at_least_one_violated = tf.logical_not(all_satisfied)

        if tf.reduce_sum(tf.cast(at_least_one_violated, tf.int32)) == 0:
            # If valid everywhere, set initial penalty to one
            initial_penalty = 1
        else:
            sum_squared = sum_squared[at_least_one_violated]
            min_sum_squared = tf.math.reduce_min(sum_squared)
            if tf.reduce_sum(tf.cast(all_satisfied, tf.int32)) == 0:
                denominator = 2 * tfp.stats.percentile(datasets[self._objective_tag].observations, 50.0, interpolation='midpoint')
            else:
                best_valid_objective = tf.math.reduce_min(datasets[self._objective_tag].observations[all_satisfied])
                denominator = 2 * best_valid_objective
            initial_penalty = min_sum_squared / denominator

        return tf.cast(tf.abs(initial_penalty), dtype=tf.float64)

    def update_acquisition_function(
            self,
            function: AcquisitionFunction,
            models: Mapping[Tag, ProbabilisticModelType],
            datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param models: The models for each tag.
        :param datasets: The data from the observer.
        :return: The augmented Lagrangian function, updating sampled unknown functions (ideally with decoupled sampling).
        """
        tf.debugging.Assert(datasets is not None, [tf.constant([])])
        self._iteration += 1
        print(f"Iteration: {self._iteration}")
        datasets = cast(Mapping[Tag, Dataset], datasets)

        objective_dataset = datasets[self._objective_tag]

        tf.debugging.assert_positive(
            len(objective_dataset),
            message="Lagrange multiplier updates are defined with respect to existing points in the"
                    " objective data, but the objective data is empty.",
        )

        # Find the best valid objective value seen so far
        satisfied_mask = tf.constant(value=True, shape=datasets[self._objective_tag].observations.shape)
        for tag, dataset in datasets.items():
            if (self._inequality_constraint_prefix is not None) and (tag.startswith(self._inequality_constraint_prefix)):
                constraint_vals = dataset.observations
                valid_constraint_vals = constraint_vals <= 0
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)
            elif (self._equality_constraint_prefix is not None) and (tag.startswith(self._equality_constraint_prefix)):
                constraint_vals = dataset.observations
                valid_constraint_vals = tf.abs(constraint_vals) <= self._epsilon
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)

        if tf.reduce_sum(tf.cast(satisfied_mask, tf.int32)) != 0:
            objective_vals = datasets[self._objective_tag].observations
            valid_y = tf.boolean_mask(objective_vals, satisfied_mask)
            self._best_valid_observation = tf.math.reduce_min(valid_y)
            print(f"Best Valid Observation: {self._best_valid_observation}")

        # Last "batch_size" points in dataset are most recent estimates of optimal x value
        opt_x = datasets[self._objective_tag].query_points[-self._batch_size:][None, ...]
        tf.debugging.assert_shapes([(opt_x, (1, None, None))])

        # Update sampled trajectories
        self._objective_trajectory = self._objective_trajectory_sampler.update_trajectory(self._objective_trajectory)
        for tag, sampler in self._inequality_constraint_trajectory_samplers.items():
            old_trajectory = self._inequality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            self._inequality_constraint_trajectories[tag] = updated_trajectory

        for tag, sampler in self._equality_constraint_trajectory_samplers.items():
            old_trajectory = self._equality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            self._equality_constraint_trajectories[tag] = updated_trajectory

        # Update Lagrange multipliers
        if self._update_lagrange_via_kkt:
            self._kkt_lagrange_update(datasets)
        else:
            self._default_lagrange_update(datasets)

        # Update penalty
        self._update_penalty(datasets)

        opt_objective_x = datasets[self._objective_tag].query_points[-self._batch_size:]
        opt_objective_value = datasets[self._objective_tag].observations[-self._batch_size:]
        print(f"Objective X: {opt_objective_x} Value: {opt_objective_value}")

        if self._save_lambda:
            # Store inequality lambda values if we're wanting to save lambda values
            if self._inequality_lambda is not None:
                for k, v in self._inequality_lambda.items():
                    self._inequality_lambda_tracker[k].append(v)

            # Store equality lambda values if we're wanting to save lambda values
            if self._equality_lambda is not None:
                for k, v in self._equality_lambda.items():
                    self._equality_lambda_tracker[k].append(v)

            self._penalty_tracker.append(self._penalty)

            if self._iteration == self._num_bo_iters:
                if self._inequality_lambda is not None:
                    with open(self._save_path + "_inequality_lambda_progression.pkl", "wb") as fp:
                        pickle.dump(self._inequality_lambda_tracker, fp)

                if self._equality_lambda is not None:
                    with open(self._save_path + "_equality_lambda_progression.pkl", "wb") as fp:
                        pickle.dump(self._equality_lambda_tracker, fp)

                with open(self._save_path + "_penalty_progression.pkl", "wb") as fp:
                    pickle.dump(self._penalty_tracker, fp)

        if self._plot:
            self._plot_models(opt_objective_x)

        self._augmented_lagrangian_fn = self._augmented_lagrangian

        return self._augmented_lagrangian_fn

    def _kkt_lagrange_update(self,
                             datasets: Mapping[Tag, Dataset]):
        # Obtain most recent query points (i.e. best points in last BO round)
        batch_query_points = datasets[self._objective_tag].query_points[-self._batch_size:][None, ...]

        # Get gradients of objective trajectories
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(batch_query_points)
            objective_predictions = self._objective_trajectory(batch_query_points)
        objective_grads = tf.squeeze(tape.gradient(objective_predictions, batch_query_points), axis=0)  # [B, D]

        # Get gradients of inequality constraints
        inequality_constraint_grad_dict = {}
        inequality_constraint_bind_dict = {}
        for tag, trajectory in self._inequality_constraint_trajectories.items():
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(batch_query_points)
                constraint_predictions = trajectory(batch_query_points)
            binding = tf.squeeze(tf.abs(constraint_predictions) <= self._epsilon, axis=0)  # If constraint near 0, consider to be binding, shape: [B, 1]
            constraint_grads = tf.squeeze(tape.gradient(constraint_predictions, batch_query_points), axis=0)  # [B, D]
            inequality_constraint_grad_dict[tag] = constraint_grads
            inequality_constraint_bind_dict[tag] = binding

        # Get gradients of equality constraints
        equality_constraint_grad_dict = {}
        equality_constraint_bind_dict = {}
        for tag, trajectory in self._equality_constraint_trajectories.items():
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(batch_query_points)
                constraint_predictions = trajectory(batch_query_points)
            binding = tf.squeeze(tf.abs(constraint_predictions) <= self._epsilon, axis=0)  # If constraint near 0, consider to be binding, shape: [B, 1]
            constraint_grads = tf.squeeze(tape.gradient(constraint_predictions, batch_query_points), axis=0)  # [B, D]
            equality_constraint_grad_dict[tag] = constraint_grads
            equality_constraint_bind_dict[tag] = binding

        # Dictionaries mapping tag -> list of updated Lagrange multipliers (of length batch_size)
        updated_inequality_lagrange_multipliers = {}
        updated_equality_lagrange_multipliers = {}
        # Calculate updated Lagrange multipliers for each element in the batch separately
        for batch in range(self._batch_size):
            objective_grad = objective_grads[batch][..., None]

            binding_inequality_constraints = OrderedDict()
            for tag, gradient in inequality_constraint_grad_dict.items():
                binding = inequality_constraint_bind_dict[tag][batch]
                # Only consider binding constraints
                if binding:
                    binding_inequality_constraints[tag] = inequality_constraint_grad_dict[tag][batch]

            binding_equality_constraints = OrderedDict()
            for tag, gradient in equality_constraint_grad_dict.items():
                binding = equality_constraint_bind_dict[tag][batch]
                # Only consider binding constraints
                if binding:
                    binding_equality_constraints[tag] = equality_constraint_grad_dict[tag][batch]

            # Construct gradient matrix
            gradient_matrix = None

            for tag, gradient in binding_inequality_constraints.items():
                if gradient_matrix is None:
                    gradient_matrix = gradient[..., None]
                else:
                    gradient_matrix = tf.concat((gradient_matrix, gradient[..., None]), axis=1)

            for tag, gradient in binding_equality_constraints.items():
                if gradient_matrix is None:
                    gradient_matrix = gradient[..., None]
                else:
                    gradient_matrix = tf.concat((gradient_matrix, gradient[..., None]), axis=1)

            if gradient_matrix is None:
                # No constraints are considered binding, so all Lagrange multipliers are set to 0
                for tag in self._inequality_lambda.keys():
                    if tag in updated_inequality_lagrange_multipliers.keys():
                        updated_inequality_lagrange_multipliers[tag].append(0)
                    else:
                        updated_inequality_lagrange_multipliers[tag] = [0]

                for tag in self._equality_lambda.keys():
                    if tag in updated_equality_lagrange_multipliers.keys():
                        updated_equality_lagrange_multipliers[tag].append(0)
                    else:
                        updated_equality_lagrange_multipliers[tag] = [0]
            else:
                num_binding_inequality_constraints = len(binding_inequality_constraints)
                num_binding_equality_constraints = len(binding_equality_constraints)
                # TODO: Need to consider case when least-squares problem doesn't have a unique solution
                lagrange_multipliers = tf.linalg.inv(tf.matmul(gradient_matrix, gradient_matrix, transpose_a=True)) @ tf.transpose(gradient_matrix) @ (-objective_grad)

                if num_binding_inequality_constraints > 0:
                    # Enforce non-negativity of inequality constraint Lagrange multipliers
                    inequality_lagrange_multipliers = tf.nn.relu(lagrange_multipliers[:num_binding_inequality_constraints])
                    binding_inequality_constraint_tags = list(binding_inequality_constraints.keys())
                    for i in range(len(binding_inequality_constraint_tags)):
                        tag = binding_inequality_constraint_tags[i]
                        if tag in updated_inequality_lagrange_multipliers.keys():
                            updated_inequality_lagrange_multipliers[tag].append(tf.squeeze(inequality_lagrange_multipliers[i]))
                        else:
                            updated_inequality_lagrange_multipliers[tag] = [tf.squeeze(inequality_lagrange_multipliers[i])]

                    # Give non-binding constraints Lagrange multipliers of 0
                    for tag in self._inequality_lambda.keys():
                        if tag not in binding_inequality_constraint_tags:
                            if tag in updated_inequality_lagrange_multipliers.keys():
                                updated_inequality_lagrange_multipliers[tag].append(0)
                            else:
                                updated_inequality_lagrange_multipliers[tag] = [0]

                if num_binding_equality_constraints > 0:
                    equality_lagrange_multipliers = lagrange_multipliers[num_binding_inequality_constraints:]
                    binding_equality_constraint_tags = list(binding_equality_constraints.keys())
                    for i in range(len(binding_equality_constraint_tags)):
                        tag = binding_equality_constraint_tags[i]
                        if tag in updated_equality_lagrange_multipliers.keys():
                            updated_equality_lagrange_multipliers[tag].append(tf.squeeze(equality_lagrange_multipliers[i]))
                        else:
                            updated_equality_lagrange_multipliers[tag] = [tf.squeeze(equality_lagrange_multipliers[i])]

                    # Give non-binding constraints Lagrange multipliers of 0
                    for tag in self._equality_lambda.keys():
                        if tag not in binding_equality_constraint_tags:
                            if tag in updated_equality_lagrange_multipliers.keys():
                                updated_equality_lagrange_multipliers[tag].append(0)
                            else:
                                updated_equality_lagrange_multipliers[tag] = [0]

        for tag, updated_lagrange_multiplier in updated_inequality_lagrange_multipliers.items():
            self._inequality_lambda[tag] = tf.Variable(updated_lagrange_multiplier, dtype=tf.float64)[None, ..., None]

        for tag, updated_lagrange_multiplier in updated_equality_lagrange_multipliers.items():
            self._equality_lambda[tag] = tf.Variable(updated_lagrange_multiplier, dtype=tf.float64)[None, ..., None]

    def _default_lagrange_update(self,
                                 datasets: Mapping[Tag, Dataset]):
        # Update Lagrange multipliers for inequality constraints
        if self._inequality_constraint_prefix is not None:
            inequality_constraint_tags = [key for key in datasets.keys() if
                                          key.startswith(self._inequality_constraint_prefix)]
            for tag in inequality_constraint_tags:
                inequality_constraint_val = datasets[tag].observations[-self._batch_size:][None, ...]  # [1, B, 1]
                tf.debugging.assert_shapes([(inequality_constraint_val, (1, None, 1))])
                slack_val = self._obtain_slacks(inequality_constraint_val, self._inequality_lambda[tag],
                                                self._penalty)  # [1, B, 1]
                assert (slack_val.shape == self._inequality_lambda[tag].shape)
                assert (slack_val.shape == inequality_constraint_val.shape)
                updated_multiplier = self._inequality_lambda[tag] + (1 / self._penalty) * (
                            inequality_constraint_val + slack_val)
                self._inequality_lambda[tag] = updated_multiplier


        # Update Lagrange multipliers for equality constraints
        if self._equality_constraint_prefix is not None:
            equality_constraint_tags = [key for key in datasets.keys() if
                                        key.startswith(self._equality_constraint_prefix)]
            for tag in equality_constraint_tags:
                equality_constraint_val = datasets[tag].observations[-self._batch_size:][None, ...]  # [1, B, 1]
                updated_multiplier = self._equality_lambda[tag] + (1 / self._penalty) * equality_constraint_val
                self._equality_lambda[tag] = updated_multiplier

    def _update_penalty(self,
                        datasets: Mapping[Tag, Dataset]):
        inequality_constraints_satisfied = True
        equality_constraints_satisfied = True

        # See if inequality constraints are satisfied
        batch_inequality_constraints_violated = tf.constant(False, shape=(1, self._batch_size, 1))
        if self._inequality_constraint_prefix is not None:
            inequality_constraint_tags = [key for key in datasets.keys() if
                                          key.startswith(self._inequality_constraint_prefix)]
            for tag in inequality_constraint_tags:
                inequality_constraint_val = datasets[tag].observations[-self._batch_size:][None, ...]  # [1, B, 1]
                tf.debugging.assert_shapes([(inequality_constraint_val, (1, None, 1))])

                batch_inequality_constraints_violated = tf.logical_or(batch_inequality_constraints_violated,
                                                                      inequality_constraint_val > 0)  # See if any of the constraints are violated
                if tf.reduce_sum(tf.cast(batch_inequality_constraints_violated, tf.int8)) != 0:
                    inequality_constraints_satisfied = False

        # See if equality constraints are satisfied
        batch_equality_constraints_violated = tf.constant(False, shape=(1, self._batch_size, 1))
        if self._equality_constraint_prefix is not None:
            equality_constraint_tags = [key for key in datasets.keys() if
                                        key.startswith(self._equality_constraint_prefix)]
            for tag in equality_constraint_tags:
                equality_constraint_val = datasets[tag].observations[-self._batch_size:][None, ...]  # [1, B, 1]
                batch_equality_constraints_violated = tf.logical_or(batch_equality_constraints_violated,
                                                                    tf.abs(equality_constraint_val) > self._epsilon)

                if tf.reduce_sum(tf.cast(batch_equality_constraints_violated, tf.int8)) != 0:
                    equality_constraints_satisfied = False

        assert (batch_inequality_constraints_violated.shape == batch_equality_constraints_violated.shape)
        batch_constraints_violated = tf.logical_or(batch_inequality_constraints_violated,
                                                   batch_equality_constraints_violated)
        sum_batch_constraints_violated = tf.reduce_sum(tf.cast(batch_constraints_violated, tf.float64))
        print(f"Num Violated: {sum_batch_constraints_violated}")
        if not (equality_constraints_satisfied and inequality_constraints_satisfied):
            if self._conservative_penalty_decrease and self._best_valid_observation is None:
                decreasing_constant = 10/9
            else:
                decreasing_constant = 2.0
            print(f"Decreasing constant: {decreasing_constant}")
            self._penalty = tf.where(batch_constraints_violated,
                                     self._penalty / tf.pow(tf.constant(decreasing_constant, dtype=tf.float64),
                                                            sum_batch_constraints_violated),
                                     self._penalty)
            print(f"Not Satisfied. Updated Penalty: {self._penalty}")
        else:
            print(f"Satisfied")

    def _augmented_lagrangian(
            self,
            x: tf.Tensor) -> tf.Tensor:
        """
        Form augmented Lagrangian
        :param x: Array of points at which to evaluate the augmented Lagrangian of shape [N, B, D] (middle axis is for batching
                  when calling the sampled trajectories)
        :return: Values of augmented Lagrangian at given x values of shape [N, B]
        """
        objective_vals = self._objective_trajectory(x)

        sum_equality_lambda_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        sum_equality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        if self._equality_constraint_prefix is not None:
            for tag, equality_constraint_trajectory in self._equality_constraint_trajectories.items():
                equality_constraint_vals = equality_constraint_trajectory(x)  # [N, B, 1]
                equality_lambda_scaled = self._equality_lambda[tag] * equality_constraint_vals  # [N, B, 1]
                equality_penalty_scaled = (1 / (2 * self._penalty)) * tf.square(equality_constraint_vals)
                sum_equality_penalty_scaled += equality_penalty_scaled
                sum_equality_lambda_scaled += equality_lambda_scaled

        # Calculate inequality constraint values using the sampled trajectory, and scale them with Lagrange multipliers
        # and penalty parameter
        sum_inequality_lambda_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        sum_inequality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        if self._inequality_constraint_prefix is not None:
            for tag, inequality_constraint_trajectory in self._inequality_constraint_trajectories.items():
                inequality_constraint_vals = inequality_constraint_trajectory(x)  # [N, B, 1]
                slack_vals = self._obtain_slacks(inequality_constraint_vals, self._inequality_lambda[tag],
                                                 self._penalty)  # [N, B, 1]
                assert (slack_vals.shape == inequality_constraint_vals.shape)
                inequality_plus_slack = inequality_constraint_vals + slack_vals
                inequality_lambda_scaled = self._inequality_lambda[tag] * inequality_plus_slack  # [N, B, 1] - This could lose first axis if N = 1
                inequality_penalty_scaled = (1 / (2 * self._penalty)) * tf.square(inequality_plus_slack)
                assert (inequality_penalty_scaled.shape == sum_inequality_penalty_scaled.shape)
                assert (inequality_lambda_scaled.shape == sum_inequality_lambda_scaled.shape)
                sum_inequality_penalty_scaled += inequality_penalty_scaled
                sum_inequality_lambda_scaled += inequality_lambda_scaled

        # Return negative of augmented Lagrangian
        neg_al = tf.squeeze(- objective_vals - sum_equality_lambda_scaled - sum_inequality_lambda_scaled - sum_equality_penalty_scaled - sum_inequality_penalty_scaled, -1)
        return neg_al

    def _obtain_slacks(
            self,
            inequality_constraint_vals: tf.Tensor,
            inequality_lambda: tf.Variable,
            penalty: tf.Variable) -> tf.Tensor:
        """
        Obtain optimal slack values for augmented Lagrangian
        :param inequality_constraint_vals: Inequality constraint values of shape [N, B, 1]
        :param inequality_lambda: Batch of Lagrangian multipliers for given inequality constraint, of shape [B, 1]
        :param penalty: Penalty for constraint violation
        :return: Optimal slack values at each x location, of shape [N, B, 1]
        """
        tf.debugging.assert_rank(inequality_constraint_vals, 3)
        slack_vals = - (inequality_lambda * penalty) - inequality_constraint_vals  # TODO: Perhaps make broadcasting explicit?
        slack_vals_non_neg = tf.nn.relu(slack_vals)
        tf.debugging.assert_shapes([(slack_vals_non_neg, (..., 1))])
        return slack_vals_non_neg

    def _plot_models(self, prev_query_point):
        """
        Plot visualisation of surrogate models for objective and inequality constraints, as well as the augmented
        Lagrangian.
        """
        x_list = tf.linspace(0, 1, 500)
        y_list = tf.linspace(0, 1, 500)
        xs, ys = tf.meshgrid(x_list, y_list)
        coordinates = tf.expand_dims(tf.stack((tf.reshape(xs, [-1]), tf.reshape(ys, [-1])), axis=1), -2)
        objective_pred = self._objective_trajectory(coordinates)
        lagrangian_pred = - self._augmented_lagrangian(coordinates)
        num_cols = max(len(self._equality_constraint_trajectories), len(self._inequality_constraint_trajectories), 3)
        fig, (ax1, ax2, ax3) = plt.subplots(3, num_cols, figsize=(15, 7))
        objective_plot = ax1[0].contourf(xs, ys, tf.reshape(objective_pred, [y_list.shape[0], x_list.shape[0]]), levels=500)
        fig.colorbar(objective_plot)
        ax1[0].set_xlabel("OBJECTIVE")

        i = 0
        for tag, trajectory in self._inequality_constraint_trajectories.items():
            inequality_trajectory_pred = trajectory(coordinates)
            inequality_plot = ax2[i].contourf(xs, ys, tf.reshape(inequality_trajectory_pred, [y_list.shape[0], x_list.shape[0]]), levels=500)
            ax2[i].set_xlabel(tag)
            fig.colorbar(inequality_plot)
            i += 1

        i = 0
        for tag, trajectory in self._equality_constraint_trajectories.items():
            equality_trajectory_pred = trajectory(coordinates)
            equality_plot = ax3[i].contourf(xs, ys, tf.reshape(equality_trajectory_pred,[y_list.shape[0], x_list.shape[0]]), levels=500)
            ax3[i].set_xlabel(tag)
            fig.colorbar(equality_plot)
            i += 1

        clipped_lagrangian_plot = ax1[1].contourf(xs, ys, tf.reshape(lagrangian_pred, [y_list.shape[0], x_list.shape[0]]), locator=ticker.SymmetricalLogLocator(base=10, linthresh=1))
        ax1[1].set_yscale('symlog')
        fig.colorbar(clipped_lagrangian_plot)
        ax1[1].set_xlabel("AUGMENTED_LAGRANGIAN (SYMLOG SCALE)")
        lagrangian_plot = ax1[2].contourf(xs, ys, tf.reshape(lagrangian_pred, [y_list.shape[0], x_list.shape[0]]), levels=500)
        fig.colorbar(lagrangian_plot)
        ax1[2].set_xlabel("AUGMENTED_LAGRANGIAN (UNCLIPPED)")
        ax2[1].text(0.5, 0.5, f"Iteration: {self._iteration} \n Previous Query: {prev_query_point}",
                    horizontalalignment='center', verticalalignment='center')
        ax2[1].axis("off")
        ax2[2].axis("off")
        ax2[2].axis("off")
        ax3[2].axis("off")
        plt.tight_layout()
        # with open(f"../results/22-02-23/visualisation/iter_{self._iteration}.png", "wb") as fp:
        #     plt.savefig(fp)
        plt.show()


# NOTE - Below updates penalties and Lagrange multipliers as in the appendix of the original paper (which slightly
# differs from the main paper). The performance difference is negligible.
class FullyConsistentBatchThompsonSamplingAugmentedLagrangian(VectorizedAcquisitionFunctionBuilder[HasTrajectorySampler]):
    """
    Builder for an augmented Lagrangian acquisition function using Thompson sampling.
    """

    def __init__(
        self,
        objective_tag: Tag,
        known_objective: Callable = None,
        inequality_constraint_prefix: Optional[Tag] = None,
        equality_constraint_prefix: Optional[Tag] = None,
        inequality_lambda: Optional[Mapping[Tag, TensorType]] = None,
        equality_lambda: Optional[Mapping[Tag, TensorType]] = None,
        batch_size: int = 1,
        penalty: TensorType = None,
        conservative_penalty_decrease: bool = False,
        epsilon: float = 0.01,
        update_lagrange_via_kkt: bool = False,
        search_space: Optional[SearchSpace] = None,
        plot: bool = False,
        save_lambda: bool = False,
        save_path: str = None,
        num_bo_iters: int = None,
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param known_objective: Known objective function (if it is being treated as known) which behaves like a trajectory.
        :param inequality_constraint_prefix: Prefix of tags for inequality constraint data/models/Lagrange multipliers.
        :param equality_constraint_prefix: Prefix for tags for equality constraint data/models/Lagrange multipliers.
        :param inequality_lambda: Initial values for Lagrange multipliers of inequality constraints.
        :param equality_lambda: Initial values for Lagrange multipliers of equality constraints.
        :param penalty: Initial penalty. If None, it is set to a default value using the _get_initial_penalty method.
        :param conservative_penalty_decrease: Whether to conservatively decrease the penalty parameter if no valid solution
                                              has been found yet.
        :param epsilon: Bound within which equality constraints are considered to be satisfied.
        :param update_lagrange_via_kkt: Whether to update Lagrange multipliers with a gradient-based approach based on
                                        KKT conditions
        :param search_space: The global search space over which the optimisation is defined. This is
            only used to determine explicit constraints.
        :param save_lambda: Whether to keep track of lagrange multipliers at each iteration (for experiments).
        :param save_path: Path to save lagrange multipliers to (method of saving is currently hacky (i.e. need to manually
                          specify num iters of BayesOpt in the code here, but is fine for sake of personal experimentation).
        """
        self._equality_constraint_trajectories = None
        self._equality_constraint_trajectory_samplers = None
        self._inequality_constraint_trajectories = None
        self._inequality_constraint_trajectory_samplers = None
        self._objective_trajectory = None
        self._objective_trajectory_sampler = None
        self._objective_tag = objective_tag
        self._known_objective = known_objective
        self._inequality_constraint_prefix = inequality_constraint_prefix
        self._equality_constraint_prefix = equality_constraint_prefix
        if inequality_lambda is None:
            self._inequality_lambda = {}
        else:
            self._inequality_lambda = inequality_lambda
        if equality_lambda is None:
            self._equality_lambda = {}
        else:
            self._equality_lambda = equality_lambda
        self._batch_size = batch_size
        self._penalty = penalty
        self._epsilon = epsilon
        self._update_lagrange_via_kkt = update_lagrange_via_kkt
        self._search_space = search_space
        self._augmented_lagrangian_fn = None
        self._plot = plot
        self._save_lambda = save_lambda
        self._save_path = save_path
        self._num_bo_iters = num_bo_iters
        self._iteration = 0
        self._inequality_lambda_tracker = {}
        self._equality_lambda_tracker = {}
        self._penalty_tracker = [self._penalty]
        self._best_valid_observation = None  # Kept to None until a valid observation is found.
        self._conservative_penalty_decrease = conservative_penalty_decrease
        if self._inequality_lambda is not None:
            for k, v in self._inequality_lambda.items():
                self._inequality_lambda_tracker[k] = [v]

        if self._equality_lambda is not None:
            for k, v in self._equality_lambda.items():
                self._equality_lambda_tracker[k] = [v]

    def __repr__(self) -> str:
        """"""
        return (
            f"ThompsonSamplingAugmentedLagrangian({self._objective_tag!r}, {self._inequality_constraint_prefix!r},"
            f" {self._equality_constraint_prefix!r}, {self._search_space!r})"
        )

    def prepare_acquisition_function(
        self,
        models: Mapping[Tag, HasTrajectorySampler],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param models: The models over each tag.
        :param datasets: The data from the observer.
        :return: The augmented Lagrangian function, using Thompson sampling to approximate all unknown functions.
        :raise KeyError: If `objective_tag` is not found in ``datasets`` and ``models``.
        :raise tf.errors.InvalidArgumentError: If the objective data is empty.
        """
        tf.debugging.Assert(datasets is not None, [tf.constant([])])
        self._iteration += 1

        # Find the best valid objective value seen so far
        satisfied_mask = tf.constant(value=True, shape=datasets[self._objective_tag].observations.shape)
        for tag, dataset in datasets.items():
            if (self._inequality_constraint_prefix is not None) and (tag.startswith(self._inequality_constraint_prefix)):
                constraint_vals = dataset.observations
                valid_constraint_vals = constraint_vals <= 0
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)
            elif (self._equality_constraint_prefix is not None) and (tag.startswith(self._equality_constraint_prefix)):
                constraint_vals = dataset.observations
                valid_constraint_vals = tf.abs(constraint_vals) <= self._epsilon
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)

        if tf.reduce_sum(tf.cast(satisfied_mask, tf.int32)) != 0:
            objective_vals = datasets[self._objective_tag].observations
            valid_y = tf.boolean_mask(objective_vals, satisfied_mask)
            self._best_valid_observation = tf.math.reduce_min(valid_y)
            print(f"Best Valid Observation: {self._best_valid_observation}")

        if self._penalty is None:
            initial_penalty = self._get_initial_penalty(datasets)
            print(f"Initial Penalty: {initial_penalty}")
            self._penalty = tf.constant(initial_penalty, dtype=tf.float64)

        if self._known_objective is not None:
            # If objective is known, we don't model it and instead use the ground truth.
            self._objective_trajectory = self._known_objective
        else:
            self._objective_trajectory_sampler = models[self._objective_tag].trajectory_sampler()
            self._objective_trajectory = self._objective_trajectory_sampler.get_trajectory()

        self._inequality_constraint_trajectory_samplers = {}
        self._inequality_constraint_trajectories = {}
        if self._inequality_constraint_prefix is not None:
            for tag, model in models.items():
                if tag.startswith(self._inequality_constraint_prefix):
                    sampler = model.trajectory_sampler()
                    trajectory = sampler.get_trajectory()
                    self._inequality_constraint_trajectory_samplers[tag] = sampler
                    self._inequality_constraint_trajectories[tag] = trajectory

        self._equality_constraint_trajectory_samplers = {}
        self._equality_constraint_trajectories = {}
        if self._equality_constraint_prefix is not None:
            for tag, model in models.items():
                if tag.startswith(self._equality_constraint_prefix):
                    sampler = model.trajectory_sampler()
                    trajectory = sampler.get_trajectory()
                    self._equality_constraint_trajectory_samplers[tag] = sampler
                    self._equality_constraint_trajectories[tag] = trajectory

        self._augmented_lagrangian_fn = self._negative_augmented_lagrangian

        return self._augmented_lagrangian_fn

    def _get_initial_penalty(self,
                             datasets: Optional[Mapping[Tag, Dataset]] = None):
        """
        :param datasets: The data from the observer.
        :return: Initial penalty to use in Augmented Lagrangian, calculated as done by https://www.mcs.anl.gov/papers/P5401-0915.pdf
                 and https://arxiv.org/pdf/1605.09466.pdf
        """
        all_satisfied = None
        sum_squared = None
        if self._inequality_constraint_prefix is not None:
            inequality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._inequality_constraint_prefix)]
            for tag in inequality_constraint_tags:
                constraint_satisfied = tf.squeeze(datasets[tag].observations) <= 0
                if all_satisfied is None:
                    all_satisfied = constraint_satisfied
                else:
                    all_satisfied = tf.logical_and(all_satisfied, constraint_satisfied)

                # Only consider invalid inequality constraints (i.e. > 0)
                constraint_squared = tf.square(tf.nn.relu(tf.squeeze(datasets[tag].observations)))
                if sum_squared is None:
                    sum_squared = constraint_squared
                else:
                    sum_squared += constraint_squared

        if self._equality_constraint_prefix is not None:
            equality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._equality_constraint_prefix)]
            for tag in equality_constraint_tags:
                constraint_satisfied = tf.abs(tf.squeeze(datasets[tag].observations)) <= self._epsilon
                if all_satisfied is None:
                    all_satisfied = constraint_satisfied
                else:
                    all_satisfied = tf.logical_and(all_satisfied, constraint_satisfied)

                constraint_squared = tf.square(tf.squeeze(datasets[tag].observations))
                if sum_squared is None:
                    sum_squared = constraint_squared
                else:
                    sum_squared += constraint_squared

        at_least_one_violated = tf.logical_not(all_satisfied)

        if tf.reduce_sum(tf.cast(at_least_one_violated, tf.int32)) == 0:
            # If valid everywhere, set initial penalty to one
            initial_penalty = 1
        else:
            sum_squared = sum_squared[at_least_one_violated]
            min_sum_squared = tf.math.reduce_min(sum_squared)
            if tf.reduce_sum(tf.cast(all_satisfied, tf.int32)) == 0:
                denominator = 2 * tfp.stats.percentile(datasets[self._objective_tag].observations, 50.0, interpolation='midpoint')
            else:
                best_valid_objective = tf.math.reduce_min(datasets[self._objective_tag].observations[all_satisfied])
                denominator = 2 * best_valid_objective
            initial_penalty = min_sum_squared / denominator

        return tf.cast(tf.abs(initial_penalty), dtype=tf.float64)

    def update_acquisition_function(
            self,
            function: AcquisitionFunction,
            models: Mapping[Tag, ProbabilisticModelType],
            datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param models: The models for each tag.
        :param datasets: The data from the observer.
        :return: The augmented Lagrangian function, updating sampled unknown functions (ideally with decoupled sampling).
        """
        tf.debugging.Assert(datasets is not None, [tf.constant([])])
        self._iteration += 1
        print(f"Iteration: {self._iteration}")
        datasets = cast(Mapping[Tag, Dataset], datasets)

        objective_dataset = datasets[self._objective_tag]

        tf.debugging.assert_positive(
            len(objective_dataset),
            message="Lagrange multiplier updates are defined with respect to existing points in the"
                    " objective data, but the objective data is empty.",
        )

        # Find the best valid objective value seen so far
        satisfied_mask = tf.constant(value=True, shape=datasets[self._objective_tag].observations.shape)
        for tag, dataset in datasets.items():
            if (self._inequality_constraint_prefix is not None) and (tag.startswith(self._inequality_constraint_prefix)):
                constraint_vals = dataset.observations
                valid_constraint_vals = constraint_vals <= 0
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)
            elif (self._equality_constraint_prefix is not None) and (tag.startswith(self._equality_constraint_prefix)):
                constraint_vals = dataset.observations
                valid_constraint_vals = tf.abs(constraint_vals) <= self._epsilon
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)

        if tf.reduce_sum(tf.cast(satisfied_mask, tf.int32)) != 0:
            objective_vals = datasets[self._objective_tag].observations
            valid_y = tf.boolean_mask(objective_vals, satisfied_mask)
            self._best_valid_observation = tf.math.reduce_min(valid_y)
        print(f"Best Valid Observation: {self._best_valid_observation}")

        # Update sampled trajectories
        if self._known_objective is None:
            # Only update objective trajectory if it is unknown, and hence we're modelling it
            self._objective_trajectory = self._objective_trajectory_sampler.update_trajectory(self._objective_trajectory)
        for tag, sampler in self._inequality_constraint_trajectory_samplers.items():
            old_trajectory = self._inequality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            self._inequality_constraint_trajectories[tag] = updated_trajectory

        for tag, sampler in self._equality_constraint_trajectory_samplers.items():
            old_trajectory = self._equality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            self._equality_constraint_trajectories[tag] = updated_trajectory

        # Obtain Augmented Lagrangian values at observed points before updating Lagrange multipliers and penalty term
        augmented_lagrangian_values_at_observed_points = self._deterministic_augmented_lagrangian(datasets)
        best_observed_augmented_lagrangian_idx = int(tf.argmin(augmented_lagrangian_values_at_observed_points))

        # Update Lagrange multipliers
        if self._update_lagrange_via_kkt:
            self._kkt_lagrange_update(datasets, best_observed_augmented_lagrangian_idx)
        else:
            self._default_lagrange_update(datasets, best_observed_augmented_lagrangian_idx)

        # Update penalty
        self._update_penalty(datasets, best_observed_augmented_lagrangian_idx)

        opt_objective_x = datasets[self._objective_tag].query_points[-self._batch_size:]
        opt_objective_value = datasets[self._objective_tag].observations[-self._batch_size:]
        print(f"Most Recent Objective X: {opt_objective_x} Value: {opt_objective_value}")
        print(f"Most Recent Ineq One Val: {datasets['INEQUALITY_CONSTRAINT_ONE'].observations[-self._batch_size:]}")
        print(f"Most Recent Ineq Two Val: {datasets['INEQUALITY_CONSTRAINT_TWO'].observations[-self._batch_size:]}")

        if self._save_lambda:
            # Store inequality lambda values if we're wanting to save lambda values
            if self._inequality_lambda is not None:
                for k, v in self._inequality_lambda.items():
                    self._inequality_lambda_tracker[k].append(v)

            # Store equality lambda values if we're wanting to save lambda values
            if self._equality_lambda is not None:
                for k, v in self._equality_lambda.items():
                    self._equality_lambda_tracker[k].append(v)

            self._penalty_tracker.append(self._penalty)

            if self._iteration == self._num_bo_iters:
                if self._inequality_lambda is not None:
                    with open(self._save_path + "_inequality_lambda_progression.pkl", "wb") as fp:
                        pickle.dump(self._inequality_lambda_tracker, fp)

                if self._equality_lambda is not None:
                    with open(self._save_path + "_equality_lambda_progression.pkl", "wb") as fp:
                        pickle.dump(self._equality_lambda_tracker, fp)

                with open(self._save_path + "_penalty_progression.pkl", "wb") as fp:
                    pickle.dump(self._penalty_tracker, fp)

        if self._plot:
            self._plot_models(opt_objective_x)

        self._augmented_lagrangian_fn = self._negative_augmented_lagrangian

        return self._augmented_lagrangian_fn

    def _kkt_lagrange_update(self,
                             datasets: Mapping[Tag, Dataset],
                             best_observed_augmented_lagrangian_idx: int):
        """
        :param datasets: Datasets containing previously observed points
        :param best_observed_augmented_lagrangian_idx: Index of observed point in dataset which minimises the Augmented
                                                       Lagrangian.

        Updates Lagrange multipliers using a gradient-based method utilising KKT conditions.
        """
        best_query_point = datasets[self._objective_tag].query_points[best_observed_augmented_lagrangian_idx][None, None, ...]  # [1, 1, D]
        best_query_point = tf.repeat(best_query_point, self._batch_size, axis=1)  # [1, B, D]

        # Get gradients of objective trajectories
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(best_query_point)
            objective_predictions = self._objective_trajectory(best_query_point)
        objective_grad = tf.squeeze(tape.gradient(objective_predictions, best_query_point), axis=0)[0][..., None]  # [D, 1]

        # Get gradients of inequality constraints
        binding_inequality_constraint_grad_dict = OrderedDict()
        for tag, trajectory in self._inequality_constraint_trajectories.items():
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(best_query_point)
                constraint_predictions = trajectory(best_query_point)
            binding = tf.squeeze(tf.abs(constraint_predictions) <= self._epsilon, axis=0)  # If constraint near 0, consider to be binding, shape: [B, 1]
            binding = binding[0]
            constraint_grad = tf.squeeze(tape.gradient(constraint_predictions, best_query_point), axis=0)[0][..., None]  # [D, 1]
            if binding:
                binding_inequality_constraint_grad_dict[tag] = constraint_grad

        # Get gradients of equality constraints
        binding_equality_constraint_grad_dict = OrderedDict()
        for tag, trajectory in self._equality_constraint_trajectories.items():
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(best_query_point)
                constraint_predictions = trajectory(best_query_point)
            binding = tf.squeeze(tf.abs(constraint_predictions) <= self._epsilon, axis=0)  # If constraint near 0, consider to be binding, shape: [B, 1]
            binding = binding[0]
            constraint_grad = tf.squeeze(tape.gradient(constraint_predictions, best_query_point), axis=0)[0][..., None]  # [D, 1]
            if binding:
                binding_equality_constraint_grad_dict[tag] = constraint_grad

        # Dictionaries mapping tag -> updated Lagrange multiplier
        updated_inequality_lagrange_multipliers = {}
        updated_equality_lagrange_multipliers = {}

        # Construct gradient matrix
        gradient_matrix = None

        for tag, gradient in binding_inequality_constraint_grad_dict.items():
            if gradient_matrix is None:
                gradient_matrix = gradient
            else:
                gradient_matrix = tf.concat((gradient_matrix, gradient), axis=1)

        for tag, gradient in binding_equality_constraint_grad_dict.items():
            if gradient_matrix is None:
                gradient_matrix = gradient
            else:
                gradient_matrix = tf.concat((gradient_matrix, gradient), axis=1)

        if gradient_matrix is None:
            # No constraints are considered binding, so all Lagrange multipliers are set to 0
            for tag in self._inequality_lambda.keys():
                updated_inequality_lagrange_multipliers[tag] = [0]

            for tag in self._equality_lambda.keys():
                updated_equality_lagrange_multipliers[tag] = [0]
        else:
            num_binding_inequality_constraints = len(binding_inequality_constraint_grad_dict)
            num_binding_equality_constraints = len(binding_equality_constraint_grad_dict)
            lagrange_multipliers = tf.linalg.lstsq(gradient_matrix, -objective_grad)

            if num_binding_inequality_constraints > 0:
                # Enforce non-negativity of inequality constraint Lagrange multipliers
                inequality_lagrange_multipliers = tf.nn.relu(lagrange_multipliers[:num_binding_inequality_constraints])
                binding_inequality_constraint_tags = list(binding_inequality_constraint_grad_dict.keys())
                for i in range(len(binding_inequality_constraint_tags)):
                    tag = binding_inequality_constraint_tags[i]
                    updated_inequality_lagrange_multipliers[tag] = [tf.squeeze(inequality_lagrange_multipliers[i])]

                # Give non-binding constraints Lagrange multipliers of 0
                for tag in self._inequality_lambda.keys():
                    if tag not in binding_inequality_constraint_tags:
                        updated_inequality_lagrange_multipliers[tag] = [0]

            if num_binding_equality_constraints > 0:
                equality_lagrange_multipliers = lagrange_multipliers[num_binding_inequality_constraints:]
                binding_equality_constraint_tags = list(binding_equality_constraint_grad_dict.keys())
                for i in range(len(binding_equality_constraint_tags)):
                    tag = binding_equality_constraint_tags[i]
                    updated_equality_lagrange_multipliers[tag] = [tf.squeeze(equality_lagrange_multipliers[i])]

                # Give non-binding constraints Lagrange multipliers of 0
                for tag in self._equality_lambda.keys():
                    if tag not in binding_equality_constraint_tags:
                        updated_equality_lagrange_multipliers[tag] = [0]

        for tag, updated_lagrange_multiplier in updated_inequality_lagrange_multipliers.items():
            self._inequality_lambda[tag] = tf.Variable(updated_lagrange_multiplier, dtype=tf.float64)

        for tag, updated_lagrange_multiplier in updated_equality_lagrange_multipliers.items():
            self._equality_lambda[tag] = tf.Variable(updated_lagrange_multiplier, dtype=tf.float64)

    def _default_lagrange_update(self,
                                 datasets: Mapping[Tag, Dataset],
                                 best_observed_augmented_lagrangian_idx: int):
        """
        :param datasets: Datasets containing previously observed points
        :param best_observed_augmented_lagrangian_idx: Index of observed point in dataset which minimises the Augmented
                                                       Lagrangian.

        Updates Lagrange multipliers given previously observed data in accordance with Picheny et al.'s paper.
        """
        print(f"Best Index: {best_observed_augmented_lagrangian_idx}")

        # Update Lagrange multipliers for inequality constraints
        if self._inequality_constraint_prefix is not None:
            inequality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._inequality_constraint_prefix)]
            for tag in inequality_constraint_tags:
                inequality_constraint_val = datasets[tag].observations[best_observed_augmented_lagrangian_idx]
                tf.debugging.assert_shapes([(inequality_constraint_val[..., None, None], (1, 1, 1))])
                slack_val = tf.squeeze(self._obtain_slacks(inequality_constraint_val[..., None, None], self._inequality_lambda[tag], self._penalty), [-1, -2])
                assert (slack_val.shape == self._inequality_lambda[tag].shape)
                assert (slack_val.shape == inequality_constraint_val.shape)
                updated_multiplier = self._inequality_lambda[tag] + (1 / self._penalty) * (inequality_constraint_val + slack_val)
                self._inequality_lambda[tag] = updated_multiplier

        # Update Lagrange multipliers for equality constraints
        if self._equality_constraint_prefix is not None:
            equality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._equality_constraint_prefix)]
            for tag in equality_constraint_tags:
                equality_constraint_val = datasets[tag].observations[best_observed_augmented_lagrangian_idx]
                updated_multiplier = self._equality_lambda[tag] + (1 / self._penalty) * equality_constraint_val
                self._equality_lambda[tag] = updated_multiplier

    def _update_penalty(self,
                        datasets: Mapping[Tag, Dataset],
                        best_observed_augmented_lagrangian_idx: int):
        """
        :param datasets: Datasets containing previously observed points
        :param best_observed_augmented_lagrangian_idx: Index of observed point in dataset which minimises the Augmented
                                                       Lagrangian.

        Updates penalty term given previously observed data in accordance with Picheny et al.'s paper.
        """

        # See if inequality constraints are satisfied
        inequality_constraints_satisfied = True
        if self._inequality_constraint_prefix is not None:
            inequality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._inequality_constraint_prefix)]
            for tag in inequality_constraint_tags:
                inequality_constraint_val = datasets[tag].observations[best_observed_augmented_lagrangian_idx]
                inequality_constraints_satisfied = inequality_constraints_satisfied and inequality_constraint_val <= 0

        # See if equality constraints are satisfied
        equality_constraints_satisfied = True
        if self._equality_constraint_prefix is not None:
            equality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._equality_constraint_prefix)]
            for tag in equality_constraint_tags:
                equality_constraint_val = datasets[tag].observations[best_observed_augmented_lagrangian_idx]
                equality_constraints_satisfied = equality_constraints_satisfied and tf.abs(equality_constraint_val) <= self._epsilon

        if not (equality_constraints_satisfied and inequality_constraints_satisfied):
            if self._conservative_penalty_decrease and self._best_valid_observation is None:
                decreasing_constant = 10/9
            else:
                decreasing_constant = 2.0
            print(f"Decreasing constant: {decreasing_constant}")
            self._penalty = self._penalty / tf.pow(tf.constant(decreasing_constant, dtype=tf.float64), self._batch_size)
            print(f"Not Satisfied. Updated Penalty: {self._penalty}")
        else:
            print(f"Satisfied")

    def _deterministic_augmented_lagrangian(self,
                                            datasets: Mapping[Tag, Dataset]) -> TensorType:
        """
        :param datasets: Datasets containing previously observed points.
        :returns: Augmented Lagrangian values at observed points in datasets.
        """

        objective_vals = datasets[self._objective_tag].observations[..., None]  # [N, 1, 1]

        sum_equality_lambda_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        sum_equality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)

        if self._equality_constraint_prefix is not None:
            equality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._equality_constraint_prefix)]
            for tag in equality_constraint_tags:
                equality_constraint_vals = datasets[tag].observations[..., None]  # [N, 1, 1]
                equality_lambda_scaled = self._equality_lambda[tag] * equality_constraint_vals  # [N, 1, 1]
                equality_penalty_scaled = (1 / (2 * self._penalty)) * tf.square(equality_constraint_vals)
                assert (equality_penalty_scaled.shape == sum_equality_penalty_scaled.shape)
                assert (equality_lambda_scaled.shape == sum_equality_lambda_scaled.shape)
                sum_equality_penalty_scaled += equality_penalty_scaled
                sum_equality_lambda_scaled += equality_lambda_scaled

        # Calculate inequality constraint values using the sampled trajectory, and scale them with Lagrange multipliers
        # and penalty parameter
        sum_inequality_lambda_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        sum_inequality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        if self._inequality_constraint_prefix is not None:
            inequality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._inequality_constraint_prefix)]
            for tag in inequality_constraint_tags:
                inequality_constraint_vals = datasets[tag].observations[..., None]  # [N, 1, 1]
                slack_vals = self._obtain_slacks(inequality_constraint_vals, self._inequality_lambda[tag],
                                                 self._penalty)  # [N, 1, 1]
                assert (slack_vals.shape == inequality_constraint_vals.shape)
                inequality_plus_slack = inequality_constraint_vals + slack_vals
                inequality_lambda_scaled = self._inequality_lambda[tag] * inequality_plus_slack  # [N, B, 1] - This could lose first axis if N = 1
                inequality_penalty_scaled = (1 / (2 * self._penalty)) * tf.square(inequality_plus_slack)
                assert (inequality_penalty_scaled.shape == sum_inequality_penalty_scaled.shape)
                assert (inequality_lambda_scaled.shape == sum_inequality_lambda_scaled.shape)
                sum_inequality_penalty_scaled += inequality_penalty_scaled
                sum_inequality_lambda_scaled += inequality_lambda_scaled

        # Return negative of augmented Lagrangian
        al = tf.squeeze(objective_vals + sum_equality_lambda_scaled + sum_inequality_lambda_scaled + sum_equality_penalty_scaled + sum_inequality_penalty_scaled, -1)
        return al

    def _negative_augmented_lagrangian(
            self,
            x: tf.Tensor) -> tf.Tensor:
        """
        Form negative of augmented Lagrangian (since acquisition function optimiser *maximises* the returned acquisition
        function)

        :param x: Array of points at which to evaluate the augmented Lagrangian of shape [N, B, D] (middle axis is for batching
                  when calling the sampled trajectories)
        :return: Values of negative augmented Lagrangian at given x values of shape [N, B]
        """
        objective_vals = self._objective_trajectory(x)

        sum_equality_lambda_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        sum_equality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        if self._equality_constraint_prefix is not None:
            for tag, equality_constraint_trajectory in self._equality_constraint_trajectories.items():
                equality_constraint_vals = equality_constraint_trajectory(x)  # [N, B, 1]
                equality_lambda_scaled = self._equality_lambda[tag] * equality_constraint_vals  # [N, B, 1]
                equality_penalty_scaled = (1 / (2 * self._penalty)) * tf.square(equality_constraint_vals)
                assert (equality_penalty_scaled.shape == sum_equality_penalty_scaled.shape)
                assert (equality_lambda_scaled.shape == sum_equality_lambda_scaled.shape)
                sum_equality_penalty_scaled += equality_penalty_scaled
                sum_equality_lambda_scaled += equality_lambda_scaled

        # Calculate inequality constraint values using the sampled trajectory, and scale them with Lagrange multipliers
        # and penalty parameter
        sum_inequality_lambda_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        sum_inequality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        if self._inequality_constraint_prefix is not None:
            for tag, inequality_constraint_trajectory in self._inequality_constraint_trajectories.items():
                inequality_constraint_vals = inequality_constraint_trajectory(x)  # [N, B, 1]
                slack_vals = self._obtain_slacks(inequality_constraint_vals, self._inequality_lambda[tag],
                                                 self._penalty)  # [N, B, 1]
                assert (slack_vals.shape == inequality_constraint_vals.shape)
                inequality_plus_slack = inequality_constraint_vals + slack_vals
                inequality_lambda_scaled = self._inequality_lambda[tag] * inequality_plus_slack  # [N, B, 1] - This could lose first axis if N = 1
                inequality_penalty_scaled = (1 / (2 * self._penalty)) * tf.square(inequality_plus_slack)
                assert (inequality_penalty_scaled.shape == sum_inequality_penalty_scaled.shape)
                assert (inequality_lambda_scaled.shape == sum_inequality_lambda_scaled.shape)
                sum_inequality_penalty_scaled += inequality_penalty_scaled
                sum_inequality_lambda_scaled += inequality_lambda_scaled

        # Return negative of augmented Lagrangian
        neg_al = tf.squeeze(- objective_vals - sum_equality_lambda_scaled - sum_inequality_lambda_scaled - sum_equality_penalty_scaled - sum_inequality_penalty_scaled, -1)
        return neg_al

    def _obtain_slacks(
            self,
            inequality_constraint_vals: tf.Tensor,
            inequality_lambda: tf.Variable,
            penalty: tf.Variable) -> tf.Tensor:
        """
        Obtain optimal slack values for augmented Lagrangian
        :param inequality_constraint_vals: Inequality constraint values of shape [N, B, 1]
        :param inequality_lambda: Lagrange multiplier for given inequality constraint
        :param penalty: Penalty for constraint violation
        :return: Optimal slack values at each x location, of shape [N, B, 1]
        """
        tf.debugging.assert_rank(inequality_constraint_vals, 3)
        slack_vals = - (inequality_lambda * penalty) - inequality_constraint_vals  # TODO: Perhaps make broadcasting explicit?
        slack_vals_non_neg = tf.nn.relu(slack_vals)
        tf.debugging.assert_shapes([(slack_vals_non_neg, (..., 1))])
        return slack_vals_non_neg


class DeterministicAugmentedLagrangian(VectorizedAcquisitionFunctionBuilder[HasTrajectorySampler]):
    """
    Builder for an augmented Lagrangian acquisition function using Thompson sampling.
    """

    def __init__(
        self,
        objective_tag: Tag,
        inequality_constraint_prefix: Optional[Tag] = None,
        equality_constraint_prefix: Optional[Tag] = None,
        inequality_lambda: Optional[Mapping[Tag, TensorType]] = None,
        equality_lambda: Optional[Mapping[Tag, TensorType]] = None,
        batch_size: int = 1,
        penalty: TensorType = None,
        conservative_penalty_decrease: bool = False,
        epsilon: float = 0.01,
        update_lagrange_via_kkt: bool = False,
        search_space: Optional[SearchSpace] = None,
        plot: bool = False,
        save_lambda: bool = False,
        save_path: str = None,
        num_bo_iters: int = None,
        gsbp: bool = False
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param inequality_constraint_prefix: Prefix of tags for inequality constraint data/models/Lagrange multipliers.
        :param equality_constraint_prefix: Prefix for tags for equality constraint data/models/Lagrange multipliers.
        :param inequality_lambda: Initial values for Lagrange multipliers of inequality constraints.
        :param equality_lambda: Initial values for Lagrange multipliers of equality constraints.
        :param penalty: Initial penalty. If None, it is set to a default value using the _get_initial_penalty method.
        :param conservative_penalty_decrease: Whether to conservatively decrease the penalty parameter if no valid solution
                                              has been found yet.
        :param epsilon: Bound within which equality constraints are considered to be satisfied.
        :param update_lagrange_via_kkt: Whether to update Lagrange multipliers with a gradient-based approach based on
                                        KKT conditions
        :param search_space: The global search space over which the optimisation is defined. This is
            only used to determine explicit constraints.
        :param save_lambda: Whether to keep track of lagrange multipliers at each iteration (for experiments).
        :param save_path: Path to save lagrange multipliers to (method of saving is currently hacky (i.e. need to manually
                          specify num iters of BayesOpt in the code here, but is fine for sake of personal experimentation).
        """
        self._objective_trajectory = None
        self._objective_trajectory_sampler = None
        self._objective_tag = objective_tag
        self._inequality_constraint_prefix = inequality_constraint_prefix
        self._equality_constraint_prefix = equality_constraint_prefix
        if inequality_lambda is None:
            self._inequality_lambda = {}
        else:
            self._inequality_lambda = inequality_lambda
        if equality_lambda is None:
            self._equality_lambda = {}
        else:
            self._equality_lambda = equality_lambda
        self._batch_size = batch_size
        self._penalty = penalty
        self._epsilon = epsilon
        self._update_lagrange_via_kkt = update_lagrange_via_kkt
        self._search_space = search_space
        self._augmented_lagrangian_fn = None
        self._plot = plot
        self._save_lambda = save_lambda
        self._save_path = save_path
        self._num_bo_iters = num_bo_iters
        self._iteration = 0
        self._inequality_lambda_tracker = {}
        self._equality_lambda_tracker = {}
        self._penalty_tracker = [self._penalty]
        self._stop_decreasing_penalty = False
        self._best_valid_observation = None  # Kept to None until a valid observation is found.
        self._conservative_penalty_decrease = conservative_penalty_decrease
        self._gsbp = gsbp

        if self._inequality_lambda is not None:
            for k, v in self._inequality_lambda.items():
                self._inequality_lambda_tracker[k] = [v]

        if self._equality_lambda is not None:
            for k, v in self._equality_lambda.items():
                self._equality_lambda_tracker[k] = [v]

    def __repr__(self) -> str:
        """"""
        return (
            f"DeterministicAugmentedLagrangian({self._objective_tag!r}, {self._inequality_constraint_prefix!r},"
            f" {self._equality_constraint_prefix!r}, {self._search_space!r})"
        )

    def prepare_acquisition_function(
        self,
        models: Mapping[Tag, HasTrajectorySampler],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param models: The models over each tag.
        :param datasets: The data from the observer.
        :return: The augmented Lagrangian function, using Thompson sampling to approximate all unknown functions.
        :raise KeyError: If `objective_tag` is not found in ``datasets`` and ``models``.
        :raise tf.errors.InvalidArgumentError: If the objective data is empty.
        """
        tf.debugging.Assert(datasets is not None, [tf.constant([])])
        self._iteration += 1

        # Find the best valid objective value seen so far
        satisfied_mask = tf.constant(value=True, shape=datasets[self._objective_tag].observations.shape)
        for tag, dataset in datasets.items():
            if (self._inequality_constraint_prefix is not None) and (tag.startswith(self._inequality_constraint_prefix)):
                constraint_vals = dataset.observations
                valid_constraint_vals = constraint_vals <= 0
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)
            elif (self._equality_constraint_prefix is not None) and (tag.startswith(self._equality_constraint_prefix)):
                constraint_vals = dataset.observations
                valid_constraint_vals = tf.abs(constraint_vals) <= self._epsilon
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)

        if tf.reduce_sum(tf.cast(satisfied_mask, tf.int32)) != 0:
            objective_vals = datasets[self._objective_tag].observations
            valid_y = tf.boolean_mask(objective_vals, satisfied_mask)
            self._best_valid_observation = tf.math.reduce_min(valid_y)
            print(f"Best Valid Observation: {self._best_valid_observation}")

        if self._penalty is None:
            initial_penalty = self._get_initial_penalty(datasets)
            print(f"Initial Penalty: {initial_penalty}")
            self._penalty = tf.constant(initial_penalty, dtype=tf.float64)

        self._augmented_lagrangian_fn = self._negative_augmented_lagrangian

        return self._augmented_lagrangian_fn

    def _get_initial_penalty(self,
                             datasets: Optional[Mapping[Tag, Dataset]] = None):
        """
        :param datasets: The data from the observer.
        :return: Initial penalty to use in Augmented Lagrangian, calculated as done by https://www.mcs.anl.gov/papers/P5401-0915.pdf
                 and https://arxiv.org/pdf/1605.09466.pdf
        """
        all_satisfied = None
        sum_squared = None
        if self._inequality_constraint_prefix is not None:
            inequality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._inequality_constraint_prefix)]
            for tag in inequality_constraint_tags:
                constraint_satisfied = tf.squeeze(datasets[tag].observations) <= 0
                if all_satisfied is None:
                    all_satisfied = constraint_satisfied
                else:
                    all_satisfied = tf.logical_and(all_satisfied, constraint_satisfied)

                # Only consider invalid inequality constraints (i.e. > 0)
                constraint_squared = tf.square(tf.nn.relu(tf.squeeze(datasets[tag].observations)))
                if sum_squared is None:
                    sum_squared = constraint_squared
                else:
                    sum_squared += constraint_squared

        if self._equality_constraint_prefix is not None:
            equality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._equality_constraint_prefix)]
            for tag in equality_constraint_tags:
                constraint_satisfied = tf.abs(tf.squeeze(datasets[tag].observations)) <= self._epsilon
                if all_satisfied is None:
                    all_satisfied = constraint_satisfied
                else:
                    all_satisfied = tf.logical_and(all_satisfied, constraint_satisfied)

                constraint_squared = tf.square(tf.squeeze(datasets[tag].observations))
                if sum_squared is None:
                    sum_squared = constraint_squared
                else:
                    sum_squared += constraint_squared

        at_least_one_violated = tf.logical_not(all_satisfied)

        if tf.reduce_sum(tf.cast(at_least_one_violated, tf.int32)) == 0:
            # If valid everywhere, set initial penalty to one
            initial_penalty = 1
        else:
            sum_squared = sum_squared[at_least_one_violated]
            min_sum_squared = tf.math.reduce_min(sum_squared)
            if tf.reduce_sum(tf.cast(all_satisfied, tf.int32)) == 0:
                denominator = 2 * tfp.stats.percentile(datasets[self._objective_tag].observations, 50.0, interpolation='midpoint')
            else:
                best_valid_objective = tf.math.reduce_min(datasets[self._objective_tag].observations[all_satisfied])
                denominator = 2 * best_valid_objective
            initial_penalty = min_sum_squared / denominator

        return tf.cast(tf.abs(initial_penalty), dtype=tf.float64)

    def update_acquisition_function(
            self,
            function: AcquisitionFunction,
            models: Mapping[Tag, ProbabilisticModelType],
            datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param models: The models for each tag.
        :param datasets: The data from the observer.
        :return: The augmented Lagrangian function, updating sampled unknown functions (ideally with decoupled sampling).
        """
        tf.debugging.Assert(datasets is not None, [tf.constant([])])
        self._iteration += 1
        print(f"Iteration: {self._iteration}")
        datasets = cast(Mapping[Tag, Dataset], datasets)

        print(f"AL @ 0.92148052 0.80743449: {self._negative_augmented_lagrangian(tf.Variable([[[0.92148052, 0.80743449]]], dtype=tf.float64))}")
        print(f"AL @ 0.19418144 0.40449058: {self._negative_augmented_lagrangian(tf.Variable([[[0.19418144, 0.40449058]]], dtype=tf.float64))}")

        x1 = tf.Variable([[[0.2, 0.2]]], dtype=tf.float64)
        x2 = tf.Variable([[[0.5, 0.5]]], dtype=tf.float64)
        x3 = tf.Variable([[[0.38604815, 0.38604815]]], dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x1, x2, x3])
            y1 = self._negative_augmented_lagrangian(x1)
            y2 = self._negative_augmented_lagrangian(x2)
            y3 = self._negative_augmented_lagrangian(x3)

        print(f"Penalty: {self._penalty}")
        print(f"x1 Val: {y1} Gradient: {tape.gradient(y1, x1)}")
        print(f"Toy Cons 1 @ x1: {constraints.toy_constraint_one(tf.squeeze(x1, 0))}")
        print(f"Lagrange Multiplier: {self._inequality_lambda['INEQUALITY_CONSTRAINT_ONE']}")
        print(f"Val: {y3} Grad @ (0.38604815 0.38604815): {tape.gradient(y3, x3)}")
        print(f"x2 Val: {y2} Gradient: {tape.gradient(y2, x2)}")


        objective_dataset = datasets[self._objective_tag]

        tf.debugging.assert_positive(
            len(objective_dataset),
            message="Lagrange multiplier updates are defined with respect to existing points in the"
                    " objective data, but the objective data is empty.",
        )

        # Find the best valid objective value seen so far
        satisfied_mask = tf.constant(value=True, shape=datasets[self._objective_tag].observations.shape)
        for tag, dataset in datasets.items():
            if (self._inequality_constraint_prefix is not None) and (tag.startswith(self._inequality_constraint_prefix)):
                constraint_vals = dataset.observations
                valid_constraint_vals = constraint_vals <= 0
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)
            elif (self._equality_constraint_prefix is not None) and (tag.startswith(self._equality_constraint_prefix)):
                constraint_vals = dataset.observations
                valid_constraint_vals = tf.abs(constraint_vals) <= self._epsilon
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)

        if tf.reduce_sum(tf.cast(satisfied_mask, tf.int32)) != 0:
            objective_vals = datasets[self._objective_tag].observations
            valid_y = tf.boolean_mask(objective_vals, satisfied_mask)
            self._best_valid_observation = tf.math.reduce_min(valid_y)
        print(f"Best Valid Observation: {self._best_valid_observation}")

        # Obtain Augmented Lagrangian values at observed points before updating Lagrange multipliers and penalty term
        augmented_lagrangian_values_at_observed_points = self._deterministic_augmented_lagrangian(datasets)
        best_observed_augmented_lagrangian_idx = int(tf.argmin(augmented_lagrangian_values_at_observed_points))
        best_observed_augmented_lagrangian_idx = -1

        # Update Lagrange multipliers
        self._default_lagrange_update(datasets, best_observed_augmented_lagrangian_idx)

        # Update penalty
        if not self._stop_decreasing_penalty:
            self._update_penalty(datasets, best_observed_augmented_lagrangian_idx)

        opt_objective_x = datasets[self._objective_tag].query_points[-self._batch_size:]
        opt_objective_value = datasets[self._objective_tag].observations[-self._batch_size:]
        print(f"Most Recent Objective X: {opt_objective_x} Value: {opt_objective_value}")
        print(f"Most Recent Ineq One Val: {datasets['INEQUALITY_CONSTRAINT_ONE'].observations[-self._batch_size:]}")
        # print(f"Most Recent Ineq Two Val: {datasets['INEQUALITY_CONSTRAINT_TWO'].observations[-self._batch_size:]}")

        self._augmented_lagrangian_fn = self._negative_augmented_lagrangian

        return self._augmented_lagrangian_fn

    def _default_lagrange_update(self,
                                 datasets: Mapping[Tag, Dataset],
                                 best_observed_augmented_lagrangian_idx: int):
        """
        :param datasets: Datasets containing previously observed points
        :param best_observed_augmented_lagrangian_idx: Index of observed point in dataset which minimises the Augmented
                                                       Lagrangian.

        Updates Lagrange multipliers given previously observed data in accordance with Picheny et al.'s paper.
        """
        print(f"Best Index: {best_observed_augmented_lagrangian_idx}")

        # Update Lagrange multipliers for inequality constraints
        if self._inequality_constraint_prefix is not None:
            inequality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._inequality_constraint_prefix)]
            for tag in inequality_constraint_tags:
                inequality_constraint_val = datasets[tag].observations[best_observed_augmented_lagrangian_idx]
                slack_val = self._obtain_slacks(inequality_constraint_val, self._inequality_lambda[tag], self._penalty)
                assert (slack_val.shape == self._inequality_lambda[tag].shape)
                assert (slack_val.shape == inequality_constraint_val.shape)
                updated_multiplier = self._inequality_lambda[tag] + (1 / self._penalty) * (inequality_constraint_val + slack_val)
                self._inequality_lambda[tag] = updated_multiplier

        # Update Lagrange multipliers for equality constraints
        if self._equality_constraint_prefix is not None:
            equality_constraint_tags = [key for key in datasets.keys() if
                                        key.startswith(self._equality_constraint_prefix)]
            for tag in equality_constraint_tags:
                equality_constraint_val = datasets[tag].observations[best_observed_augmented_lagrangian_idx]
                updated_multiplier = self._equality_lambda[tag] + (1 / self._penalty) * equality_constraint_val
                self._equality_lambda[tag] = updated_multiplier

    def _update_penalty(self,
                        datasets: Mapping[Tag, Dataset],
                        best_observed_augmented_lagrangian_idx: int):
        """
        :param datasets: Datasets containing previously observed points
        :param best_observed_augmented_lagrangian_idx: Index of observed point in dataset which minimises the Augmented
                                                       Lagrangian.

        Updates penalty term given previously observed data in accordance with Picheny et al.'s paper.
        """

        # See if inequality constraints are satisfied
        inequality_constraints_satisfied = True
        if self._inequality_constraint_prefix is not None:
            inequality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._inequality_constraint_prefix)]
            for tag in inequality_constraint_tags:
                inequality_constraint_val = datasets[tag].observations[best_observed_augmented_lagrangian_idx]
                inequality_constraints_satisfied = inequality_constraints_satisfied and inequality_constraint_val <= 0

        # See if equality constraints are satisfied
        equality_constraints_satisfied = True
        if self._equality_constraint_prefix is not None:
            equality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._equality_constraint_prefix)]
            for tag in equality_constraint_tags:
                equality_constraint_val = datasets[tag].observations[best_observed_augmented_lagrangian_idx]
                equality_constraints_satisfied = equality_constraints_satisfied and tf.abs(equality_constraint_val) <= self._epsilon

        if not (equality_constraints_satisfied and inequality_constraints_satisfied):
            if self._conservative_penalty_decrease and self._best_valid_observation is None:
                decreasing_constant = 10/9
            else:
                decreasing_constant = 2.0
            print(f"Decreasing constant: {decreasing_constant}")
            self._penalty = self._penalty / tf.pow(tf.constant(decreasing_constant, dtype=tf.float64), self._batch_size)
            print(f"Not Satisfied. Updated Penalty: {self._penalty}")
        else:
            print(f"Satisfied")

    def _deterministic_augmented_lagrangian(self,
                                            datasets: Mapping[Tag, Dataset]) -> TensorType:
        """
        :param datasets: Datasets containing previously observed points.
        :returns: Augmented Lagrangian values at observed points in datasets.
        """

        objective_vals = datasets[self._objective_tag].observations  # [N, 1]

        sum_equality_lambda_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        sum_equality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        if self._equality_constraint_prefix is not None:
            equality_constraint_tags = [key for key in datasets.keys() if
                                        key.startswith(self._equality_constraint_prefix)]
            for tag in equality_constraint_tags:
                equality_constraint_vals = datasets[tag].observations  # [N, 1]
                equality_lambda_scaled = self._equality_lambda[tag] * equality_constraint_vals  # [N, 1]
                equality_penalty_scaled = (1 / (2 * self._penalty)) * tf.square(equality_constraint_vals)
                assert (equality_penalty_scaled.shape == sum_equality_penalty_scaled.shape)
                assert (equality_lambda_scaled.shape == sum_equality_lambda_scaled.shape)
                sum_equality_penalty_scaled += equality_penalty_scaled
                sum_equality_lambda_scaled += equality_lambda_scaled

        # Calculate inequality constraint values using the sampled trajectory, and scale them with Lagrange multipliers
        # and penalty parameter
        sum_inequality_lambda_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        sum_inequality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        if self._inequality_constraint_prefix is not None:
            inequality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._inequality_constraint_prefix)]
            for tag in inequality_constraint_tags:
                inequality_constraint_vals = datasets[tag].observations  # [N, 1]
                slack_vals = - (self._inequality_lambda[tag] * self._penalty) - inequality_constraint_vals  # TODO: Perhaps make broadcasting explicit?
                slack_vals = tf.nn.relu(slack_vals)
                # slack_vals = self._obtain_slacks(inequality_constraint_vals, self._inequality_lambda[tag],
                #                                  self._penalty)  # [N, 1]
                assert (slack_vals.shape == inequality_constraint_vals.shape)
                inequality_plus_slack = inequality_constraint_vals + slack_vals
                inequality_lambda_scaled = self._inequality_lambda[tag] * inequality_plus_slack  # [N, B, 1] - This could lose first axis if N = 1
                inequality_penalty_scaled = (1 / (2 * self._penalty)) * tf.square(inequality_plus_slack)
                assert (inequality_penalty_scaled.shape == sum_inequality_penalty_scaled.shape)
                assert (inequality_lambda_scaled.shape == sum_inequality_lambda_scaled.shape)
                sum_inequality_penalty_scaled += inequality_penalty_scaled
                sum_inequality_lambda_scaled += inequality_lambda_scaled

        # Return augmented Lagrangian
        al = tf.squeeze(objective_vals + sum_equality_lambda_scaled + sum_inequality_lambda_scaled + sum_equality_penalty_scaled + sum_inequality_penalty_scaled, -1)
        return al

    def increase_penalty_and_return_acquisition_function(self):
        """
        Occasionally once penalty becomes very small L-BFGS-B will fail. I strongly suspect this is due to the gradient
        near the constraint satisfaction boundary becoming extremely large. If this starts to happen, we stop decreasing
        the penalty parameter and increase, until optimisation starts to work again.
        """
        self._stop_decreasing_penalty = True
        self._penalty = 2.0 * self._penalty
        self._augmented_lagrangian_fn = self._negative_augmented_lagrangian
        return self._augmented_lagrangian_fn

    def _negative_augmented_lagrangian(
            self,
            x: tf.Tensor) -> tf.Tensor:
        """
        Form negative of augmented Lagrangian (since acquisition function optimiser *maximises* the returned acquisition
        function)

        :param x: Array of points at which to evaluate the augmented Lagrangian of shape [N, B, D] (middle axis is for batching
                  when calling the sampled trajectories)
        :return: Values of negative augmented Lagrangian at given x values of shape [N, B]
        """
        squeezed_x = tf.squeeze(x, 1)

        objective_vals = objectives.linear_objective(squeezed_x)

        sum_equality_lambda_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        sum_equality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        equality_tags = ['EQUALITY_CONSTRAINT_ONE', 'EQUALITY_CONSTRAINT_TWO']

        if self._gsbp:
            if self._equality_constraint_prefix is not None:
                for tag in equality_tags:
                    if tag == 'EQUALITY_CONSTRAINT_ONE':
                        equality_constraint_vals = constraints.centered_branin(squeezed_x)
                    else:
                        equality_constraint_vals = constraints.parr_constraint(squeezed_x)
                    equality_lambda_scaled = self._equality_lambda[tag] * equality_constraint_vals  # [N, 1]
                    equality_penalty_scaled = (1 / (2 * self._penalty)) * tf.square(equality_constraint_vals)
                    assert (equality_penalty_scaled.shape == sum_equality_penalty_scaled.shape)
                    assert (equality_lambda_scaled.shape == sum_equality_lambda_scaled.shape)
                    sum_equality_penalty_scaled += equality_penalty_scaled
                    sum_equality_lambda_scaled += equality_lambda_scaled

        # Calculate inequality constraint values using the sampled trajectory, and scale them with Lagrange multipliers
        # and penalty parameter
        sum_inequality_lambda_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        sum_inequality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)

        if not self._gsbp:
            inequality_tags = ['INEQUALITY_CONSTRAINT_ONE', 'INEQUALITY_CONSTRAINT_TWO']
        else:
            inequality_tags = ['INEQUALITY_CONSTRAINT_ONE']
        if self._inequality_constraint_prefix is not None:
            for tag in inequality_tags:
                if tag == 'INEQUALITY_CONSTRAINT_ONE':
                    inequality_constraint_vals = constraints.toy_constraint_one(squeezed_x)  # [N, 1]
                else:
                    inequality_constraint_vals = constraints.toy_constraint_two(squeezed_x)  # [N, 1]
                # slack_vals = self._obtain_slacks(inequality_constraint_vals, self._inequality_lambda[tag],
                #                                  self._penalty)  # [N, B, 1]
                # assert (slack_vals.shape == inequality_constraint_vals.shape)
                # inequality_lambda_scaled = self._inequality_lambda[tag] * tf.nn.relu(inequality_constraint_vals)  # [N, 1] - This could lose first axis if N = 1
                # inequality_penalty_scaled = (1 / (2 * self._penalty)) * tf.square(tf.nn.relu(inequality_constraint_vals))
                # assert (inequality_penalty_scaled.shape == sum_inequality_penalty_scaled.shape)
                # assert (inequality_lambda_scaled.shape == sum_inequality_lambda_scaled.shape)
                # sum_inequality_penalty_scaled += inequality_penalty_scaled
                # sum_inequality_lambda_scaled += inequality_lambda_scaled
                slack_vals = self._obtain_slacks(inequality_constraint_vals, self._inequality_lambda[tag],
                                                 self._penalty)  # [N, B, 1]
                assert (slack_vals.shape == inequality_constraint_vals.shape)
                inequality_plus_slack = inequality_constraint_vals + slack_vals
                inequality_lambda_scaled = self._inequality_lambda[tag] * inequality_plus_slack  # [N, 1] - This could lose first axis if N = 1
                inequality_penalty_scaled = (1 / (2 * self._penalty)) * tf.square(inequality_plus_slack)
                assert (inequality_penalty_scaled.shape == sum_inequality_penalty_scaled.shape)
                assert (inequality_lambda_scaled.shape == sum_inequality_lambda_scaled.shape)
                sum_inequality_penalty_scaled += inequality_penalty_scaled
                sum_inequality_lambda_scaled += inequality_lambda_scaled

        # Return negative of augmented Lagrangian
        neg_al = - objective_vals - sum_inequality_lambda_scaled - sum_equality_lambda_scaled - sum_inequality_penalty_scaled - sum_equality_penalty_scaled
        return neg_al

    def _obtain_slacks(
            self,
            inequality_constraint_vals: tf.Tensor,
            inequality_lambda: tf.Variable,
            penalty: tf.Variable) -> tf.Tensor:
        """
        Obtain optimal slack values for augmented Lagrangian
        :param inequality_constraint_vals: Inequality constraint values of shape [N, 1]
        :param inequality_lambda: Lagrange multiplier for given inequality constraint
        :param penalty: Penalty for constraint violation
        :return: Optimal slack values at each x location, of shape [N, 1]
        """
        slack_vals = - (inequality_lambda * penalty) - inequality_constraint_vals  # TODO: Perhaps make broadcasting explicit?
        slack_vals_non_neg = tf.nn.relu(slack_vals)
        tf.debugging.assert_shapes([(slack_vals_non_neg, (..., 1))])
        return slack_vals_non_neg
