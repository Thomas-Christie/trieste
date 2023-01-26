from __future__ import annotations

from typing import Mapping, Optional, cast

import tensorflow as tf

from ...data import Dataset
from ...models.interfaces import HasTrajectorySampler
from ...space import SearchSpace
from ...types import Tag
from ..interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    ProbabilisticModelType,
)

# Makes use of Lagrange multipliers easier to read, but less vectorised
class UpdatedThompsonSamplingAugmentedLagrangian(AcquisitionFunctionBuilder[HasTrajectorySampler]):
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
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param inequality_constraint_prefix: Prefix of tags for inequality constraint data/models/Lagrange multipliers.
        :param equality_constraint_prefix: Prefix for tags for equality constraint data/models/Lagrange multipliers.
        :param inequality_lambda: Initial values for Lagrange multipliers of inequality constraints.
        :param equality_lambda: Initial values for Lagrange multipliers of equality constraints.
        :param penalty: Initial penalty.
        :param epsilon: Bound within which equality constraints are considered to be satisfied.
        :param search_space: The global search space over which the optimisation is defined. This is
            only used to determine explicit constraints.
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
            inequality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._inequality_constraint_prefix)]
            for tag in inequality_constraint_tags:
                inequality_constraint_val = datasets[tag].observations[-1][None, ...]
                tf.debugging.assert_shapes([(inequality_constraint_val, (1, 1))])
                slack_val = self._obtain_slacks(inequality_constraint_val, self._inequality_lambda[tag], self._penalty)
                updated_multiplier = self._inequality_lambda[tag] + (1 / self._penalty) * (inequality_constraint_val + slack_val)
                self._inequality_lambda[tag] = updated_multiplier

            inequality_constraints_opt_x = tf.stack([datasets[tag].observations[-1] for tag in inequality_constraint_tags])
            num_inequality_constraints_satisfied = tf.reduce_sum(tf.cast(inequality_constraints_opt_x <= 0, tf.int8))
            if num_inequality_constraints_satisfied != len(inequality_constraint_tags):
                inequality_constraints_satisfied = False

        # Update Lagrange multipliers for equality constraints
        # TODO: Test out code for updating equality constraints
        if self._equality_constraint_prefix is not None:
            equality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._equality_constraint_prefix)]
            for tag in equality_constraint_tags:
                equality_constraint_val = datasets[tag].observations[-1][None, ...]
                updated_multiplier = self._equality_lambda[tag] + (1 / self._penalty) * equality_constraint_val
                self._equality_lambda[tag] = updated_multiplier

            equality_constraints_opt_x = tf.stack([datasets[tag].observations[-1] for tag in equality_constraint_tags])
            num_equality_constraints_satisfied = tf.reduce_sum(tf.cast(tf.abs(equality_constraints_opt_x) <= self._epsilon, tf.int8))
            if num_equality_constraints_satisfied != len(equality_constraint_tags):
                equality_constraints_satisfied = False

        print(f"Inequality Lambda: {self._inequality_lambda}")
        if not (equality_constraints_satisfied and inequality_constraints_satisfied):
            print(f"Not Satisfied")
            self._penalty = self._penalty / 2
        else:
            print(f"Satisfied")

        print(f"Objective X: {opt_objective_x} Value: {opt_objective_value}")

        # Update sampled trajectories
        self._objective_trajectory = self._objective_trajectory_sampler.update_trajectory(self._objective_trajectory)
        for tag, sampler in self._inequality_constraint_trajectory_samplers.items():
            old_trajectory = self._inequality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            print(f"{tag}: Trajectory: {updated_trajectory(tf.constant([[0.5, 0.5]], dtype=tf.float64))}")
            self._inequality_constraint_trajectories[tag] = updated_trajectory

        for tag, sampler in self._equality_constraint_trajectory_samplers.items():
            old_trajectory = self._equality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            self._equality_constraint_trajectories[tag] = updated_trajectory

        self._augmented_lagrangian_fn = self._augmented_lagrangian

        return self._augmented_lagrangian_fn

    def _augmented_lagrangian(
            self,
            x: tf.Tensor) -> tf.Tensor:
        """
        Form augmented Lagrangian from given objective and constraints
        :param x: Array of points at which to evaluate the lagrangian of shape [N, 1, M] (middle axis is for batching
                  when calling the sampled trajectories)
        :return:
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
                slack_vals = self._obtain_slacks(inequality_constraint_vals, self._inequality_lambda[tag], self._penalty)
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

