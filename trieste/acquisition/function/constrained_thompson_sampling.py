from __future__ import annotations

from functools import partial
from typing import Callable, List, Mapping, Optional, cast

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


class ThompsonSamplingAugmentedLagrangian(AcquisitionFunctionBuilder[HasTrajectorySampler]):
    """
    Builder for the *expected constrained improvement* acquisition function defined in
    :cite:`gardner14`. The acquisition function computes the expected improvement from the best
    feasible point, where feasible points are those that (probably) satisfy some constraint. Where
    there are no feasible points, this builder simply builds the constraint function.
    """

    def __init__(
        self,
        objective_tag: Tag,
        inequality_constraint_prefix: Optional[Tag] = None,
        equality_constraint_prefix: Optional[Tag] = None,
        inequality_lambda: Optional[tf.Tensor] = None,
        equality_lambda: Optional[tf.Tensor] = None,
        penalty: tf.Variable = 1,
        epsilon: float = 0.001,
        search_space: Optional[SearchSpace] = None,
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param constraint_builder: The builder for the constraint function.
        :param min_feasibility_probability: The minimum probability of feasibility for a
            "best point" to be considered feasible.
        :param search_space: The global search space over which the optimisation is defined. This is
            only used to determine explicit constraints.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``min_feasibility_probability``
            is not a scalar in the unit interval :math:`[0, 1]`.
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
        :return: The expected constrained improvement acquisition function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        :raise KeyError: If `objective_tag` is not found in ``datasets`` and ``models``.
        :raise tf.errors.InvalidArgumentError: If the objective data is empty.
        """
        tf.debugging.Assert(datasets is not None, [tf.constant([])])
        datasets = cast(Mapping[Tag, Dataset], datasets)

        objective_model = models[self._objective_tag]
        objective_dataset = datasets[self._objective_tag]

        tf.debugging.assert_positive(
            len(objective_dataset),
            message="Expected improvement is defined with respect to existing points in the"
            " objective data, but the objective data is empty.",
        )

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


        self._augmented_lagrangian_fn = partial(self._augmented_lagrangian, self._objective_trajectory,
                                                                   list(self._inequality_constraint_trajectories.values()),
                                                                   list(self._equality_constraint_trajectories.values()))

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
        """
        tf.debugging.Assert(datasets is not None, [tf.constant([])])
        datasets = cast(Mapping[Tag, Dataset], datasets)

        objective_model = models[self._objective_tag]
        objective_dataset = datasets[self._objective_tag]

        tf.debugging.assert_positive(
            len(objective_dataset),
            message="Expected improvement is defined with respect to existing points in the"
            " objective data, but the objective data is empty.",
        )

        # Last point in dataset is most recent estimate of optimal x value
        opt_x = datasets[self._objective_tag].query_points[-1][None, ...]
        tf.debugging.assert_shapes([(opt_x, (1, 2))])

        inequality_constraints_satisfied = True
        equality_constraints_satisfied = True

        # TODO: Rename to make clear it uses opt_x?
        opt_objective_x = datasets[self._objective_tag].query_points[-1]
        opt_objective_value = datasets[self._objective_tag].observations[-1]
        if self._inequality_constraint_prefix is not None:
            inequality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._inequality_constraint_prefix)]
            for j in range(len(self._inequality_lambda)):
                # TODO: May need to reshape below
                inequality_constraint_val = datasets[inequality_constraint_tags[j]].observations[-1][None, None, ...]
                tf.debugging.assert_shapes([(inequality_constraint_val, (1, 1, 1))])
                slack_val = self._obtain_slacks(inequality_constraint_val, self._inequality_lambda[j], self._penalty)
                updated_multiplier = self._inequality_lambda[j][0] + (1 / self._penalty) * (inequality_constraint_val[0] + slack_val)
                self._inequality_lambda = tf.tensor_scatter_nd_update(self._inequality_lambda, [[j]], updated_multiplier)

            # TODO: Below could be sketchy
            inequality_constraints_opt_x = tf.stack([datasets[tag].observations[-1][None, ...] for tag in inequality_constraint_tags])
            num_inequality_constraints_satisfied = tf.reduce_sum(tf.cast(inequality_constraints_opt_x <= 0, tf.int8))
            if num_inequality_constraints_satisfied != len(inequality_constraint_tags):
                inequality_constraints_satisfied = False

        if self._equality_constraint_prefix is not None:
            equality_constraint_tags = [key for key in datasets.keys() if key.startswith(self._equality_constraint_prefix)]
            for j in range(len(self._equality_lambda)):
                equality_constraint_val = datasets[equality_constraint_tags[j]].observations[-1][None, ...]
                updated_multiplier = self._equality_lambda[j][0] + (1 / self._penalty) * equality_constraint_val
                self._equality_lambda = tf.tensor_scatter_nd_update(self._equality_lambda, [[j]], updated_multiplier)

            equality_constraints_opt_x = tf.stack([datasets[tag].observations[-1][None, ...] for tag in equality_constraint_tags])
            num_equality_constraints_satisfied = tf.reduce_sum(tf.cast(tf.abs(equality_constraints_opt_x) <= self._epsilon, tf.int8))
            if num_equality_constraints_satisfied != len(equality_constraint_tags):
                equality_constraints_satisfied = False

        if not (equality_constraints_satisfied and inequality_constraints_satisfied):
            print(f"Not Satisfied")
            self._penalty = self._penalty / 2
        else:
            print(f"Satisfied")

        print(f"Objective X: {opt_objective_x} Value: {opt_objective_value}")

        # Update sampled trajectories
        self._objective_trajectory = self._objective_trajectory_sampler.update_trajectory(self._objective_trajectory)
        print(f"Objective Trajectory: {self._objective_trajectory(tf.constant([[0.5, 0.5]], dtype=tf.float64))}")
        for tag, sampler in self._inequality_constraint_trajectory_samplers.items():
            old_trajectory = self._inequality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            print(f"{tag}: Trajectory: {updated_trajectory(tf.constant([[0.5, 0.5]], dtype=tf.float64))}")
            # TODO: Does sampler get updated? If so, will need to put new sampler back in dict
            self._inequality_constraint_trajectories[tag] = updated_trajectory

        for tag, sampler in self._equality_constraint_trajectory_samplers.items():
            old_trajectory = self._equality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            # TODO: Does sampler get updated? If so, will need to put new sampler back in dict
            self._equality_constraint_trajectories[tag] = updated_trajectory

        self._augmented_lagrangian_fn = partial(self._augmented_lagrangian, self._objective_trajectory,
                                                list(self._inequality_constraint_trajectories.values()),
                                                list(self._equality_constraint_trajectories.values()))

        return self._augmented_lagrangian_fn

    def _augmented_lagrangian(
            self,
            objective: Callable,
            inequality_constraints: Optional[List[Callable]],
            equality_constraints: Optional[List[Callable]],
            x: tf.Tensor) -> tf.Tensor:
        """
        Form augmented Lagrangian from given objective and constraints
        :param objective: Objective function
        :param equality_constraints: List of equality constraint functions
        :param inequality_constraints: List of inequality constraint functions
        :param equality_lambda: Array of Lagrange multipliers for equality constraints of shape ["num_eq_constraints", 1]
        :param inequality_lambda: Array of Lagrange multipliers for inequality constraints of shape ["num_ineq_constraints", 1]
        :param penalty: Penalty parameter for violated constraints
        :param x: Array of points at which to evaluate the lagrangian of shape [N, 1, M]
        :return:
        """
        objective_vals = objective(x)
        objective_vals = tf.squeeze(objective_vals, -2)
        if len(equality_constraints) > 0:
            equality_constraint_vals = tf.stack([constraint(x) for constraint in equality_constraints])
            equality_lambd_scaled = tf.reshape(tf.einsum("ij,ikj->k", self._equality_lambda, equality_constraint_vals),
                                               [-1, 1])
            equality_penalty_scaled = (1 / (2 * self._penalty)) * tf.reduce_sum(tf.square(equality_constraint_vals), axis=0)

        else:
            equality_lambd_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
            equality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)

        if len(inequality_constraints) > 0:
            inequality_constraint_vals = tf.stack([constraint(x) for constraint in inequality_constraints])
            inequality_constraint_vals = tf.squeeze(inequality_constraint_vals, -2)
            slack_vals = tf.stack([self._obtain_slacks(inequality_constraint_vals[i][..., None], self._inequality_lambda[i], self._penalty) for i in
                                   range(len(inequality_constraints))])
            assert (slack_vals.shape == inequality_constraint_vals.shape)
            inequality_plus_slack = inequality_constraint_vals + slack_vals
            inequality_lambd_scaled = tf.reshape(tf.einsum("ij,ikj->k", self._inequality_lambda, inequality_plus_slack),
                                                 [-1, 1])
            inequality_penalty_scaled = (1 / (2 * self._penalty)) * tf.reduce_sum(tf.square(inequality_plus_slack), axis=0)
        else:
            inequality_lambd_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
            inequality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        return - objective_vals - equality_lambd_scaled - inequality_lambd_scaled - equality_penalty_scaled - inequality_penalty_scaled


    def _obtain_slacks(
            self,
            inequality_constraint_vals: tf.Tensor,
            inequality_lambda: tf.Tensor,
            penalty: tf.Variable) -> tf.Tensor:
        """
        Obtain optimal slack values for augmented Lagrangian
        :param inequality_constraint_vals: Inequality constraint values of shape [N, 1, 1]
        :param x: Input values at which slack values should be obtained of shape [N, M]
        :param inequality_lambd: Lagrangian multiplier for given inequality constraint
        :param penalty: Penalty for constraint violation
        :return: Optimal slack values at each x location, of shape [N, 1]
        """
        tf.debugging.assert_rank(inequality_constraint_vals, 3)
        slack_vals = - inequality_lambda[0] * penalty - inequality_constraint_vals
        slack_vals = tf.squeeze(slack_vals, -2)
        slack_vals_non_neg = tf.nn.relu(slack_vals)
        tf.debugging.assert_shapes([(slack_vals_non_neg, (..., 1))])
        return slack_vals_non_neg

