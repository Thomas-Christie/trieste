from __future__ import annotations
from typing import Mapping, Optional, cast, Callable
import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset
from ...models.interfaces import HasTrajectorySampler
from ...observer import OBJECTIVE, INEQUALITY_CONSTRAINT_PREFIX, EQUALITY_CONSTRAINT_PREFIX
from ...space import SearchSpace
from ...types import Tag, TensorType
from ..interface import (
    AcquisitionFunction,
    ProbabilisticModelType,
    VectorizedAcquisitionFunctionBuilder
)


class ThompsonSamplingAugmentedLagrangian(VectorizedAcquisitionFunctionBuilder[HasTrajectorySampler]):
    """
    Builder for an augmented Lagrangian acquisition function using Thompson sampling.
    """

    def __init__(
        self,
        known_objective: Optional[Callable] = None,
        inequality_lambda: Optional[Mapping[Tag, TensorType]] = None,
        equality_lambda: Optional[Mapping[Tag, TensorType]] = None,
        batch_size: int = 1,
        penalty: Optional[TensorType] = None,
        epsilon: float = 0.01,
        search_space: Optional[SearchSpace] = None,
    ):
        """
        :param known_objective: Known objective function (if it is being treated as known) which behaves like a trajectory.
        :param inequality_lambda: Initial values for Lagrange multipliers of inequality constraints.
        :param equality_lambda: Initial values for Lagrange multipliers of equality
            constraints.
        :param batch_size: Batch size used when calling the sampled trajectories.
        :param penalty: Initial penalty. If None, it is set to a default value using the
            _get_initial_penalty method.
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
        self._known_objective = known_objective
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
        self._search_space = search_space
        self._augmented_lagrangian_fn = None
        self._iteration = 0
        self._best_valid_observation = None  # Kept to None until a valid observation is found.

    def __repr__(self) -> str:
        """"""
        return (
            f"ThompsonSamplingAugmentedLagrangian({self._search_space!r})"
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
        satisfied_mask = tf.constant(value=True, shape=datasets[OBJECTIVE].observations.shape)
        for tag, dataset in datasets.items():
            if tag.startswith(INEQUALITY_CONSTRAINT_PREFIX):
                constraint_vals = dataset.observations
                valid_constraint_vals = constraint_vals <= 0
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)
            elif tag.startswith(EQUALITY_CONSTRAINT_PREFIX):
                constraint_vals = dataset.observations
                valid_constraint_vals = tf.abs(constraint_vals) <= self._epsilon
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)

        if tf.reduce_sum(tf.cast(satisfied_mask, tf.int32)) != 0:
            objective_vals = datasets[OBJECTIVE].observations
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
            self._objective_trajectory_sampler = models[OBJECTIVE].trajectory_sampler()
            self._objective_trajectory = self._objective_trajectory_sampler.get_trajectory()

        self._inequality_constraint_trajectory_samplers = {}
        self._inequality_constraint_trajectories = {}
        for tag, model in models.items():
            if tag.startswith(INEQUALITY_CONSTRAINT_PREFIX):
                sampler = model.trajectory_sampler()
                trajectory = sampler.get_trajectory()
                self._inequality_constraint_trajectory_samplers[tag] = sampler
                self._inequality_constraint_trajectories[tag] = trajectory

        self._equality_constraint_trajectory_samplers = {}
        self._equality_constraint_trajectories = {}
        for tag, model in models.items():
            if tag.startswith(EQUALITY_CONSTRAINT_PREFIX):
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

        inequality_constraint_tags = [key for key in datasets.keys() if key.startswith(INEQUALITY_CONSTRAINT_PREFIX)]
        for tag in inequality_constraint_tags:
            constraint_satisfied = tf.squeeze(datasets[tag].observations, axis=-1) <= 0
            if all_satisfied is None:
                all_satisfied = constraint_satisfied
            else:
                all_satisfied = tf.logical_and(all_satisfied, constraint_satisfied)

            # Only consider invalid inequality constraints (i.e. > 0)
            constraint_squared = tf.square(tf.nn.relu(tf.squeeze(datasets[tag].observations, axis=-1)))
            if sum_squared is None:
                sum_squared = constraint_squared
            else:
                sum_squared += constraint_squared

        equality_constraint_tags = [key for key in datasets.keys() if key.startswith(EQUALITY_CONSTRAINT_PREFIX)]
        for tag in equality_constraint_tags:
            constraint_satisfied = tf.abs(tf.squeeze(datasets[tag].observations, axis=-1)) <= self._epsilon
            if all_satisfied is None:
                all_satisfied = constraint_satisfied
            else:
                all_satisfied = tf.logical_and(all_satisfied, constraint_satisfied)

            constraint_squared = tf.square(tf.squeeze(datasets[tag].observations, axis=-1))
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
                denominator = 2 * tfp.stats.percentile(datasets[OBJECTIVE].observations, 50.0, interpolation='midpoint')
            else:
                best_valid_objective = tf.math.reduce_min(datasets[OBJECTIVE].observations[all_satisfied])
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

        objective_dataset = datasets[OBJECTIVE]

        tf.debugging.assert_positive(
            len(objective_dataset),
            message="Lagrange multiplier updates are defined with respect to existing points in the"
                    " objective data, but the objective data is empty.",
        )

        # Find the best valid objective value seen so far
        satisfied_mask = tf.constant(value=True, shape=datasets[OBJECTIVE].observations.shape)
        for tag, dataset in datasets.items():
            if tag.startswith(INEQUALITY_CONSTRAINT_PREFIX):
                constraint_vals = dataset.observations
                valid_constraint_vals = constraint_vals <= 0
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)
            elif tag.startswith(EQUALITY_CONSTRAINT_PREFIX):
                constraint_vals = dataset.observations
                valid_constraint_vals = tf.abs(constraint_vals) <= self._epsilon
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)

        if tf.reduce_sum(tf.cast(satisfied_mask, tf.int32)) != 0:
            objective_vals = datasets[OBJECTIVE].observations
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
        best_observed_augmented_lagrangian_idx = int(tf.squeeze(tf.argmin(augmented_lagrangian_values_at_observed_points), axis=-1))

        # Update Lagrange multipliers
        self._update_lagrange_multipliers(datasets, best_observed_augmented_lagrangian_idx)

        # Update penalty
        self._update_penalty(datasets, best_observed_augmented_lagrangian_idx)

        self._augmented_lagrangian_fn = self._negative_augmented_lagrangian

        return self._augmented_lagrangian_fn

    def _update_lagrange_multipliers(self,
                                     datasets: Mapping[Tag, Dataset],
                                     best_observed_augmented_lagrangian_idx: int):
        """
        :param datasets: Datasets containing previously observed points
        :param best_observed_augmented_lagrangian_idx: Index of observed point in dataset which minimises the Augmented
                                                       Lagrangian.

        Updates Lagrange multipliers given previously observed data in accordance with Picheny et al.'s paper.
        """
     
        # Update Lagrange multipliers for inequality constraints
        inequality_constraint_tags = [key for key in datasets.keys() if key.startswith(INEQUALITY_CONSTRAINT_PREFIX)]
        for tag in inequality_constraint_tags:
            inequality_constraint_val = datasets[tag].observations[best_observed_augmented_lagrangian_idx]
            tf.debugging.assert_shapes([(inequality_constraint_val[..., None, None], (1, 1, 1))])
            slack_val = tf.squeeze(self._obtain_slacks(inequality_constraint_val[..., None, None], self._inequality_lambda[tag], self._penalty), [-1, -2])
            assert (slack_val.shape == self._inequality_lambda[tag].shape)
            assert (slack_val.shape == inequality_constraint_val.shape)
            updated_multiplier = self._inequality_lambda[tag] + (1 / self._penalty) * (inequality_constraint_val + slack_val)
            self._inequality_lambda[tag] = updated_multiplier

        # Update Lagrange multipliers for equality constraints
        equality_constraint_tags = [key for key in datasets.keys() if key.startswith(EQUALITY_CONSTRAINT_PREFIX)]
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
        inequality_constraint_tags = [key for key in datasets.keys() if key.startswith(INEQUALITY_CONSTRAINT_PREFIX)]
        for tag in inequality_constraint_tags:
            inequality_constraint_val = datasets[tag].observations[best_observed_augmented_lagrangian_idx]
            inequality_constraints_satisfied = inequality_constraints_satisfied and inequality_constraint_val <= 0

        # See if equality constraints are satisfied
        equality_constraints_satisfied = True
        equality_constraint_tags = [key for key in datasets.keys() if key.startswith(EQUALITY_CONSTRAINT_PREFIX)]
        for tag in equality_constraint_tags:
            equality_constraint_val = datasets[tag].observations[best_observed_augmented_lagrangian_idx]
            equality_constraints_satisfied = equality_constraints_satisfied and tf.abs(equality_constraint_val) <= self._epsilon

        if not (equality_constraints_satisfied and inequality_constraints_satisfied):
            self._penalty = self._penalty / tf.pow(tf.constant(2.0, dtype=tf.float64), min(self._batch_size, 10))
            print(f"Not Satisfied. Updated Penalty: {self._penalty}")
        else:
            print("Satisfied")

    def _deterministic_augmented_lagrangian(self,
                                            datasets: Mapping[Tag, Dataset]) -> TensorType:
        """
        :param datasets: Datasets containing previously observed points.
        :returns: Augmented Lagrangian values at observed points in datasets.
        """

        objective_vals = datasets[OBJECTIVE].observations[..., None]  # [N, 1, 1]

        # Calculate equality constraint values, and scale them with Lagrange multipliers
        # and penalty parameter
        sum_equality_lambda_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        sum_equality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        equality_constraint_tags = [key for key in datasets.keys() if key.startswith(EQUALITY_CONSTRAINT_PREFIX)]
        for tag in equality_constraint_tags:
            equality_constraint_vals = datasets[tag].observations[..., None]  # [N, 1, 1]
            equality_lambda_scaled = self._equality_lambda[tag] * equality_constraint_vals  # [N, 1, 1]
            equality_penalty_scaled = (1 / (2 * self._penalty)) * tf.square(equality_constraint_vals)
            assert (equality_penalty_scaled.shape == sum_equality_penalty_scaled.shape)
            assert (equality_lambda_scaled.shape == sum_equality_lambda_scaled.shape)
            sum_equality_penalty_scaled += equality_penalty_scaled
            sum_equality_lambda_scaled += equality_lambda_scaled

        # Calculate inequality constraint values, and scale them with Lagrange multipliers
        # and penalty parameter
        sum_inequality_lambda_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        sum_inequality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        inequality_constraint_tags = [key for key in datasets.keys() if key.startswith(INEQUALITY_CONSTRAINT_PREFIX)]
        for tag in inequality_constraint_tags:
            inequality_constraint_vals = datasets[tag].observations[..., None]  # [N, 1, 1]
            slack_vals = self._obtain_slacks(inequality_constraint_vals, self._inequality_lambda[tag],
                                             self._penalty)  # [N, 1, 1]
            assert (slack_vals.shape == inequality_constraint_vals.shape)
            inequality_plus_slack = inequality_constraint_vals + slack_vals
            inequality_lambda_scaled = self._inequality_lambda[tag] * inequality_plus_slack  # [N, 1, 1]
            inequality_penalty_scaled = (1 / (2 * self._penalty)) * tf.square(inequality_plus_slack)
            assert (inequality_penalty_scaled.shape == sum_inequality_penalty_scaled.shape)
            assert (inequality_lambda_scaled.shape == sum_inequality_lambda_scaled.shape)
            sum_inequality_penalty_scaled += inequality_penalty_scaled
            sum_inequality_lambda_scaled += inequality_lambda_scaled

        # Return augmented Lagrangian
        al = tf.squeeze(objective_vals + sum_equality_lambda_scaled + sum_inequality_lambda_scaled + sum_equality_penalty_scaled + sum_inequality_penalty_scaled, -1)
        return al

    def _negative_augmented_lagrangian(
            self,
            x: tf.Tensor) -> tf.Tensor:
        """
        Form negative of augmented Lagrangian (since acquisition function optimiser *maximises* the returned acquisition
        function)

        :param x: Array of points at which to evaluate the augmented Lagrangian of shape [N, B, D] (middle axis is for
                  batching when calling the sampled trajectories)
        :return: Values of negative augmented Lagrangian at given x values of shape [N, B]
        """
        objective_vals = self._objective_trajectory(x)

        # Calculate equality constraint values using the sampled trajectory, and scale them with Lagrange multipliers
        # and penalty parameter
        sum_equality_lambda_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        sum_equality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
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
        for tag, inequality_constraint_trajectory in self._inequality_constraint_trajectories.items():
            inequality_constraint_vals = inequality_constraint_trajectory(x)  # [N, B, 1]
            slack_vals = self._obtain_slacks(inequality_constraint_vals, self._inequality_lambda[tag],
                                             self._penalty)  # [N, B, 1]
            assert (slack_vals.shape == inequality_constraint_vals.shape)
            inequality_plus_slack = inequality_constraint_vals + slack_vals
            inequality_lambda_scaled = self._inequality_lambda[tag] * inequality_plus_slack  # [N, B, 1]
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
        slack_vals = - (inequality_lambda * penalty) - inequality_constraint_vals
        slack_vals_non_neg = tf.nn.relu(slack_vals)
        tf.debugging.assert_shapes([(slack_vals_non_neg, (..., 1))])
        return slack_vals_non_neg
