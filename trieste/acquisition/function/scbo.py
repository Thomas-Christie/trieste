from __future__ import annotations
from typing import Mapping, Optional
import tensorflow as tf

from ...data import Dataset
from ...models.interfaces import HasTrajectorySampler
from ...types import Tag
from ..interface import (
    AcquisitionFunction,
    ProbabilisticModelType,
    VectorizedAcquisitionFunctionBuilder
)
from ...observer import OBJECTIVE, INEQUALITY_CONSTRAINT_PREFIX


class SCBO(VectorizedAcquisitionFunctionBuilder[HasTrajectorySampler]):
    """
    Builder for Eriksson et al.'s Scalable Constrained Bayesian Optimisation (SCBO)
    acquisition function descibed in https://arxiv.org/pdf/2002.08526.pdf. This draws
    samples from the objective and constraint models, and if any sampled points
    satisfy all constraints, then will return the (negative) objective function value for the
    points at which all constraints are satisfied, and -infinity for points at which
    not all constraints are satisfied. On the other hand, if no points satisfy all
    constraints, then it will return the (negative) sum of constraint violations for
    each point. Negations take place as the acquisition function optimiser *maximises* the
    returned acquisition function.
    """

    def __init__(
        self,
    ):
        self._inequality_constraint_trajectories = None
        self._inequality_constraint_trajectory_samplers = None
        self._objective_trajectory = None
        self._objective_trajectory_sampler = None
        self._iteration = 0
        self._best_valid_observation = None 

    def __repr__(self) -> str:
        return ("SCBO")

    def prepare_acquisition_function(
        self,
        models: Mapping[Tag, HasTrajectorySampler],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param models: The models over each tag.
        :param datasets: The data from the observer.
        :return: The SCBO acquisition function, using Thompson sampling to approximate all unknown functions.
        :raise KeyError: If `objective_tag` is not found in ``datasets`` and ``models``.
        :raise tf.errors.InvalidArgumentError: If the objective data is empty.
        """
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

        self._scbo_fn = self._scbo
        return self._scbo_fn

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
        :return: The SCBO acquisition function.
        """
        self._iteration += 1
        print(f"Iteration: {self._iteration}")

        # Find the best valid objective value seen so far
        satisfied_mask = tf.constant(value=True, shape=datasets[OBJECTIVE].observations.shape)
        for tag, dataset in datasets.items():
            if tag.startswith(INEQUALITY_CONSTRAINT_PREFIX):
                constraint_vals = dataset.observations
                valid_constraint_vals = constraint_vals <= 0
                satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)

        if tf.reduce_sum(tf.cast(satisfied_mask, tf.int32)) != 0:
            objective_vals = datasets[OBJECTIVE].observations
            valid_y = tf.boolean_mask(objective_vals, satisfied_mask)
            self._best_valid_observation = tf.math.reduce_min(valid_y)
        print(f"Best Valid Observation: {self._best_valid_observation}")

        # Update sampled trajectories
        self._objective_trajectory = self._objective_trajectory_sampler.update_trajectory(self._objective_trajectory)
        for tag, sampler in self._inequality_constraint_trajectory_samplers.items():
            old_trajectory = self._inequality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            self._inequality_constraint_trajectories[tag] = updated_trajectory

        self._scbo_fn = self._scbo
        return self._scbo_fn

    def _scbo(self,
              x: tf.Tensor) -> tf.Tensor:
        """
        Form SCBO acquisition function described in Eriksson et
        al. 2019 (https://arxiv.org/pdf/1910.01739.pdf).

        :param x: Array of points at which to evaluate SCBO of shape [N, B, D] 
        :return: Values of SCBO at given x values of shape [N, B]
        """
        objective_vals = tf.squeeze(self._objective_trajectory(x), axis=-1)  # [N, B]

        satisfied_mask = tf.constant(value=True, shape=(x.shape[0], x.shape[1])) # [N, B]
        constraint_violation_sum = tf.zeros(shape=(x.shape[0], x.shape[1]), dtype=tf.float64)  # [N, B]
        for _, inequality_constraint_trajectory in self._inequality_constraint_trajectories.items():
            inequality_constraint_vals = tf.squeeze(inequality_constraint_trajectory(x), axis=-1)  # [N, B]
            valid_constraint_vals = inequality_constraint_vals <= 0
            constraint_violation_sum += tf.nn.relu(inequality_constraint_vals)
            satisfied_mask = tf.logical_and(satisfied_mask, valid_constraint_vals)

        int_satisfied_mask = tf.cast(satisfied_mask, tf.int32)
        batch_satisfaction = tf.reduce_sum(int_satisfied_mask, axis=0, keepdims=True) # [1, B]
        batch_satisfaction = tf.repeat((batch_satisfaction > 0), repeats=x.shape[0], axis=0)  # [N, B]
        acq_vals = -1.0 * objective_vals
        # For batches where at least one point satisfies all constraints, for each point
        # in the batch return the value of the objective function at that point if all
        # constraints are satisfied, otherwise return -inf (the best point in this batch
        # is the one with the best objective value that satisfies all constraints)
        acq_vals = tf.where(tf.logical_and(batch_satisfaction, tf.logical_not(satisfied_mask)), tf.float64.min, acq_vals)  
        # For batches where no point satisfies all constraints, for each point in the
        # batch return the value of the negative sum of constraint violations at that
        # point (the best point in this batch is the one with the smallest amount of
        # constraint violation. Note that we're negating as the acquisition function
        # optimiser *maximises* the returned acquisition function).
        acq_vals = tf.where(tf.logical_not(batch_satisfaction), -constraint_violation_sum, acq_vals)
        return acq_vals
