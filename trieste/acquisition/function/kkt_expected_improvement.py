from __future__ import annotations

import copy
import os
from typing import Mapping, Optional, cast
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset
from ...models.interfaces import HasTrajectorySampler
from ...space import SearchSpace
from ...types import Tag
from ..interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    ProbabilisticModelType
)

# Combine KKT conditions with Expected Improvement for constrained optimisation
# as done in https://pure.uvt.nl/ws/portalfiles/portal/63795955/2022_022.pdf
class KKTExpectedImprovement(AcquisitionFunctionBuilder[ProbabilisticModelType]):
    """
    Builder for an augmented Lagrangian acquisition function using Thompson sampling.
    """

    def __init__(
        self,
        objective_tag: Tag,
        inequality_constraint_prefix: Optional[Tag] = None,
        equality_constraint_prefix: Optional[Tag] = None,
        epsilon = 0.001,
        search_space: Optional[SearchSpace] = None,
        plot: bool = False
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param inequality_constraint_prefix: Prefix of tags for inequality constraint data/models/Lagrange multipliers.
        :param equality_constraint_prefix: Prefix for tags for equality constraint data/models/Lagrange multipliers.
        :param epsilon: Bound within which constraints are considered to be satisfied.
        :param search_space: The global search space over which the optimisation is defined. This is
            only used to determine explicit constraints.
        :param plot: Whether to plot the modelled functions at each iteration of BayesOpt.
        """
        self._equality_constraint_models = {}
        self._inequality_constraint_models = {}
        self._objective_model = {}
        self._objective_tag = objective_tag
        self._inequality_constraint_prefix = inequality_constraint_prefix
        self._equality_constraint_prefix = equality_constraint_prefix
        self._epsilon = epsilon
        self._search_space = search_space
        self._augmented_lagrangian_fn = None
        self._plot = plot
        self._iteration = 0
        # TODO: Think of better way to do below
        self._best_valid_observation = 1000 # For when we haven't yet found any valid points
        self._kkt_expected_improvement_fn = None

    def __repr__(self) -> str:
        """"""
        return (
            f"KKTExpectedImprovement({self._objective_tag!r}, {self._inequality_constraint_prefix!r},"
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
        print(f"Iteration: {self._iteration}")

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

        for tag, model in models.items():
            if tag.startswith(self._objective_tag):
                self._objective_model = copy.deepcopy(model)
            elif (self._equality_constraint_prefix is not None) and (tag.startswith(self._equality_constraint_prefix)):
                self._equality_constraint_models[tag] = copy.deepcopy(model)
            elif (self._inequality_constraint_prefix is not None) and (tag.startswith(self._inequality_constraint_prefix)):
                self._inequality_constraint_models[tag] = copy.deepcopy(model)

        self._kkt_expected_improvement_fn = self._efficient_kkt_expected_improvement

        return self._kkt_expected_improvement_fn

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

        # Last point in dataset is most recent estimate of optimal x value
        most_recent_observation = datasets[self._objective_tag].observations[-1]
        most_recent_query_point = datasets[self._objective_tag].query_points[-1]
        print(f"Most Recent Query Point: {most_recent_query_point} Most Recent Observation: {most_recent_observation}")

        # Check to see if most recent observation is valid and better than previously seen observations
        all_constraints_satisfied = True
        for tag, dataset in datasets.items():
            if (self._inequality_constraint_prefix is not None) and (tag.startswith(self._inequality_constraint_prefix)):
                constraint_val = dataset.observations[-1]
                constraint_satisfied = constraint_val <= 0
                all_constraints_satisfied = all_constraints_satisfied and constraint_satisfied
            elif (self._equality_constraint_prefix is not None) and (tag.startswith(self._equality_constraint_prefix)):
                constraint_val = dataset.observations[-1]
                constraint_satisfied = tf.abs(constraint_val) <= self._epsilon
                all_constraints_satisfied = all_constraints_satisfied and constraint_satisfied

        if all_constraints_satisfied:
            self._best_valid_observation = min(self._best_valid_observation, most_recent_observation)

        # Plot Models
        self._plot_models(most_recent_query_point)

        # Update models
        # TODO: Currently using deepcopy for producing consistent plots, may remove in future
        for tag, model in models.items():
            if tag.startswith(self._objective_tag):
                self._objective_model = copy.deepcopy(model)
            elif (self._equality_constraint_prefix is not None) and (tag.startswith(self._equality_constraint_prefix)):
                self._equality_constraint_models[tag] = copy.deepcopy(model)
            elif (self._inequality_constraint_prefix is not None) and (tag.startswith(self._inequality_constraint_prefix)):
                self._inequality_constraint_models[tag] = copy.deepcopy(model)

        self._kkt_expected_improvement_fn = self._efficient_kkt_expected_improvement

        return self._kkt_expected_improvement_fn

    def _kkt_expected_improvement(self,
                                  x: tf.Tensor) -> tf.Tensor:
        """
        Form KKT expected improvement from objective and constraints
        :param x: Array of points at which to evaluate the kkt expected improvement at of shape [N, 1, M]
        """
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        mean, variance = self._objective_model.predict(tf.squeeze(x, -2))
        normal = tfp.distributions.Normal(mean, tf.sqrt(variance))

        # TODO: Double check expected improvement definition
        expected_improvement = (self._best_valid_observation - mean) * normal.cdf(self._best_valid_observation) + variance * normal.prob(self._best_valid_observation)
        num_x_vals = x.shape[0]
        cosine_similarities = []
        print('Starting Optimisation')
        for i in range(num_x_vals):
            inequality_grad_dict = {}
            equality_grad_dict = {}
            x_val = x[i]
            tf.debugging.assert_shapes([(x_val, [1, 2])]) # TODO: Hard-coded 2

            # Calculate objective gradient
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_val)
                objective_pred_mean, objective_pred_var = self._objective_model.predict(x_val)
            mle_objective_grad = tf.transpose(tape.gradient(objective_pred_mean, x_val))
            tf.debugging.assert_shapes([(mle_objective_grad, [2, 1])]) # TODO: Remove hard-coding of dimension

            # Calculate inequality constraint gradients
            for tag, model in self._inequality_constraint_models.items():
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(x_val)
                    constraint_pred_mean, constraint_pred_var = model.predict(x_val)
                    binding = tf.abs(constraint_pred_mean) / tf.sqrt(constraint_pred_var) <= 1.2816
                # Only consider binding constraints
                if binding:
                    constraint_grad = tape.gradient(constraint_pred_mean, x_val)
                    inequality_grad_dict[tag] = constraint_grad

            # Calculate equality constraint gradients
            for tag, model in self._equality_constraint_models.items():
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(x_val)
                    constraint_pred_mean, constraint_pred_var = model.predict(x_val)
                    # TODO: Change hard-coding of constant below
                    binding = tf.abs(constraint_pred_mean) / tf.sqrt(constraint_pred_var) <= 1.2816
                if binding:
                    constraint_grad = tape.gradient(constraint_pred_mean, x_val)
                    equality_grad_dict[tag] = constraint_grad

            # Construct gradient matrix
            gradient_matrix = None
            for tag, gradient in equality_grad_dict.items():
                if gradient_matrix is None:
                    gradient_matrix = tf.transpose(gradient)
                else:
                    gradient_matrix = tf.concat((gradient_matrix, tf.transpose(gradient)), axis=1)

            for tag, gradient in inequality_grad_dict.items():
                if gradient_matrix is None:
                    gradient_matrix = tf.transpose(gradient)
                else:
                    gradient_matrix = tf.concat((gradient_matrix, tf.transpose(gradient)), axis=1)

            if gradient_matrix is None:
                # No constraints are considered binding, so unlikely to be optimal point
                cosine_similarities.append(0)
            else:
                # TODO: Should we enforce non-negativity of inequality constraint Lagrange multipliers?
                lagrange_multipliers = tf.linalg.inv(tf.matmul(gradient_matrix, gradient_matrix, transpose_a=True)) @ tf.transpose(gradient_matrix) @ (-mle_objective_grad)
                lagrange_multipliers = tf.nn.relu(lagrange_multipliers)
                ls_objective_grad = - tf.matmul(gradient_matrix, lagrange_multipliers)
                normalized_mle_objective_grad = tf.math.l2_normalize(mle_objective_grad, axis=0)
                normalized_ls_objective_grad = tf.math.l2_normalize(ls_objective_grad, axis=0)
                cosine_similarity = tf.reduce_sum(tf.multiply(normalized_mle_objective_grad, normalized_ls_objective_grad))
                cosine_similarities.append(cosine_similarity)

        cosine_similarities = tf.convert_to_tensor(cosine_similarities, dtype=tf.float64)[..., None]
        assert (expected_improvement.shape == cosine_similarities.shape)
        return tf.multiply(cosine_similarities, expected_improvement)

    def _efficient_kkt_expected_improvement(self,
                                  x: tf.Tensor) -> tf.Tensor:
        """
        Form KKT expected improvement from objective and constraints
        :param x: Array of points at which to evaluate the kkt expected improvement at of shape [N, 1, M]
        """
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        x = tf.squeeze(x, -2)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            objective_mean, objective_var = self._objective_model.predict(x)
        mle_objective_grad = tape.gradient(objective_mean, x)
        tf.debugging.assert_shapes([(mle_objective_grad, [..., 2])])
        normal = tfp.distributions.Normal(objective_mean, tf.sqrt(objective_var))

        # TODO: Double check expected improvement definition
        expected_improvement = (self._best_valid_observation - objective_mean) * normal.cdf(self._best_valid_observation) + objective_var * normal.prob(self._best_valid_observation)
        num_x_vals = x.shape[0]
        cosine_similarities = []

        inequality_constraint_grad_dict = {}
        inequality_constraint_bind_dict = {}
        for tag, model in self._inequality_constraint_models.items():
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                constraint_pred_mean, constraint_pred_var = model.predict(x)
            binding = tf.abs(constraint_pred_mean) / tf.sqrt(constraint_pred_var) <= 1.2816
            constraint_grad = tape.gradient(constraint_pred_mean, x)
            inequality_constraint_grad_dict[tag] = constraint_grad
            inequality_constraint_bind_dict[tag] = binding

        for i in range(num_x_vals):
            point_inequality_grad_dict = {}
            point_mle_objective_grad = mle_objective_grad[i][..., None]

            # Calculate inequality constraint gradients
            for tag, _ in self._inequality_constraint_models.items():
                binding = inequality_constraint_bind_dict[tag][i]
                # Only consider binding constraints
                if binding:
                    point_constraint_grad = inequality_constraint_grad_dict[tag][i]
                    point_inequality_grad_dict[tag] = point_constraint_grad

            # Construct gradient matrix
            gradient_matrix = None

            for tag, gradient in point_inequality_grad_dict.items():
                if gradient_matrix is None:
                    gradient_matrix = gradient[..., None]
                else:
                    gradient_matrix = tf.concat((gradient_matrix, gradient[..., None]), axis=1)

            if gradient_matrix is None:
                # No constraints are considered binding, so unlikely to be optimal point
                cosine_similarities.append(0)
            else:
                # TODO: Should we enforce non-negativity of inequality constraint Lagrange multipliers? Currently we are
                lagrange_multipliers = tf.linalg.inv(tf.matmul(gradient_matrix, gradient_matrix, transpose_a=True)) @ tf.transpose(gradient_matrix) @ (-point_mle_objective_grad)
                lagrange_multipliers = tf.nn.relu(lagrange_multipliers)
                ls_objective_grad = - tf.matmul(gradient_matrix, lagrange_multipliers)
                normalized_mle_objective_grad = tf.math.l2_normalize(point_mle_objective_grad, axis=0)
                normalized_ls_objective_grad = tf.math.l2_normalize(ls_objective_grad, axis=0)
                cosine_similarity = tf.reduce_sum(tf.multiply(normalized_mle_objective_grad, normalized_ls_objective_grad))
                cosine_similarities.append(cosine_similarity)

        cosine_similarities = tf.convert_to_tensor(cosine_similarities, dtype=tf.float64)[..., None]
        assert (expected_improvement.shape == cosine_similarities.shape)
        return tf.multiply(cosine_similarities, expected_improvement)

    def _plot_models(self, prev_query_point):
        """
        Plot visualisation of surrogate models for objective and inequality constraints, as well as the augmented
        Lagrangian.
        """
        print(f"Prev Query: {prev_query_point}")
        x_list = tf.linspace(0, 1, 100)
        y_list = tf.linspace(0, 1, 100)
        xs, ys = tf.meshgrid(x_list, y_list)
        coordinates = tf.expand_dims(tf.stack((tf.reshape(xs, [-1]), tf.reshape(ys, [-1])), axis=1), -2)
        objective_pred, _ = self._objective_model.predict(coordinates)
        kkt_ei_pred = self._efficient_kkt_expected_improvement(coordinates)
        fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(15, 7))
        objective_plot = ax1[0].contourf(xs, ys, tf.reshape(objective_pred, [y_list.shape[0], x_list.shape[0]]), levels=500)
        ax1[0].scatter(prev_query_point[0], prev_query_point[1], marker="x")
        fig.colorbar(objective_plot)
        ax1[0].set_xlabel("OBJECTIVE")

        constraint_one_pred, _ = self._inequality_constraint_models["INEQUALITY_CONSTRAINT_ONE"].predict(coordinates)
        constraint_one_plot = ax1[1].contourf(xs, ys, tf.reshape(constraint_one_pred, [y_list.shape[0], x_list.shape[0]]),
                                              levels=500)
        ax1[1].scatter(prev_query_point[0], prev_query_point[1], marker="x")
        fig.colorbar(constraint_one_plot)
        ax1[1].set_xlabel("INEQUALITY CONSTRAINT ONE")

        constraint_two_pred, _ = self._inequality_constraint_models["INEQUALITY_CONSTRAINT_TWO"].predict(coordinates)
        constraint_two_plot = ax1[2].contourf(xs, ys, tf.reshape(constraint_two_pred, [y_list.shape[0], x_list.shape[0]]),
                                              levels=500)
        ax1[2].scatter(prev_query_point[0], prev_query_point[1], marker="x")
        fig.colorbar(constraint_two_plot)
        ax1[2].set_xlabel("INEQUALITY CONSTRAINT TWO")

        kkt_plot = ax2[0].contourf(xs, ys, tf.reshape(kkt_ei_pred, [y_list.shape[0], x_list.shape[0]]), levels= 500, extend="both")
        ax2[0].scatter(prev_query_point[0], prev_query_point[1], marker="x")
        fig.colorbar(kkt_plot)
        ax2[0].set_xlabel("KKT Expected Improvement)")
        ax2[1].text(0.5, 0.5, f"Iteration: {self._iteration - 1} \n  Query: {prev_query_point} \n Best Valid Observation: {self._best_valid_observation}",
                    horizontalalignment='center', verticalalignment='center')
        ax2[1].axis("off")
        ax2[2].axis("off")
        plt.tight_layout()
        print(f"Curr Dir: {os.getcwd()}")
        with open(f"../results/03-03-23/visualisation/run_two/iter_{self._iteration - 1}.png", "wb") as fp:
            plt.savefig(fp)
        plt.show()
