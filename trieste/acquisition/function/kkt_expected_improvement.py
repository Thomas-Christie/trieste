from __future__ import annotations

import copy
from functools import partial
from typing import Mapping, Optional, cast
from scipy.stats import norm
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
    KKTAcquisitionFunctionBuilder,
    ProbabilisticModelType
)

# Combine KKT conditions with Expected Improvement for constrained optimisation
# as done in https://pure.uvt.nl/ws/portalfiles/portal/63795955/2022_022.pdf
class KKTExpectedImprovement(KKTAcquisitionFunctionBuilder[ProbabilisticModelType]):
    """
    Builder for an acquisition function combining the KKT conditions and Expected Improvement.
    """

    def __init__(self, objective_tag: Tag, inequality_constraint_prefix: Optional[Tag] = None,
                 equality_constraint_prefix: Optional[Tag] = None, epsilon=0.001,
                 search_space: Optional[SearchSpace] = None, feasible_region_only : bool = False, plot: bool = False):
        """
        :param objective_tag: The tag for the objective data and model.
        :param inequality_constraint_prefix: Prefix of tags for inequality constraint data/models/Lagrange multipliers.
        :param equality_constraint_prefix: Prefix for tags for equality constraint data/models/Lagrange multipliers.
        :param epsilon: Bound within which equality constraints are considered to be satisfied.
        :param search_space: The global search space over which the optimisation is defined. This is
            only used to determine explicit constraints.
        :param plot: Whether to plot the modelled functions at each iteration of BayesOpt.
        """
        super().__init__()
        self._equality_constraint_models = {}
        self._inequality_constraint_models = {}
        self._objective_model = None
        self._objective_tag = objective_tag
        self._inequality_constraint_prefix = inequality_constraint_prefix
        self._equality_constraint_prefix = equality_constraint_prefix
        self._epsilon = epsilon
        self._search_space = search_space
        self._augmented_lagrangian_fn = None
        self._feasible_region_only = feasible_region_only
        self._plot = plot
        self._iteration = 0
        # TODO: Think of better way to do below
        self.best_valid_observation = 10  # For when we haven't yet found any valid points
        self._kkt_expected_improvement_fn = None
        self._alpha = None

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
        alpha: float = 0.20,
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

        self._alpha = alpha

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
            self.best_valid_observation = tf.math.reduce_min(valid_y)

        for tag, model in models.items():
            if tag.startswith(self._objective_tag):
                self._objective_model = copy.deepcopy(model)
            elif (self._equality_constraint_prefix is not None) and (tag.startswith(self._equality_constraint_prefix)):
                self._equality_constraint_models[tag] = copy.deepcopy(model)
            elif (self._inequality_constraint_prefix is not None) and (tag.startswith(self._inequality_constraint_prefix)):
                self._inequality_constraint_models[tag] = copy.deepcopy(model)

        # Plot Models
        if self._plot:
            self._plot_models([0.0, 0.0])

        self._kkt_expected_improvement_fn = partial(self._efficient_kkt_expected_improvement, False)

        return self._kkt_expected_improvement_fn

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
        alpha: float = 0.20
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

        self._alpha = alpha

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
            self.best_valid_observation = min(self.best_valid_observation, most_recent_observation)

        # Plot Models
        if self._plot:
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

        opt_x = tf.Variable([[0.1954, 0.4044]], dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(opt_x)
            ineq_one_mean, ineq_one_var = models["INEQUALITY_CONSTRAINT_ONE"].predict(opt_x)
        grad = tape.gradient(ineq_one_mean, opt_x)
        binding = tf.abs(ineq_one_mean) / tf.sqrt(ineq_one_var) <= 1.2816
        relaxed_binding = tf.abs(ineq_one_mean) / tf.sqrt(ineq_one_var) <= 2 * 1.2816
        very_relaxed_binding = tf.abs(ineq_one_mean) / tf.sqrt(ineq_one_var) <= 10 * 1.2816
        print(f"Inequality Constraint One Gradient @ Optimum: {grad}")
        print(f"Inequality Constraint One Prediction @ Optimum: {ineq_one_mean} Var: {ineq_one_var}, Std: {tf.sqrt(ineq_one_var)}")
        print(f"Considered Binding: {binding} (Val: {tf.abs(ineq_one_mean) / tf.sqrt(ineq_one_var)}) Relaxed Binding: {relaxed_binding} Very Relaxed Binding: {very_relaxed_binding}")
        print(f"Best Valid Observation: {self.best_valid_observation}")
        self._kkt_expected_improvement_fn = partial(self._efficient_kkt_expected_improvement, False)

        return self._kkt_expected_improvement_fn

    def _efficient_kkt_expected_improvement(self,
                                            return_separate_components: bool,
                                            x: tf.Tensor) -> tf.Tensor | (tf.Tensor, tf.Tensor):
        """
        Form KKT expected improvement from objective and constraints
        :param: return_separate_components: Whether to return cosine similarity and expected improvement separately (
                sometimes used for plotting purposes).
        :param x: Array of points at which to evaluate the kkt expected improvement at of shape [N, 1, M]
        """
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        x = tf.squeeze(x, -2)  # [N, D]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            objective_mean, objective_var = self._objective_model.predict(x)
        mle_objective_grad = tape.gradient(objective_mean, x)
        tf.debugging.assert_shapes([(mle_objective_grad, [..., 2])])
        normal = tfp.distributions.Normal(objective_mean, tf.sqrt(objective_var))

        # TODO: Double check expected improvement definition
        expected_improvement = (self.best_valid_observation - objective_mean) * normal.cdf(self.best_valid_observation) + objective_var * normal.prob(self.best_valid_observation)
        num_x_vals = x.shape[0]
        cosine_similarities = []

        inequality_constraint_grad_dict = {}
        inequality_constraint_bind_dict = {}
        inequality_constraints_satisfied = tf.constant(True, shape=(x.shape[0], 1))  # [N, 1]
        for tag, model in self._inequality_constraint_models.items():
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                constraint_pred_mean, constraint_pred_var = model.predict(x)
            binding = tf.abs(constraint_pred_mean) / tf.sqrt(constraint_pred_var) <= norm.ppf(1 - self._alpha/2)
            inequality_constraints_satisfied = tf.logical_and(inequality_constraints_satisfied, constraint_pred_mean <= 0)
            constraint_grad = tape.gradient(constraint_pred_mean, x)
            inequality_constraint_grad_dict[tag] = constraint_grad
            inequality_constraint_bind_dict[tag] = binding

        equality_constraint_grad_dict = {}
        equality_constraint_bind_dict = {}
        all_equality_constraints_binding = tf.constant(True, shape=(x.shape[0], 1))
        for tag, model in self._equality_constraint_models.items():
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                constraint_pred_mean, constraint_pred_var = model.predict(x)
            binding = tf.abs(constraint_pred_mean) / tf.sqrt(constraint_pred_var) <= norm.ppf(1 - self._alpha/2)
            all_equality_constraints_binding = tf.logical_and(all_equality_constraints_binding, binding)
            constraint_grad = tape.gradient(constraint_pred_mean, x)
            equality_constraint_grad_dict[tag] = constraint_grad
            equality_constraint_bind_dict[tag] = binding

        for i in range(num_x_vals):
            point_inequality_grad_dict = {}
            point_equality_grad_dict = {}
            point_mle_objective_grad = mle_objective_grad[i][..., None]  # [D, 1]

            # Calculate inequality constraint gradients
            for tag, _ in self._inequality_constraint_models.items():
                binding = inequality_constraint_bind_dict[tag][i]
                # Only consider binding constraints
                if binding:
                    point_constraint_grad = inequality_constraint_grad_dict[tag][i]
                    point_inequality_grad_dict[tag] = point_constraint_grad

            # Calculate equality constraint gradients
            for tag, _ in self._equality_constraint_models.items():
                binding = equality_constraint_bind_dict[tag][i]
                # Only consider binding constraints
                if binding:
                    point_constraint_grad = equality_constraint_grad_dict[tag][i]
                    point_equality_grad_dict[tag] = point_constraint_grad

            # Construct gradient matrix
            gradient_matrix = None

            for tag, gradient in point_inequality_grad_dict.items():
                if gradient_matrix is None:
                    gradient_matrix = gradient[..., None]
                else:
                    gradient_matrix = tf.concat((gradient_matrix, gradient[..., None]), axis=1)

            for tag, gradient in point_equality_grad_dict.items():
                if gradient_matrix is None:
                    gradient_matrix = gradient[..., None]
                else:
                    gradient_matrix = tf.concat((gradient_matrix, gradient[..., None]), axis=1)

            num_binding_inequality_constraints = len(point_inequality_grad_dict)
            if gradient_matrix is None:
                # No constraints are considered binding, so unlikely to be optimal point
                cosine_similarities.append(0)
            else:
                lagrange_multipliers = tf.linalg.lstsq(gradient_matrix, -point_mle_objective_grad)
                # lagrange_multipliers = tf.linalg.inv(tf.matmul(gradient_matrix, gradient_matrix, transpose_a=True)) @ tf.transpose(gradient_matrix) @ (-point_mle_objective_grad)
                if num_binding_inequality_constraints > 0:
                    # Enforce non-negativity of inequality constraint Lagrange multipliers
                    inequality_indices = tf.Variable([[i] for i in range(num_binding_inequality_constraints)])
                    lagrange_multipliers = tf.tensor_scatter_nd_update(lagrange_multipliers, inequality_indices, tf.nn.relu(lagrange_multipliers)[:num_binding_inequality_constraints])
                ls_objective_grad = - tf.matmul(gradient_matrix, lagrange_multipliers)
                normalized_mle_objective_grad = tf.math.l2_normalize(point_mle_objective_grad, axis=0)
                normalized_ls_objective_grad = tf.math.l2_normalize(ls_objective_grad, axis=0)
                cosine_similarity = tf.reduce_sum(tf.multiply(normalized_mle_objective_grad, normalized_ls_objective_grad))
                cosine_similarities.append(cosine_similarity)

        cosine_similarities = tf.convert_to_tensor(cosine_similarities, dtype=tf.float64)[..., None]
        assert (expected_improvement.shape == cosine_similarities.shape)
        assert (all_equality_constraints_binding.shape == expected_improvement.shape)
        assert (inequality_constraints_satisfied.shape == expected_improvement.shape)
        all_equality_constraints_binding = tf.cast(all_equality_constraints_binding, tf.float64)
        inequality_constraints_satisfied = tf.cast(inequality_constraints_satisfied, tf.float64)
        if return_separate_components:
            if self._feasible_region_only:
                return cosine_similarities, all_equality_constraints_binding * inequality_constraints_satisfied * expected_improvement
            else:
                return cosine_similarities, all_equality_constraints_binding * expected_improvement
        else:
            if self._feasible_region_only:
                return all_equality_constraints_binding * inequality_constraints_satisfied * cosine_similarities * expected_improvement
            else:
                return all_equality_constraints_binding * cosine_similarities * expected_improvement

    def get_expected_improvement(self, x: tf.Tensor) -> tf.Tensor:
        """
        Get expected improvement at given point. Used when testing if the point returned by the acquisition function
        has sufficiently large expected improvement.
        """
        # TODO: Improve docstring
        tf.debugging.assert_shapes(
            [(x, [1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        print(f"Expected improvement query x: {x}, shape: {x.shape}")

        objective_mean, objective_var = self._objective_model.predict(x)
        normal = tfp.distributions.Normal(objective_mean, tf.sqrt(objective_var))
        expected_improvement = (self.best_valid_observation - objective_mean) * normal.cdf(self.best_valid_observation) + objective_var * normal.prob(self.best_valid_observation)
        return expected_improvement


    def get_satisfied_objective_values(self, x: tf.Tensor) -> tf.Tensor:
        """
        Returns objective value functions where the upper bounds of the 90% confidence intervals on the constraints
        are below 0 (suggesting they're all satisfied). Where not satisfied, replaces true function value with arbitrary
        large function value.
        """
        # TODO: Improve docstring
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        x = tf.squeeze(x, -2)

        objective_mean, objective_var = self._objective_model.predict(x)
        satisfied_objective_values = objective_mean

        for tag, model in self._inequality_constraint_models.items():
            constraint_pred_mean, constraint_pred_var = model.predict(x)
            satisfied = constraint_pred_mean + norm.ppf(0.9) * tf.sqrt(constraint_pred_var) <= 0
            assert (satisfied_objective_values.shape == satisfied.shape)

            # Where not satisfied, give large value so doesn't get chosen
            satisfied_objective_values = tf.where(satisfied, satisfied_objective_values, 1000) # TODO - Remove hard-coded bad value

        # The optimiser maximises function given, so in order to find the minimum we need to negate
        return - satisfied_objective_values


    # Used for plotting in '10-03-23/run_four'
    # def _plot_models(self, prev_query_point):
    #     """
    #     Plot visualisation of surrogate models for objective and inequality constraints, as well as the augmented
    #     Lagrangian.
    #     """
    #     print(f"Prev Query: {prev_query_point}")
    #     x_list = tf.linspace(0, 1, 100)
    #     y_list = tf.linspace(0, 1, 100)
    #     xs, ys = tf.meshgrid(x_list, y_list)
    #     coordinates = tf.expand_dims(tf.stack((tf.reshape(xs, [-1]), tf.reshape(ys, [-1])), axis=1), -2)
    #     objective_pred, _ = self._objective_model.predict(coordinates)
    #     kkt_ei_pred = self._efficient_kkt_expected_improvement(coordinates)
    #     cons_one_mean, cons_one_var = self._inequality_constraint_models["INEQUALITY_CONSTRAINT_ONE"].predict(coordinates)
    #     cosine_similarities, expected_improvement = self._efficient_kkt_expected_improvement_separate(coordinates)
    #     fig, (ax1, ax2, ax3, ax4) = plt.subplots(3, 3, figsize=(15, 12))
    #     objective_plot = ax1[0].contourf(xs, ys, tf.reshape(objective_pred, [y_list.shape[0], x_list.shape[0]]), levels=500)
    #     ax1[0].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     fig.colorbar(objective_plot)
    #     ax1[0].set_xlabel("OBJECTIVE")
    #
    #     constraint_one_pred, _ = self._inequality_constraint_models["INEQUALITY_CONSTRAINT_ONE"].predict(coordinates)
    #     constraint_one_plot = ax1[1].contourf(xs, ys, tf.reshape(constraint_one_pred, [y_list.shape[0], x_list.shape[0]]),
    #                                           levels=500)
    #     ax1[1].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     fig.colorbar(constraint_one_plot)
    #     ax1[1].set_xlabel("INEQUALITY CONSTRAINT ONE")
    #
    #     constraint_two_pred, _ = self._inequality_constraint_models["INEQUALITY_CONSTRAINT_TWO"].predict(coordinates)
    #     constraint_two_plot = ax1[2].contourf(xs, ys, tf.reshape(constraint_two_pred, [y_list.shape[0], x_list.shape[0]]),
    #                                           levels=500)
    #     ax1[2].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     fig.colorbar(constraint_two_plot)
    #     ax1[2].set_xlabel("INEQUALITY CONSTRAINT TWO")
    #
    #     kkt_plot = ax2[0].contourf(xs, ys, tf.reshape(kkt_ei_pred, [y_list.shape[0], x_list.shape[0]]), levels= 500, extend="both")
    #     ax2[0].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     ax2[0].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(kkt_plot)
    #     ax2[0].set_xlabel("KKT Expected Improvement)")
    #
    #     cosine_plot = ax2[1].contourf(xs, ys, tf.reshape(cosine_similarities, [y_list.shape[0], x_list.shape[0]]), levels=500,
    #                                extend="both")
    #     ax2[1].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     ax2[1].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(cosine_plot)
    #     ax2[1].set_xlabel("Cosine Similarities")
    #
    #     ei_plot = ax2[2].contourf(xs, ys, tf.reshape(expected_improvement, [y_list.shape[0], x_list.shape[0]]),
    #                                   levels=500,
    #                                   extend="both")
    #     ax2[2].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     ax2[2].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(ei_plot)
    #     ax2[2].set_xlabel("Expected Improvement")
    #
    #     ax3[0].text(0.5, 0.5,
    #                 f"Iteration: {self._iteration - 1} \n  Query: {prev_query_point} \n Best Valid Observation: {self._best_valid_observation}",
    #                 horizontalalignment='center', verticalalignment='center')
    #
    #     cons_one_std_plot = ax3[1].contourf(xs, ys,
    #                                         tf.reshape(tf.sqrt(cons_one_var), [y_list.shape[0], x_list.shape[0]]),
    #                                         levels=500,
    #                                         extend="both")
    #     ax3[1].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     ax3[1].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(cons_one_std_plot)
    #     ax3[1].set_xlabel("Constraint One STD")
    #
    #     cons_one_binding = tf.cast(tf.abs(cons_one_mean) / tf.sqrt(cons_one_var) <= 1.2816, dtype=tf.int32)
    #     cons_one_binding_plot = ax3[2].contourf(xs, ys,tf.reshape(cons_one_binding, [y_list.shape[0], x_list.shape[0]]),
    #                                         levels=500,
    #                                         extend="both")
    #     ax3[2].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     ax3[2].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(cons_one_binding_plot)
    #     ax3[2].set_xlabel("Constraint One Binding")
    #
    #     ax3[0].axis("off")
    #
    #     ax4[0]
    #
    #     plt.tight_layout()
    #     with open(f"../results/10-03-23/run_four/visualisation/iter_{self._iteration - 1}.png", "wb") as fp:
    #         plt.savefig(fp)
    #     plt.show()

    # Used for plotting in '13-03-23/run_one' for testing out gradients
    # def _plot_models(self, prev_query_point):
    #     """
    #     Plot visualisation of surrogate models for objective and inequality constraints, as well as the augmented
    #     Lagrangian.
    #     """
    #     print(f"Prev Query: {prev_query_point}")
    #     x_list = tf.linspace(0, 1, 100)
    #     y_list = tf.linspace(0, 1, 100)
    #     xs, ys = tf.meshgrid(x_list, y_list)
    #     coordinates = tf.expand_dims(tf.stack((tf.reshape(xs, [-1]), tf.reshape(ys, [-1])), axis=1), -2)
    #     kkt_ei_pred = self._efficient_kkt_expected_improvement(coordinates)
    #
    #     with tf.GradientTape(persistent=True) as tape:
    #         tape.watch(coordinates)
    #         cons_one_mean, cons_one_var = self._inequality_constraint_models["INEQUALITY_CONSTRAINT_ONE"].predict(coordinates)
    #
    #     cons_one_grad = tape.gradient(cons_one_mean, coordinates)
    #     cons_one_grad = tf.squeeze(cons_one_grad, -2)
    #     cosine_similarities, expected_improvement = self._efficient_kkt_expected_improvement_separate(coordinates)
    #     fig, (ax1, ax2, ax3) = plt.subplots(3, 3, figsize=(15, 7))
    #
    #     constraint_one_plot = ax1[0].contourf(xs, ys, tf.reshape(cons_one_mean, [y_list.shape[0], x_list.shape[0]]), levels=500)
    #     ax1[0].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     fig.colorbar(constraint_one_plot)
    #     ax1[0].set_xlabel("INEQUALITY CONSTRAINT ONE MODEL")
    #
    #     x0_grad_plot = ax1[1].contourf(xs, ys, tf.reshape(cons_one_grad[:,0], [y_list.shape[0], x_list.shape[0]]), levels=500)
    #     ax1[1].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(x0_grad_plot)
    #     ax1[1].set_xlabel("GRADIENT OF INEQUALITY CONSTRAINT ONE MODEL WRT X0")
    #
    #     x1_grad_plot = ax1[2].contourf(xs, ys, tf.reshape(cons_one_grad[:, 1], [y_list.shape[0], x_list.shape[0]]),
    #                                    levels=500)
    #     ax1[2].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(x1_grad_plot)
    #     ax1[2].set_xlabel("GRADIENT OF INEQUALITY CONSTRAINT ONE MODEL WRT X1")
    #
    #     kkt_plot = ax2[0].contourf(xs, ys, tf.reshape(kkt_ei_pred, [y_list.shape[0], x_list.shape[0]]), levels= 500, extend="both")
    #     ax2[0].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     ax2[0].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(kkt_plot)
    #     ax2[0].set_xlabel("KKT Expected Improvement)")
    #
    #     cosine_plot = ax2[1].contourf(xs, ys, tf.reshape(cosine_similarities, [y_list.shape[0], x_list.shape[0]]), levels=500,
    #                                extend="both")
    #     ax2[1].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     ax2[1].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(cosine_plot)
    #     ax2[1].set_xlabel("Cosine Similarities")
    #
    #     ei_plot = ax2[2].contourf(xs, ys, tf.reshape(expected_improvement, [y_list.shape[0], x_list.shape[0]]),
    #                                   levels=500,
    #                                   extend="both")
    #     ax2[2].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     ax2[2].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(ei_plot)
    #     ax2[2].set_xlabel("Expected Improvement")
    #
    #     ax3[0].text(0.5, 0.5,
    #                 f"Iteration: {self._iteration - 1} \n  Query: {prev_query_point} \n Best Valid Observation: {self._best_valid_observation}",
    #                 horizontalalignment='center', verticalalignment='center')
    #
    #     cons_one_std_plot = ax3[1].contourf(xs, ys,
    #                                         tf.reshape(tf.sqrt(cons_one_var), [y_list.shape[0], x_list.shape[0]]),
    #                                         levels=500,
    #                                         extend="both")
    #     ax3[1].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     ax3[1].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(cons_one_std_plot)
    #     ax3[1].set_xlabel("Constraint One STD")
    #
    #     cons_one_binding = tf.cast(tf.abs(cons_one_mean) / tf.sqrt(cons_one_var) <= 1.2816, dtype=tf.int32)
    #     cons_one_binding_plot = ax3[2].contourf(xs, ys,tf.reshape(cons_one_binding, [y_list.shape[0], x_list.shape[0]]),
    #                                         levels=500,
    #                                         extend="both")
    #     ax3[2].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     ax3[2].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(cons_one_binding_plot)
    #     ax3[2].set_xlabel("Constraint One Binding")
    #
    #     ax3[0].axis("off")
    #
    #     plt.tight_layout()
    #     with open(f"../results/13-03-23/run_one/visualisation/iter_{self._iteration - 1}.png", "wb") as fp:
    #         plt.savefig(fp)
    #     plt.show()

    # Used for plotting in '14-03-23/zoomed_in_cosine_similarities_two' for visualising cosine similarities close
    # to optimal point in later iterations
    # def _plot_models(self, prev_query_point):
    #     """
    #     Plot visualisation of surrogate models for objective and inequality constraints, as well as the augmented
    #     Lagrangian.
    #     """
    #     print(f"Prev Query: {prev_query_point}")
    #     x_list = tf.linspace(0, 1, 100)
    #     y_list = tf.linspace(0, 1, 100)
    #     xs, ys = tf.meshgrid(x_list, y_list)
    #     coordinates = tf.expand_dims(tf.stack((tf.reshape(xs, [-1]), tf.reshape(ys, [-1])), axis=1), -2)
    #
    #     zoomed_x_list = tf.cast(tf.linspace(0.18, 0.22, 100), dtype=tf.float64)
    #     zoomed_y_list = tf.cast(tf.linspace(0.38, 0.41, 100), dtype=tf.float64)
    #     zoomed_xs, zoomed_ys = tf.meshgrid(zoomed_x_list, zoomed_y_list)
    #     zoomed_coordinates = tf.expand_dims(tf.stack((tf.reshape(zoomed_xs, [-1]), tf.reshape(zoomed_ys, [-1])), axis=1), -2)
    #     kkt_ei_pred = self._efficient_kkt_expected_improvement(coordinates)
    #
    #     cons_one_mean, cons_one_var = self._inequality_constraint_models["INEQUALITY_CONSTRAINT_ONE"].predict(coordinates)
    #     strict_cosine_similarities, expected_improvement = self._efficient_kkt_expected_improvement_separate(coordinates, alpha=0.2)
    #     zoomed_strict_cosine_similarities, _ = self._efficient_kkt_expected_improvement_separate(zoomed_coordinates, alpha=0.2)
    #     zoomed_relaxed_cosine_similarities, _ = self._efficient_kkt_expected_improvement_separate(zoomed_coordinates, alpha=0.1)
    #     zoomed_very_relaxed_cosine_similarities, _ = self._efficient_kkt_expected_improvement_separate(zoomed_coordinates, alpha=0.02)
    #     fig, (ax1, ax2, ax3) = plt.subplots(3, 3, figsize=(15, 7))
    #
    #     constraint_one_plot = ax1[0].contourf(xs, ys, tf.reshape(cons_one_mean, [y_list.shape[0], x_list.shape[0]]), levels=500)
    #     ax1[0].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     fig.colorbar(constraint_one_plot)
    #     ax1[0].set_xlabel("INEQUALITY CONSTRAINT ONE MEAN")
    #
    #     cons_one_std_plot = ax1[1].contourf(xs, ys, tf.reshape(tf.sqrt(cons_one_var), [y_list.shape[0], x_list.shape[0]]),
    #                                         levels=500, extend="both")
    #     ax1[1].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     ax1[1].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(cons_one_std_plot)
    #     ax1[1].set_xlabel("INEQUALITY CONSTRAINT ONE STD")
    #
    #     ax1[2].text(0.5, 0.5,
    #                 f"Iteration: {self._iteration - 1} \n  Query: {prev_query_point} \n Best Valid Observation: {self.best_valid_observation}",
    #                 horizontalalignment='center', verticalalignment='center')
    #
    #     kkt_plot = ax2[0].contourf(xs, ys, tf.reshape(kkt_ei_pred, [y_list.shape[0], x_list.shape[0]]), levels= 500, extend="both")
    #     ax2[0].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     ax2[0].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(kkt_plot)
    #     ax2[0].set_xlabel("KKT Expected Improvement)")
    #
    #     cosine_plot = ax2[1].contourf(xs, ys, tf.reshape(strict_cosine_similarities, [y_list.shape[0], x_list.shape[0]]), levels=500,
    #                                extend="both")
    #     ax2[1].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     ax2[1].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(cosine_plot)
    #     ax2[1].set_xlabel(f"Cosine Similarities, Z-Score = {norm.ppf(1 - 0.2/2)}")
    #
    #     ei_plot = ax2[2].contourf(xs, ys, tf.reshape(expected_improvement, [y_list.shape[0], x_list.shape[0]]),
    #                                   levels=500,
    #                                   extend="both")
    #     ax2[2].scatter(prev_query_point[0], prev_query_point[1], marker="x")
    #     ax2[2].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(ei_plot)
    #     ax2[2].set_xlabel("Expected Improvement")
    #
    #     zoomed_strict_cosine_plot = ax3[0].contourf(zoomed_xs, zoomed_ys, tf.reshape(zoomed_strict_cosine_similarities, [zoomed_y_list.shape[0], zoomed_x_list.shape[0]]),
    #                                         levels=500,
    #                                         extend="both")
    #     ax3[0].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(zoomed_strict_cosine_plot)
    #     ax3[0].set_xlabel(f"Zoomed Cosine Similarities, Z-Score = {norm.ppf(0.90)}")
    #
    #     zoomed_relaxed_cosine_plot = ax3[1].contourf(zoomed_xs, zoomed_ys, tf.reshape(zoomed_relaxed_cosine_similarities, [zoomed_y_list.shape[0], zoomed_x_list.shape[0]]),
    #                                                  levels=500, extend="both")
    #     ax3[1].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(zoomed_relaxed_cosine_plot)
    #     ax3[1].set_xlabel(f"Zoomed Cosine Similarities, Z-Score = {norm.ppf(0.95)}")
    #
    #     zoomed_very_relaxed_cosine_plot = ax3[2].contourf(zoomed_xs, zoomed_ys, tf.reshape(zoomed_very_relaxed_cosine_similarities, [zoomed_y_list.shape[0], zoomed_x_list.shape[0]]),
    #                                                  levels=500, extend="both")
    #     ax3[2].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
    #     fig.colorbar(zoomed_very_relaxed_cosine_plot)
    #     ax3[2].set_xlabel(f"Zoomed Cosine Similarities, Z-Score = {norm.ppf(0.99)}")
    #
    #     ax1[2].axis("off")
    #
    #     plt.tight_layout()
    #     with open(f"../results/14-03-23/zoomed_in_cosine_similarities_two/visualisation/iter_{self._iteration - 1}.png", "wb") as fp:
    #         plt.savefig(fp)
    #     plt.show()

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
        cosine_similarities, expected_improvement = self._efficient_kkt_expected_improvement(True, coordinates)
        kkt_ei_pred = tf.multiply(cosine_similarities, expected_improvement)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 3, figsize=(15, 12))
        objective_plot = ax1[0].contourf(xs, ys, tf.reshape(objective_pred, [y_list.shape[0], x_list.shape[0]]), levels=500)
        ax1[0].scatter(prev_query_point[0], prev_query_point[1], marker="x")
        ax1[0].scatter(0.94769, 0.46856, facecolors='none', edgecolors='r')
        fig.colorbar(objective_plot)
        ax1[0].set_xlabel("OBJECTIVE")

        constraint_one_pred, _ = self._inequality_constraint_models["INEQUALITY_CONSTRAINT_ONE"].predict(coordinates)
        constraint_one_plot = ax1[1].contourf(xs, ys, tf.reshape(constraint_one_pred, [y_list.shape[0], x_list.shape[0]]),
                                              levels=500)
        ax1[1].scatter(prev_query_point[0], prev_query_point[1], marker="x")
        ax1[1].scatter(0.94769, 0.46856, facecolors='none', edgecolors='r')
        fig.colorbar(constraint_one_plot)
        ax1[1].set_xlabel("INEQUALITY CONSTRAINT ONE")

        equality_constraint_one_pred, _ = self._equality_constraint_models["EQUALITY_CONSTRAINT_ONE"].predict(coordinates)
        equality_constraint_one_plot = ax1[2].contourf(xs, ys, tf.reshape(equality_constraint_one_pred, [y_list.shape[0], x_list.shape[0]]),
                                                       levels=500)
        ax1[2].scatter(prev_query_point[0], prev_query_point[1], marker="x")
        ax1[2].scatter(0.94769, 0.46856, facecolors='none', edgecolors='r')
        fig.colorbar(equality_constraint_one_plot)
        ax1[2].set_xlabel("EQUALITY CONSTRAINT ONE")

        kkt_plot = ax2[0].contourf(xs, ys, tf.reshape(kkt_ei_pred, [y_list.shape[0], x_list.shape[0]]), levels= 500, extend="both")
        ax2[0].scatter(prev_query_point[0], prev_query_point[1], marker="x")
        ax2[0].scatter(0.94769, 0.46856, facecolors='none', edgecolors='r')
        fig.colorbar(kkt_plot)
        ax2[0].set_xlabel("KKT Expected Improvement)")

        cosine_plot = ax2[1].contourf(xs, ys, tf.reshape(cosine_similarities, [y_list.shape[0], x_list.shape[0]]), levels=500,
                                   extend="both")
        ax2[1].scatter(prev_query_point[0], prev_query_point[1], marker="x")
        ax2[1].scatter(0.94769, 0.46856, facecolors='none', edgecolors='r')
        fig.colorbar(cosine_plot)
        ax2[1].set_xlabel("Cosine Similarities")

        ei_plot = ax2[2].contourf(xs, ys, tf.reshape(expected_improvement, [y_list.shape[0], x_list.shape[0]]),
                                      levels=500,
                                      extend="both")
        ax2[2].scatter(prev_query_point[0], prev_query_point[1], marker="x")
        ax2[2].scatter(0.94769, 0.46856, facecolors='none', edgecolors='r')
        fig.colorbar(ei_plot)
        ax2[2].set_xlabel("Expected Improvement")

        ax3[0].text(0.5, 0.5,
                    f"Iteration: {self._iteration - 1} \n  Query: {prev_query_point} \n Best Valid Observation: {self.best_valid_observation}",
                    horizontalalignment='center', verticalalignment='center')

        equality_constraint_two_pred, _ = self._equality_constraint_models["EQUALITY_CONSTRAINT_TWO"].predict(coordinates)
        equality_constraint_two_plot = ax3[1].contourf(xs, ys, tf.reshape(equality_constraint_two_pred, [y_list.shape[0], x_list.shape[0]]),
                                                       levels=500)
        ax3[1].scatter(prev_query_point[0], prev_query_point[1], marker="x")
        fig.colorbar(equality_constraint_two_plot)
        ax3[1].set_xlabel("EQUALITY CONSTRAINT TWO")

        ax3[0].axis("off")
        ax3[2].axis("off")

        plt.tight_layout()
        with open(f"../results/10-03-23/run_four/visualisation/iter_{self._iteration - 1}.png", "wb") as fp:
            plt.savefig(fp)
        plt.show()


# Combine KKT conditions with Expected Improvement for constrained optimisation
# as done in https://pure.uvt.nl/ws/portalfiles/portal/63795955/2022_022.pdf, but use
# Thompson samples of constraints for calculating gradients of constraint functions.
class KKTThompsonSamplingExpectedImprovement(KKTAcquisitionFunctionBuilder[ProbabilisticModelType]):
    """
    Builder for an acquisition function combining the KKT conditions and Expected Improvement.
    """

    def __init__(self, objective_tag: Tag, inequality_constraint_prefix: Optional[Tag] = None,
                 equality_constraint_prefix: Optional[Tag] = None, epsilon=0.001,
                 search_space: Optional[SearchSpace] = None, feasible_region_only: bool = False, plot: bool = False):
        """
        :param objective_tag: The tag for the objective data and model.
        :param inequality_constraint_prefix: Prefix of tags for inequality constraint data/models/Lagrange multipliers.
        :param equality_constraint_prefix: Prefix for tags for equality constraint data/models/Lagrange multipliers.
        :param epsilon: Bound within which equality constraints are considered to be satisfied.
        :param search_space: The global search space over which the optimisation is defined. This is
            only used to determine explicit constraints.
        :param plot: Whether to plot the modelled functions at each iteration of BayesOpt.
        """
        super().__init__()
        self._equality_constraint_models = {}
        self._inequality_constraint_models = {}
        self._equality_constraint_trajectories = {}
        self._equality_constraint_trajectory_samplers = {}
        self._inequality_constraint_trajectories = {}
        self._inequality_constraint_trajectory_samplers = {}
        self._objective_model = None
        self._objective_tag = objective_tag
        self._inequality_constraint_prefix = inequality_constraint_prefix
        self._equality_constraint_prefix = equality_constraint_prefix
        self._epsilon = epsilon
        self._search_space = search_space
        self._augmented_lagrangian_fn = None
        self._feasible_region_only = feasible_region_only
        self._plot = plot
        self._iteration = 0
        # TODO: Think of better way to do below
        self.best_valid_observation = 10  # For when we haven't yet found any valid points
        self._kkt_expected_improvement_fn = None
        self._alpha = None

    def __repr__(self) -> str:
        """"""
        return (
            f"KKTThompsonSamplingExpectedImprovement({self._objective_tag!r}, {self._inequality_constraint_prefix!r},"
            f" {self._equality_constraint_prefix!r}, {self._search_space!r})"
        )

    def prepare_acquisition_function(
        self,
        models: Mapping[Tag, HasTrajectorySampler],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
        alpha: float = 0.20,
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

        self._alpha = alpha

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
            self.best_valid_observation = tf.math.reduce_min(valid_y)

        for tag, model in models.items():
            if tag.startswith(self._objective_tag):
                self._objective_model = copy.deepcopy(model)
            elif (self._equality_constraint_prefix is not None) and (tag.startswith(self._equality_constraint_prefix)):
                self._equality_constraint_models[tag] = model
                sampler = model.trajectory_sampler()
                trajectory = sampler.get_trajectory()
                self._equality_constraint_trajectory_samplers[tag] = sampler
                self._equality_constraint_trajectories[tag] = trajectory
            elif (self._inequality_constraint_prefix is not None) and (tag.startswith(self._inequality_constraint_prefix)):
                self._inequality_constraint_models[tag] = model
                sampler = model.trajectory_sampler()
                trajectory = sampler.get_trajectory()
                self._inequality_constraint_trajectory_samplers[tag] = sampler
                self._inequality_constraint_trajectories[tag] = trajectory

        self._kkt_expected_improvement_fn = partial(self._efficient_kkt_expected_improvement, False)

        return self._kkt_expected_improvement_fn

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
        alpha: float = 0.20,
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

        self._alpha = alpha

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
            self.best_valid_observation = min(self.best_valid_observation, most_recent_observation)

        # Update models
        self._objective_model = copy.deepcopy(models[self._objective_tag])

        for tag, model in models.items():
            if (self._equality_constraint_prefix is not None) and (tag.startswith(self._equality_constraint_prefix)):
                self._equality_constraint_models[tag] = model
            elif (self._inequality_constraint_prefix is not None) and (tag.startswith(self._inequality_constraint_prefix)):
                self._inequality_constraint_models[tag] = model

        for tag, sampler in self._inequality_constraint_trajectory_samplers.items():
            old_trajectory = self._inequality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            self._inequality_constraint_trajectories[tag] = updated_trajectory

        for tag, sampler in self._equality_constraint_trajectory_samplers.items():
            old_trajectory = self._equality_constraint_trajectories[tag]
            updated_trajectory = sampler.update_trajectory(old_trajectory)
            self._equality_constraint_trajectories[tag] = updated_trajectory

        # Plot Models
        if self._plot:
            self._plot_models(most_recent_query_point)

        print(f"Best Valid Observation: {self.best_valid_observation}")
        self._kkt_expected_improvement_fn = partial(self._efficient_kkt_expected_improvement, False)

        return self._kkt_expected_improvement_fn

    def _efficient_kkt_expected_improvement(self,
                                            return_separate_components: bool,
                                            x: tf.Tensor) -> tf.Tensor | (tf.Tensor, tf.Tensor):
        """
        Form KKT expected improvement from objective and constraints
        :param: return_separate_components: Whether to return cosine similarity and expected improvement separately (
                sometimes used for plotting purposes).
        :param x: Array of points at which to evaluate the kkt expected improvement at of shape [N, 1, D]
        """
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )

        # Standard GP prediction takes tensor of shape [N, D]
        squeezed_x = tf.squeeze(x, -2)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(squeezed_x)
            objective_mean, objective_var = self._objective_model.predict(squeezed_x)
        mle_objective_grad = tape.gradient(objective_mean, squeezed_x)  # [N, D]
        tf.debugging.assert_shapes([(mle_objective_grad, [..., 2])])
        normal = tfp.distributions.Normal(objective_mean, tf.sqrt(objective_var))

        # TODO: Double check expected improvement definition
        expected_improvement = (self.best_valid_observation - objective_mean) * normal.cdf(self.best_valid_observation) + objective_var * normal.prob(self.best_valid_observation)
        num_x_vals = x.shape[0]
        cosine_similarities = []

        inequality_constraint_grad_dict = {}
        inequality_constraint_bind_dict = {}
        inequality_constraints_satisfied = tf.constant(True, shape=(x.shape[0], 1))  # [N, 1]
        for tag, trajectory in self._inequality_constraint_trajectories.items():
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                constraint_vals = trajectory(x)  # [N, B, 1]
            constraint_mean, constraint_var = self._inequality_constraint_models[tag].predict(tf.squeeze(x, -2))
            binding = tf.abs(constraint_mean) / tf.sqrt(constraint_var) <= norm.ppf(1 - self._alpha/2)
            inequality_constraints_satisfied = tf.logical_and(inequality_constraints_satisfied, tf.squeeze(constraint_vals, -2) <= 0)

            constraint_grad = tf.squeeze(tape.gradient(constraint_vals, x), axis=1)
            print(f"Constraint Grad Shape: {constraint_grad.shape}")
            print(f"Binding Shape: {binding.shape}")
            inequality_constraint_grad_dict[tag] = constraint_grad
            inequality_constraint_bind_dict[tag] = binding

        equality_constraint_grad_dict = {}
        equality_constraint_bind_dict = {}
        all_equality_constraints_binding = tf.constant(True, shape=(x.shape[0], 1))  # [N, 1]
        for tag, trajectory in self._equality_constraint_trajectories.items():
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                constraint_vals = trajectory(x)  # [N, B, 1]
            constraint_mean, constraint_var = self._equality_constraint_models[tag].predict(tf.squeeze(x, -2))
            binding = tf.abs(constraint_mean) / tf.sqrt(constraint_var) <= norm.ppf(1 - self._alpha/2)
            all_equality_constraints_binding = tf.logical_and(all_equality_constraints_binding, binding)
            constraint_grad = tf.squeeze(tape.gradient(constraint_vals, x), axis=1)
            equality_constraint_grad_dict[tag] = constraint_grad
            equality_constraint_bind_dict[tag] = binding

        for i in range(num_x_vals):
            point_inequality_grad_dict = {}
            point_equality_grad_dict = {}
            point_mle_objective_grad = mle_objective_grad[i][..., None]

            # Calculate inequality constraint gradients
            for tag, _ in self._inequality_constraint_trajectories.items():
                binding = inequality_constraint_bind_dict[tag][i]
                # Only consider binding constraints
                if binding:
                    point_constraint_grad = inequality_constraint_grad_dict[tag][i]
                    point_inequality_grad_dict[tag] = point_constraint_grad

            # Calculate equality constraint gradients
            for tag, _ in self._equality_constraint_trajectories.items():
                binding = equality_constraint_bind_dict[tag][i]
                # Only consider binding constraints
                if binding:
                    point_constraint_grad = equality_constraint_grad_dict[tag][i]
                    point_equality_grad_dict[tag] = point_constraint_grad

            # Construct gradient matrix
            gradient_matrix = None

            for tag, gradient in point_inequality_grad_dict.items():
                if gradient_matrix is None:
                    gradient_matrix = gradient[..., None]
                else:
                    gradient_matrix = tf.concat((gradient_matrix, gradient[..., None]), axis=1)

            for tag, gradient in point_equality_grad_dict.items():
                if gradient_matrix is None:
                    gradient_matrix = gradient[..., None]
                else:
                    gradient_matrix = tf.concat((gradient_matrix, gradient[..., None]), axis=1)

            num_binding_inequality_constraints = len(point_inequality_grad_dict)
            if gradient_matrix is None:
                # No constraints are considered binding, so unlikely to be optimal point
                cosine_similarities.append(0)
            else:
                lagrange_multipliers = tf.linalg.lstsq(gradient_matrix, -point_mle_objective_grad)
                if num_binding_inequality_constraints > 0:
                    # Enforce non-negativity of inequality constraint Lagrange multipliers
                    inequality_indices = tf.Variable([[i] for i in range(num_binding_inequality_constraints)])
                    lagrange_multipliers = tf.tensor_scatter_nd_update(lagrange_multipliers, inequality_indices, tf.nn.relu(lagrange_multipliers)[:num_binding_inequality_constraints])
                ls_objective_grad = - tf.matmul(gradient_matrix, lagrange_multipliers)
                normalized_mle_objective_grad = tf.math.l2_normalize(point_mle_objective_grad, axis=0)
                normalized_ls_objective_grad = tf.math.l2_normalize(ls_objective_grad, axis=0)
                cosine_similarity = tf.reduce_sum(tf.multiply(normalized_mle_objective_grad, normalized_ls_objective_grad))
                cosine_similarities.append(cosine_similarity)

        cosine_similarities = tf.convert_to_tensor(cosine_similarities, dtype=tf.float64)[..., None]
        assert (expected_improvement.shape == cosine_similarities.shape)
        assert (all_equality_constraints_binding.shape == expected_improvement.shape)
        assert (inequality_constraints_satisfied.shape == expected_improvement.shape)
        all_equality_constraints_binding = tf.cast(all_equality_constraints_binding, tf.float64)
        inequality_constraints_satisfied = tf.cast(inequality_constraints_satisfied, tf.float64)
        if return_separate_components:
            if self._feasible_region_only:
                return cosine_similarities, all_equality_constraints_binding * inequality_constraints_satisfied * expected_improvement
            else:
                return cosine_similarities, all_equality_constraints_binding * expected_improvement
        else:
            if self._feasible_region_only:
                return all_equality_constraints_binding * inequality_constraints_satisfied * cosine_similarities * expected_improvement
            else:
                return all_equality_constraints_binding * cosine_similarities * expected_improvement

    def get_expected_improvement(self, x: tf.Tensor) -> tf.Tensor:
        """
        Get expected improvement at given point. Used when testing if the point returned by the acquisition function
        has sufficiently large expected improvement.
        """
        # TODO: Improve docstring
        tf.debugging.assert_shapes(
            [(x, [1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        print(f"Expected improvement query x: {x}, shape: {x.shape}")

        objective_mean, objective_var = self._objective_model.predict(x)
        normal = tfp.distributions.Normal(objective_mean, tf.sqrt(objective_var))
        expected_improvement = (self.best_valid_observation - objective_mean) * normal.cdf(self.best_valid_observation) + objective_var * normal.prob(self.best_valid_observation)
        return expected_improvement

    def get_satisfied_objective_values(self, x: tf.Tensor) -> tf.Tensor:
        """
        Returns objective value functions where Thompson samples of the constraints are satisfied. Where not satisfied,
        replaces true function value with arbitrary large function value.
        """
        # TODO: Improve docstring
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )

        objective_mean, objective_var = self._objective_model.predict(tf.squeeze(x, -2))
        satisfied_objective_values = objective_mean  # [N, 1]

        satisfied = tf.constant(True, shape=satisfied_objective_values.shape)
        for tag, trajectory in self._inequality_constraint_trajectories.items():
            constraint_vals = tf.squeeze(trajectory(x), axis=-2)
            satisfied = tf.logical_and(satisfied, constraint_vals <= 0)
            assert (satisfied_objective_values.shape == satisfied.shape)

        for tag, trajectory in self._equality_constraint_trajectories.items():
            constraint_vals = tf.squeeze(trajectory.predict(x), axis=-2)
            satisfied = tf.logical_and(satisfied, tf.abs(constraint_vals) <= self._epsilon)
            assert (satisfied_objective_values.shape == satisfied.shape)

        # Where not satisfied, give large value so doesn't get chosen
        satisfied_objective_values = tf.where(satisfied, satisfied_objective_values, 1000)  # TODO - Remove hard-coded bad value

        # The optimiser maximises function given, so in order to find the minimum we need to negate
        return - satisfied_objective_values

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
        cosine_similarities, expected_improvement = self._efficient_kkt_expected_improvement(True, coordinates)
        kkt_ei_pred = tf.multiply(cosine_similarities, expected_improvement)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 3, figsize=(15, 12))
        objective_plot = ax1[0].contourf(xs, ys, tf.reshape(objective_pred, [y_list.shape[0], x_list.shape[0]]), levels=500)
        fig.colorbar(objective_plot)
        ax1[0].set_xlabel("OBJECTIVE")

        constraint_one_pred = self._inequality_constraint_trajectories["INEQUALITY_CONSTRAINT_ONE"](coordinates)
        constraint_one_plot = ax1[1].contourf(xs, ys, tf.reshape(constraint_one_pred, [y_list.shape[0], x_list.shape[0]]),
                                              levels=500)
        fig.colorbar(constraint_one_plot)
        ax1[1].set_xlabel("INEQUALITY CONSTRAINT ONE")

        constraint_two_pred = self._inequality_constraint_trajectories["INEQUALITY_CONSTRAINT_TWO"](coordinates)
        constraint_two_plot = ax1[2].contourf(xs, ys, tf.reshape(constraint_two_pred, [y_list.shape[0], x_list.shape[0]]),
                                              levels=500)
        fig.colorbar(constraint_two_plot)
        ax1[2].set_xlabel("INEQUALITY CONSTRAINT TWO")

        kkt_plot = ax2[0].contourf(xs, ys, tf.reshape(kkt_ei_pred, [y_list.shape[0], x_list.shape[0]]), levels= 500, extend="both")
        ax2[0].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
        fig.colorbar(kkt_plot)
        ax2[0].set_xlabel("KKT Expected Improvement")

        cosine_plot = ax2[1].contourf(xs, ys, tf.reshape(cosine_similarities, [y_list.shape[0], x_list.shape[0]]), levels=500,
                                      extend="both")
        ax2[1].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
        fig.colorbar(cosine_plot)
        ax2[1].set_xlabel("Cosine Similarities")

        ei_plot = ax2[2].contourf(xs, ys, tf.reshape(expected_improvement, [y_list.shape[0], x_list.shape[0]]),
                                      levels=500,
                                      extend="both")
        ax2[2].scatter(0.1954, 0.4044, facecolors='none', edgecolors='r')
        fig.colorbar(ei_plot)
        ax2[2].set_xlabel("Expected Improvement")

        ax3[1].text(0.5, 0.5,
                    f"Iteration: {self._iteration - 1} \n  Previous Query: {prev_query_point} \n Best Valid Observation: {self.best_valid_observation}",
                    horizontalalignment='center', verticalalignment='center')

        ax3[0].axis("off")
        ax3[1].axis("off")
        ax3[2].axis("off")

        plt.tight_layout()
        with open(f"../results/07-04-23/kkt_lsq_ts/visualisations/iter_{self._iteration - 1}.png", "wb") as fp:
            plt.savefig(fp)
        # plt.show()
