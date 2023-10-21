# Copyright 2021 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
This module contains functionality for optimizing
:data:`~trieste.acquisition.AcquisitionFunction`\ s over :class:`~trieste.space.SearchSpace`\ s.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, cast

import greenlet as gr
import numpy as np
import scipy.optimize as spo
import tensorflow as tf
import tensorflow_probability as tfp

from .. import logging
from ..space import (
    Box,
    Constraint,
    DiscreteSearchSpace,
    SearchSpace,
    SearchSpaceType,
    TaggedProductSearchSpace,
)
from ..types import TensorType
from .interface import AcquisitionFunction

NUM_SAMPLES_MIN: int = 5000
"""
The default minimum number of initial samples for :func:`generate_continuous_optimizer` and
:func:`generate_random_search_optimizer` function, used for determining the number of initial
samples in the multi-start acquisition function optimization.
"""


NUM_SAMPLES_DIM: int = 1000
"""
The default minimum number of initial samples per dimension of the search space for
:func:`generate_continuous_optimizer` function in :func:`automatic_optimizer_selector`, used for
determining the number of initial samples in the multi-start acquisition function optimization.
"""

NUM_RUNS_DIM: int = 10
"""
The default minimum number of optimization runs per dimension of the search space for
:func:`generate_continuous_optimizer` function in :func:`automatic_optimizer_selector`, used for
determining the number of acquisition function optimizations to be performed in parallel.
"""


class FailedOptimizationError(Exception):
    """Raised when an acquisition optimizer fails to optimize"""


AcquisitionOptimizer = Callable[
    [SearchSpaceType, Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]]],
    TensorType,
]
"""
Type alias for a function that returns the single point that maximizes an acquisition function over
a search space or the V points that maximize a vectorized acquisition function (as represented by an
acquisition-int tuple).

If this function receives a search space with points of shape [D] and an acquisition function
with input shape [..., 1, D] output shape [..., 1], the :const:`AcquisitionOptimizer` return shape
should be [1, D].

If instead it receives a search space and a tuple containing the acquisition function and its
vectorization V then the :const:`AcquisitionOptimizer` return shape should be [V, D].
"""

ConstrainedTrustRegionAcquisitionOptimizer = Callable[
    [
        SearchSpaceType,
        Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
        TensorType,
        float,
    ],
    TensorType,
]
"""
Similar to a plain `AcquisitionOpimizer`, but with an addtional two arguments:
    - `x_min` - Best point found so far (which serves as the center of the trust region).
    - `perturbation_prob` - Probability of perturbing points from `x_min`. Once we are
      into high dimensions, Eriksson et al. 2019 (https://arxiv.org/pdf/1910.01739.pdf)
      suggests that when choosing the point to query next, we shouldn't venture too far
      from the best point found so far, and so the next point should be equal to `x_min` 
      with probability `1 - perturbation_prob` along a given dimension.
"""


def automatic_optimizer_selector(
    space: SearchSpace,
    target_func: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
) -> TensorType:
    """
    A wrapper around our :const:`AcquisitionOptimizer`s. This class performs
    an :const:`AcquisitionOptimizer` appropriate for the
    problem's :class:`~trieste.space.SearchSpace`.

    :param space: The space of points over which to search, for points with shape [D].
    :param target_func: The function to maximise, with input shape [..., 1, D] and output shape
            [..., 1].
    :return: The batch of points in ``space`` that maximises ``target_func``, with shape [1, D].
    """

    if isinstance(space, DiscreteSearchSpace):
        return optimize_discrete(space, target_func)

    elif isinstance(space, (Box, TaggedProductSearchSpace)):
        num_samples = tf.maximum(
            NUM_SAMPLES_MIN, NUM_SAMPLES_DIM * tf.shape(space.lower)[-1]
        )
        num_runs = NUM_RUNS_DIM * tf.shape(space.lower)[-1]
        return generate_continuous_optimizer(
            num_initial_samples=num_samples,
            num_optimization_runs=num_runs,
        )(space, target_func)

    else:
        raise NotImplementedError(
            f""" No optimizer currently supports acquisition function
                    maximisation over search spaces of type {space}.
                    Try specifying the optimize_random optimizer"""
        )


def _get_max_discrete_points(
    points: TensorType,
    target_func: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
    x_min: Optional[TensorType] = None,
    perturbation_prob: Optional[float] = None,
    batchify_joint: bool = False,
    batch_size: Optional[int] = None,
) -> TensorType:
    # check if we need a vectorized optimizer
    if isinstance(target_func, tuple):
        target_func, V = target_func
    else:
        V = 1

    if V < 0:
        raise ValueError(f"vectorization must be positive, got {V}")

    if x_min is not None and perturbation_prob is not None:
        # If a perturbation probability is provided, we perturb each dimension of the
        # best point found so far with probability `perturbation_prob`. Eriksson et al. 2019
        # (https://arxiv.org/pdf/1910.01739.pdf) reason that once we're in
        # high-dimensional spaces we shouldn't perturb all dimensions of the best
        # discovered point so far when choosing a new point to query.
        tf.debugging.assert_rank(x_min, 1)
        if not batchify_joint:
            x_min = x_min[None, None, ...]
            perturb = tf.cast(tfp.distributions.Bernoulli(probs=perturbation_prob).sample(sample_shape=points.shape), tf.bool)
            points = tf.where(perturb, points, x_min)
        else:
            # Reshape x_min to be consistent with reshaping of candidate points done for batchify_joint
            x_min = tf.tile(x_min, [batch_size])[None, None, ...]
            perturb = tf.cast(tfp.distributions.Bernoulli(probs=perturbation_prob).sample(sample_shape=points.shape), tf.bool)
            points = tf.where(perturb, points, x_min)

    tiled_points = tf.tile(points, [1, V, 1])
    target_func_values = target_func(tiled_points)
    tf.debugging.assert_shapes(
        [(target_func_values, ("_", V))],
        message=(
            f"""
            The result of function target_func has shape
            {tf.shape(target_func_values)}, however, expected a trailing
            dimension of size {V}.
            """
        ),
    )
    best_indices = tf.math.argmax(target_func_values, axis=0)  # [V]
    return tf.gather(
        tf.transpose(tiled_points, [1, 0, 2]), best_indices, batch_dims=1
    )  # [V, D]

def optimize_discrete(
    space: DiscreteSearchSpace,
    target_func: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
) -> TensorType:
    """
    An :const:`AcquisitionOptimizer` for :class:'DiscreteSearchSpace' spaces.

    When this functions receives an acquisition-integer tuple as its `target_func`,it evaluates
    all the points in the search space for each of the individual V functions making
    up `target_func`.

    :param space: The space of points over which to search, for points with shape [D].
    :param target_func: The function to maximise, with input shape [..., V, D] and output shape
            [..., V].
    :return: The V points in ``space`` that maximises ``target_func``, with shape [V, D].
    """
    points = space.points[:, None, :]
    return _get_max_discrete_points(points, target_func)


# WARNING: Below has been adapted solely for running paper experiments.
def generate_continuous_optimizer(
    num_initial_samples: int = NUM_SAMPLES_MIN,
    num_optimization_runs: int = 10,
    num_recovery_runs: int = 10,
    optimizer_args: Optional[dict[str, Any]] = None,
) -> ConstrainedTrustRegionAcquisitionOptimizer[Box | TaggedProductSearchSpace]:
    """
    Generate a gradient-based optimizer for :class:'Box' and :class:'TaggedProductSearchSpace'
    spaces and batches of size 1. In the case of a :class:'TaggedProductSearchSpace', We perform
    gradient-based optimization across all :class:'Box' subspaces, starting from the best location
    found across a sample of `num_initial_samples` random points.

    We advise the user to either use the default `NUM_SAMPLES_MIN` for `num_initial_samples`, or
    `NUM_SAMPLES_DIM` times the dimensionality of the search space, whichever is greater.
    Similarly, for `num_optimization_runs`, we recommend using `NUM_RUNS_DIM` times the
    dimensionality of the search space.

    This optimizer uses Scipy's L-BFGS-B optimizer. We run `num_optimization_runs` separate
    optimizations in parallel, each starting from one of the best `num_optimization_runs` initial
    query points.

    If all `num_optimization_runs` optimizations fail to converge then we run
    `num_recovery_runs` additional runs starting from random locations (also ran in parallel).

    :param num_initial_samples: The size of the random sample used to find the starting point(s) of
        the optimization.
    :param num_optimization_runs: The number of separate optimizations to run.
    :param num_recovery_runs: The maximum number of recovery optimization runs in case of failure.
    :param optimizer_args: The keyword arguments to pass to the Scipy L-BFGS-B optimizer.
        Check `minimize` method  of :class:`~scipy.optimize` for details of which arguments
        can be passed. Note that method, jac and bounds cannot/should not be changed.
    :return: The acquisition optimizer.
    """
    if num_initial_samples <= 0:
        raise ValueError(
            f"num_initial_samples must be positive, got {num_initial_samples}"
        )

    if num_optimization_runs < 0:
        raise ValueError(
            f"num_optimization_runs must be positive, got {num_optimization_runs}"
        )

    if num_initial_samples < num_optimization_runs:
        raise ValueError(
            f"""
            num_initial_samples {num_initial_samples} must be at
            least num_optimization_runs {num_optimization_runs}
            """
        )

    if num_recovery_runs <= -1:
        raise ValueError(
            f"num_recovery_runs must be zero or greater, got {num_recovery_runs}"
        )

    def optimize_continuous(
        space: Box | TaggedProductSearchSpace,
        target_func: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
        x_min: Optional[TensorType] = None,
        perturbation_prob: Optional[float] = None,
    ) -> TensorType:
        """
        A gradient-based :const:`AcquisitionOptimizer` for :class:'Box'
        and :class:`TaggedProductSearchSpace' spaces.

        For :class:'TaggedProductSearchSpace' we only apply gradient updates to
        its class:'Box' subspaces.

        When this functions receives an acquisition-integer tuple as its `target_func`,it
        optimizes each of the individual V functions making up `target_func`, i.e.
        evaluating `num_initial_samples` samples, running `num_optimization_runs` runs, and
        (if necessary) running `num_recovery_runs` recovery run for each of the individual
        V functions.

        :param space: The space over which to search.
        :param target_func: The function to maximise, with input shape [..., V, D] and output shape
                [..., V].
        :x_min: Location of best point found in the search space so far.
        :perturbation_prob: Probability of perturbing a dimension of the
        best value found so far. Eriksson et al. 2019
        (https://arxiv.org/pdf/1910.01739.pdf) reason that once we're in
        high-dimensional spaces we shouldn't perturb all dimensions of the best
        discovered point so far when choosing a new point to query.
        :return: The V points in ``space`` that maximises``target_func``, with shape [V, D].
        """

        if isinstance(target_func, tuple):  # check if we need a vectorized optimizer
            target_func, V = target_func
        else:
            V = 1
        
        if V != 1:
            raise ValueError("This modified optimizer only supports vectorization of 1")

        candidates = space.sample(num_initial_samples)[
            :, None, :
        ]  # [num_initial_samples, 1, D]

        if x_min is None and perturbation_prob is None:
            # If we havent't set a `perturbation_prob` value (as will be the case if we aren't using a
            # trust-region based approach) then we set it to 1.0 so that all dimensions
            # of the best value found so far are perturbed.
            x_min = tf.zeros(shape=(1, 1, space.dimension), dtype=tf.float64)
            perturbation_prob = 1.0
        else:
            tf.debugging.assert_rank(x_min, 1)
            x_min = x_min[None, None, ...]

        perturb = tf.cast(
            tfp.distributions.Bernoulli(probs=perturbation_prob).sample(
                sample_shape=candidates.shape
            ),
            tf.bool,
        )
        candidates = tf.where(perturb, candidates, x_min)
        tiled_candidates = tf.tile(candidates, [1, V, 1])  # [num_initial_samples, V, D]
        tiled_perturb = tf.tile(perturb, [1, V, 1])  # [num_initial_samples, V, D]

        target_func_values = target_func(tiled_candidates)  # [num_samples, V]
        tf.debugging.assert_shapes(
            [(target_func_values, ("_", V))],
            message=(
                f"""
                The result of function target_func has shape
                {tf.shape(target_func_values)}, however, expected a trailing
                dimension of size {V}.
                """
            ),
        )

        _, top_k_indices = tf.math.top_k(
            tf.transpose(target_func_values), k=num_optimization_runs
        )  # [1, num_optimization_runs] or [V, num_optimization_runs]

        tiled_candidates = tf.transpose(
            tiled_candidates, [1, 0, 2]
        )  # [V, num_initial_samples, D]
        tiled_perturb = tf.transpose(tiled_perturb, [1, 0, 2])  # [V, num_initial_samples, D]
        top_k_points = tf.gather(
            tiled_candidates, top_k_indices, batch_dims=1
        )  # [V, num_optimization_runs, D]
        top_k_tiled_perturb = tf.gather(tiled_perturb, top_k_indices, batch_dims=1)  # [V, num_optimization_runs, D]
        initial_points = tf.transpose(
            top_k_points, [1, 0, 2]
        )  # [num_optimization_runs,V,D]
        initial_points_perturb = tf.transpose(top_k_tiled_perturb, [1, 0, 2])  # [num_optimization_runs, V, D]
        # Use this mask to zero out gradients of non-perturbed points to ensure that
        # they don't get perturbed by gradient-based optimisation.
        perturb_grad_mask = tf.cast(initial_points_perturb, tf.float64)

        (
            successes,
            fun_values,
            chosen_x,
            nfev,
        ) = _perform_parallel_continuous_optimization(  # [num_optimization_runs, V]
            target_func,
            space,
            initial_points,
            perturb_grad_mask,
            optimizer_args or {},
        )

        # fun_values: [num_optimization_runs, V] - We fix num_optimization_runs = 1
        # chosen_x: [num_optimization_runs, V, D] - We fix num_optimization_runs = 1
        fun_values = tf.squeeze(fun_values, axis=0)  # [V]
        chosen_x = tf.squeeze(chosen_x, axis=0)  # [V, D]
        squeezed_initial_points = tf.squeeze(initial_points, axis=0)
        chosen_points = tf.where(
            tf.squeeze(successes, axis=0)[:, None], chosen_x, squeezed_initial_points
        )  # [V, D]

        return chosen_points

    return optimize_continuous


def _perform_parallel_continuous_optimization(
    target_func: AcquisitionFunction,
    space: SearchSpace,
    starting_points: TensorType,
    perturb_grad_mask: TensorType,
    optimizer_args: dict[str, Any],
) -> Tuple[TensorType, TensorType, TensorType, TensorType]:
    """
    A function to perform parallel optimization of our acquisition functions
    using Scipy. We perform L-BFGS-B starting from each of the locations contained
    in `starting_points`, i.e. the number of individual optimization runs is
    given by the leading dimension of `starting_points`.

    To provide a parallel implementation of Scipy's L-BFGS-B that can leverage
    batch calculations with TensorFlow, this function uses the Greenlet package
    to run each individual optimization on micro-threads.

    L-BFGS-B updates for each individual optimization are performed by
    independent greenlets working with Numpy arrays, however, the evaluation
    of our acquisition function (and its gradients) is calculated in parallel
    (for each optimization step) using Tensorflow.

    For :class:'TaggedProductSearchSpace' we only apply gradient updates to
    its :class:'Box' subspaces, fixing the discrete elements to the best values
    found across the initial random search. To fix these discrete elements, we
    optimize over a continuous :class:'Box' relaxation of the discrete subspaces
    which has equal upper and lower bounds, i.e. we specify an equality constraint
    for this dimension in the scipy optimizer.

    This function also support the maximization of vectorized target functions (with
    vectorization V).

    :param target_func: The function(s) to maximise, with input shape [..., V, D] and
        output shape [..., V].
    :param space: The original search space.
    :param starting_points: The points at which to begin our optimizations of shape
        [num_optimization_runs, V, D]. The leading dimension of
        `starting_points` controls the number of individual optimization runs
        for each of the V target functions.
    :param perturb_grad_mask: A mask to zero out gradients of non-perturbed points to
        ensure that they don't get perturbed by gradient-based optimisation. Has same
        shape as `starting_points` i.e. [num_optimization_runs, V, D].
    :param optimizer_args: Keyword arguments to pass to the Scipy optimizer.
    :return: A tuple containing the failure statuses, maximum values, maximisers and
        number of evaluations for each of our optimizations.
    """

    tf_dtype = starting_points.dtype  # type for communication with Trieste

    num_optimization_runs_per_function = tf.shape(starting_points)[0].numpy()

    V = tf.shape(starting_points)[-2].numpy()  # vectorized batch size
    D = tf.shape(starting_points)[-1].numpy()  # search space dimension
    num_optimization_runs = num_optimization_runs_per_function * V

    vectorized_starting_points = tf.reshape(
        starting_points, [-1, D]
    )  # [num_optimization_runs*V, D]
    vectorized_perturb_grad_mask = tf.reshape(
        perturb_grad_mask, [-1, D]
    )  # [num_optimization_runs*V, D]

    def _objective_value(vectorized_x: TensorType) -> TensorType:  # [N, D] -> [N, 1]
        vectorized_x = vectorized_x[:, None, :]  # [N, 1, D]
        x = tf.reshape(vectorized_x, [-1, V, D])  # [N/V, V, D]
        evals = -target_func(x)  # [N/V, V]
        vectorized_evals = tf.reshape(evals, [-1, 1])  # [N, 1]
        return vectorized_evals

    def _objective_value_and_gradient(x: TensorType) -> Tuple[TensorType, TensorType]:
        val, gradient = tfp.math.value_and_gradient(
            _objective_value, x
        )  # [len(x), 1], [len(x), D]
        masked_gradient = vectorized_perturb_grad_mask * gradient
        return val, masked_gradient

    if isinstance(
        space, TaggedProductSearchSpace
    ):  # build continuous relaxation of discrete subspaces
        bounds = [
            get_bounds_of_box_relaxation_around_point(
                space, vectorized_starting_points[i : i + 1]
            )
            for i in tf.range(num_optimization_runs)
        ]
    else:
        bounds = [spo.Bounds(space.lower, space.upper)] * num_optimization_runs

    # Initialize the numpy arrays to be passed to the greenlets
    np_batch_x = np.zeros(
        (num_optimization_runs, tf.shape(starting_points)[-1]), dtype=np.float64
    )
    np_batch_y = np.zeros((num_optimization_runs,), dtype=np.float64)
    np_batch_dy_dx = np.zeros(
        (num_optimization_runs, tf.shape(starting_points)[-1]), dtype=np.float64
    )

    # Set up child greenlets
    child_greenlets = [ScipyOptimizerGreenlet() for _ in range(num_optimization_runs)]
    vectorized_child_results: List[
        Union[spo.OptimizeResult, "np.ndarray[Any, Any]"]
    ] = [
        gr.switch(
            vectorized_starting_points[i].numpy(),
            bounds[i],
            space.constraints,
            optimizer_args,
        )
        for i, gr in enumerate(child_greenlets)
    ]

    while True:
        all_done = True
        for i, result in enumerate(
            vectorized_child_results
        ):  # Process results from children.
            if isinstance(result, spo.OptimizeResult):
                continue  # children return a `spo.OptimizeResult` if they are finished
            all_done = False
            assert isinstance(
                result, np.ndarray
            )  # or an `np.ndarray` with the query `x` otherwise
            np_batch_x[i, :] = result

        if all_done:
            break

        # Batch evaluate query `x`s from all children.
        batch_x = tf.constant(np_batch_x, dtype=tf_dtype)  # [num_optimization_runs, d]
        batch_y, batch_dy_dx = _objective_value_and_gradient(batch_x)
        np_batch_y = batch_y.numpy().astype("float64")
        np_batch_dy_dx = batch_dy_dx.numpy().astype("float64")

        for i, greenlet in enumerate(
            child_greenlets
        ):  # Feed `y` and `dy_dx` back to children.
            if greenlet.dead:  # Allow for crashed greenlets
                continue
            vectorized_child_results[i] = greenlet.switch(
                np_batch_y[i], np_batch_dy_dx[i, :]
            )

    final_vectorized_child_results: List[spo.OptimizeResult] = vectorized_child_results
    vectorized_successes = tf.constant(
        [result.success for result in final_vectorized_child_results]
    )  # [num_optimization_runs]
    vectorized_fun_values = tf.constant(
        [-result.fun for result in final_vectorized_child_results], dtype=tf_dtype
    )  # [num_optimization_runs]
    vectorized_chosen_x = tf.constant(
        [result.x for result in final_vectorized_child_results], dtype=tf_dtype
    )  # [num_optimization_runs, D]
    vectorized_nfev = tf.constant(
        [result.nfev for result in final_vectorized_child_results], dtype=tf_dtype
    )

    # Ensure chosen points satisfy any constraints in the search-space.
    if space.has_constraints:
        is_feasible = space.is_feasible(vectorized_chosen_x)
        vectorized_successes = tf.logical_and(vectorized_successes, is_feasible)

    successes = tf.reshape(vectorized_successes, [-1, V])  # [num_optimization_runs, V]
    fun_values = tf.reshape(
        vectorized_fun_values, [-1, V]
    )  # [num_optimization_runs, V]
    chosen_x = tf.reshape(
        vectorized_chosen_x, [-1, V, D]
    )  # [num_optimization_runs, V, D]
    nfev = tf.reshape(vectorized_nfev, [-1, V])  # [num_optimization_runs, V]

    return (successes, fun_values, chosen_x, nfev)


class ScipyOptimizerGreenlet(gr.greenlet):  # type: ignore[misc]
    """
    Worker greenlet that runs a single Scipy L-BFGS-B (by default). Each greenlet performs all the
    optimizer update steps required for an individual optimization. However, the evaluation
    of our acquisition function (and its gradients) is delegated back to the main Tensorflow
    process (the parent greenlet) where evaluations can be made efficiently in parallel.
    """

    def run(
        self,
        start: "np.ndarray[Any, Any]",
        bounds: spo.Bounds,
        constraints: Sequence[Constraint],
        optimizer_args: Optional[dict[str, Any]] = None,
    ) -> spo.OptimizeResult:
        cache_x = start + 1  # Any value different from `start`.
        cache_y: Optional["np.ndarray[Any, Any]"] = None
        cache_dy_dx: Optional["np.ndarray[Any, Any]"] = None

        def value_and_gradient(
            x: "np.ndarray[Any, Any]",
        ) -> Tuple["np.ndarray[Any, Any]", "np.ndarray[Any, Any]"]:
            # Collect function evaluations from parent greenlet
            nonlocal cache_x
            nonlocal cache_y
            nonlocal cache_dy_dx

            if not (cache_x == x).all():
                cache_x[:] = x  # Copy the value of `x`. DO NOT copy the reference.
                # Send `x` to parent greenlet, which will evaluate all `x`s in a batch.
                cache_y, cache_dy_dx = self.parent.switch(cache_x)

            return cast("np.ndarray[Any, Any]", cache_y), cast(
                "np.ndarray[Any, Any]", cache_dy_dx
            )

        method = "trust-constr" if len(constraints) else "l-bfgs-b"
        optimizer_args = dict(
            dict(method=method, constraints=constraints), **(optimizer_args or {})
        )
        return spo.minimize(
            lambda x: value_and_gradient(x)[0],
            start,
            jac=lambda x: value_and_gradient(x)[1],
            bounds=bounds,
            **optimizer_args,
        )


def get_bounds_of_box_relaxation_around_point(
    space: TaggedProductSearchSpace, current_point: TensorType
) -> spo.Bounds:
    """
    A function to return the bounds of a continuous relaxation of
    a :class:'TaggedProductSearchSpace' space, i.e. replacing discrete
    spaces with continuous spaces. In particular, all :class:'DiscreteSearchSpace'
    subspaces are replaced with a new :class:'DiscreteSearchSpace' fixed at their
    respective component of the specified 'current_point'. Note that
    all :class:'Box' subspaces remain the same.

    :param space: The original search space.
    :param current_point: The point at which to make the continuous relaxation.
    :return: Bounds for the Scipy optimizer.
    """
    tf.debugging.Assert(isinstance(space, TaggedProductSearchSpace), [tf.constant([])])

    space_with_fixed_discrete = space
    for tag in space.subspace_tags:
        if isinstance(
            space.get_subspace(tag), DiscreteSearchSpace
        ):  # convert discrete subspaces to box spaces.
            subspace_value = space.get_subspace_component(tag, current_point)
            space_with_fixed_discrete = space_with_fixed_discrete.fix_subspace(
                tag, subspace_value
            )
    return spo.Bounds(space_with_fixed_discrete.lower, space_with_fixed_discrete.upper)


def batchify_joint(
    batch_size_one_optimizer: AcquisitionOptimizer[SearchSpaceType],
    batch_size: int,
) -> AcquisitionOptimizer[SearchSpaceType]:
    """
    A wrapper around our :const:`AcquisitionOptimizer`s. This class wraps a
    :const:`AcquisitionOptimizer` to allow it to jointly optimize the batch elements considered
    by a batch acquisition function.

    :param batch_size_one_optimizer: An optimizer that returns only batch size one, i.e. produces a
            single point with shape [1, D].
    :param batch_size: The number of points in the batch.
    :return: An :const:`AcquisitionOptimizer` that will provide a batch of points with shape [B, D].
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    def optimizer(
        search_space: SearchSpaceType,
        f: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
    ) -> TensorType:
        expanded_search_space = search_space**batch_size  # points have shape [B * D]

        if isinstance(f, tuple):
            raise ValueError(
                "batchify_joint cannot be applied to a vectorized acquisition function"
            )
        af: AcquisitionFunction = f  # type checking can get confused by closure of f

        def target_func_with_vectorized_inputs(
            x: TensorType,
        ) -> TensorType:  # [..., 1, B * D] -> [..., 1]
            return af(tf.reshape(x, x.shape[:-2].as_list() + [batch_size, -1]))

        vectorized_points = batch_size_one_optimizer(  # [1, B * D]
            expanded_search_space, target_func_with_vectorized_inputs
        )
        return tf.reshape(vectorized_points, [batch_size, -1])  # [B, D]

    return optimizer


def batchify_vectorize(
    batch_size_one_optimizer: AcquisitionOptimizer[SearchSpaceType],
    batch_size: int,
) -> AcquisitionOptimizer[SearchSpaceType]:
    """
    A wrapper around our :const:`AcquisitionOptimizer`s. This class wraps a
    :const:`AcquisitionOptimizer` to allow it to optimize batch acquisition functions.

    Unlike :func:`batchify_joint`, :func:`batchify_vectorize` is suitable
    for a :class:`AcquisitionFunction` whose individual batch element can be
    optimized independently (i.e. they can be vectorized).

    :param batch_size_one_optimizer: An optimizer that returns only batch size one, i.e. produces a
            single point with shape [1, D].
    :param batch_size: The number of points in the batch.
    :return: An :const:`AcquisitionOptimizer` that will provide a batch of points with shape [V, D].
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    def optimizer(
        search_space: SearchSpaceType,
        f: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
    ) -> TensorType:
        if isinstance(f, tuple):
            raise ValueError(
                "batchify_vectorize cannot be applied to an already vectorized acquisition function"
            )

        return batch_size_one_optimizer(search_space, (f, batch_size))

    return optimizer


def constrained_turbo_batchify_vectorize(
    batch_size_one_optimizer: ConstrainedTrustRegionAcquisitionOptimizer[
        SearchSpaceType
    ],
    batch_size: int,
) -> ConstrainedTrustRegionAcquisitionOptimizer[SearchSpaceType]:
    """
    Adjusted version of :func:`batchify_vectorize` for use with the constrained trust
    region acquisition optimizer, which takes two additional arguments, `x_min` and
    `perturbation_prob`.

    A wrapper around our :const:`AcquisitionOptimizer`s. This class wraps a
    :const:`AcquisitionOptimizer` to allow it to optimize batch acquisition functions.

    Unlike :func:`batchify_joint`, :func:`batchify_vectorize` is suitable
    for a :class:`AcquisitionFunction` whose individual batch element can be
    optimized independently (i.e. they can be vectorized).

    :param batch_size_one_optimizer: An optimizer that returns only batch size one, i.e. produces a
            single point with shape [1, D].
    :param batch_size: The number of points in the batch.
    :return: An :const:`AcquisitionOptimizer` that will provide a batch of points with shape [V, D].
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    def optimizer(
        search_space: SearchSpaceType,
        f: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
        x_min: TensorType,
        perturbation_prob: float,
    ) -> TensorType:
        if isinstance(f, tuple):
            raise ValueError(
                "batchify_vectorize cannot be applied to an already vectorized acquisition function"
            )

        return batch_size_one_optimizer(
            search_space,
            (f, batch_size),
            x_min=x_min,
            perturbation_prob=perturbation_prob,
        )

    return optimizer

# NOTE: Only to be used with `get_max_discrete_points`
def constrained_turbo_batchify_joint(
    batch_size_one_optimizer: ConstrainedTrustRegionAcquisitionOptimizer[SearchSpaceType],
    batch_size: int,
) -> ConstrainedTrustRegionAcquisitionOptimizer[SearchSpaceType]:
    """
    Adjusted version of :func:`batchify_joint` for use with the constrained trust
    region acquisition optimizer, which takes two additional arguments, `x_min` and
    `perturbation_prob`.

    A wrapper around our :const:`AcquisitionOptimizer`s. This class wraps a
    :const:`AcquisitionOptimizer` to allow it to jointly optimize the batch elements considered
    by a batch acquisition function.

    :param batch_size_one_optimizer: An optimizer that returns only batch size one, i.e. produces a
            single point with shape [1, D].
    :param batch_size: The number of points in the batch.
    :return: An :const:`AcquisitionOptimizer` that will provide a batch of points with shape [B, D].
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    def optimizer(
        search_space: SearchSpaceType,
        f: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
        x_min: TensorType,
        perturbation_prob: float,
    ) -> TensorType:
        expanded_search_space = search_space**batch_size  # points have shape [B * D]

        if isinstance(f, tuple):
            raise ValueError(
                "batchify_joint cannot be applied to a vectorized acquisition function"
            )
        af: AcquisitionFunction = f  # type checking can get confused by closure of f

        def target_func_with_vectorized_inputs(
            x: TensorType,
        ) -> TensorType:  # [..., 1, B * D] -> [..., 1]
            return af(tf.reshape(x, x.shape[:-2].as_list() + [batch_size, -1]))

        vectorized_points = batch_size_one_optimizer(  # [1, B * D]
            expanded_search_space, target_func_with_vectorized_inputs, x_min, perturbation_prob, True, batch_size
        )
        return tf.reshape(vectorized_points, [batch_size, -1])  # [B, D]

    return optimizer


def generate_adam_optimizer(
    num_initial_samples: int = NUM_SAMPLES_MIN,
) -> ConstrainedTrustRegionAcquisitionOptimizer[Box]:
    """
    Uses Tensorflow's Adam optimizer to optimize the acquisition function. Box
    constraints are enforced by heavily penalising points outside the box in the loss
    function being minimised.

    We advise the user to either use the default `NUM_SAMPLES_MIN` for `num_initial_samples`, or
    `NUM_SAMPLES_DIM` times the dimensionality of the search space, whichever is greater.
    Similarly, for `num_optimization_runs`, we recommend using `NUM_RUNS_DIM` times the
    dimensionality of the search space.

    :param num_initial_samples: The size of the random sample used to find the starting point(s) of
        the optimization.
    :return: The acquisition optimizer.
    """

    def al_adam_optimize_continuous(
        space: Box | TaggedProductSearchSpace,
        target_func: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
        x_min: Optional[TensorType] = None,
        perturbation_prob: Optional[float] = None,
    ) -> TensorType:
        """
        A gradient-based :const:`ConstrainedTrustRegionAcquisitionOptimizer` for
        :class:'Box' spaces. This uses Tensorflow's Adam optimizer the optimize the
        acquisition function. Box constraints are enforced by heavily penalising points
        outside the box in the loss function being minimised. It supports TURBO-based
        optimization by allowing the user to specify a perturbation probability and a
        minimum point found so far, more details of which can be found in Eriksson et
        al. 2019 (https://arxiv.org/pdf/1910.01739.pdf).

        :param space: The space over which to search.
        :param target_func: The function to maximise, with input shape [..., V, D] and output shape
                [..., V].
        :x_min: Location of best point found in the search space so far.
        :perturbation_prob: Probability of perturbing a dimension of the
        best value found so far. Eriksson et al. 2019
        (https://arxiv.org/pdf/1910.01739.pdf) reason that once we're in
        high-dimensional spaces we shouldn't perturb all dimensions of the best
        discovered point so far when choosing a new point to query.
        :return: The V points in ``space`` that maximises``target_func``, with shape [V, D].
        """

        if isinstance(target_func, tuple):  # check if we need a vectorized optimizer
            target_func, V = target_func
        else:
            V = 1

        candidates = space.sample(num_initial_samples)[
            :, None, :
        ]  # [num_initial_samples, 1, D]

        if x_min is None and perturbation_prob is None:
            # If we havent't set a `perturbation_prob` value (as will be the case if we aren't using a
            # trust-region based approach) then we set it to 1.0 so that all dimensions
            # of the best value found so far are perturbed.
            x_min = tf.zeros(shape=(1, 1, space.dimension), dtype=tf.float64)
            perturbation_prob = 1.0
        else:
            tf.debugging.assert_rank(x_min, 1)
            x_min = x_min[None, None, ...]

        perturb = tf.cast(
            tfp.distributions.Bernoulli(probs=perturbation_prob).sample(
                sample_shape=candidates.shape
            ),
            tf.bool,
        )
        candidates = tf.where(perturb, candidates, x_min)
        tiled_candidates = tf.tile(candidates, [1, V, 1])  # [num_initial_samples, V, D]
        tiled_perturb = tf.tile(perturb, [1, V, 1])  # [num_initial_samples, V, D]

        target_func_values = target_func(tiled_candidates)  # [num_samples, V]
        tf.debugging.assert_shapes(
            [(target_func_values, ("_", V))],
            message=(
                f"""
                The result of function target_func has shape
                {tf.shape(target_func_values)}, however, expected a trailing
                dimension of size {V}.
                """
            ),
        )

        _, top_k_indices = tf.math.top_k(
            tf.transpose(target_func_values), k=1
        )  # [1, num_optimization_runs] or [V, num_optimization_runs]

        tiled_candidates = tf.transpose(
            tiled_candidates, [1, 0, 2]
        )  # [V, num_initial_samples, D]
        tiled_perturb = tf.transpose(tiled_perturb, [1, 0, 2])  # [V, num_initial_samples, D]
        top_k_points = tf.gather(
            tiled_candidates, top_k_indices, batch_dims=1
        )  # [V, 1, D]
        top_k_tiled_perturb = tf.gather(tiled_perturb, top_k_indices, batch_dims=1)  # [V, 1, D]
        initial_points = tf.transpose(top_k_points, [1, 0, 2])  # [1, V, D]
        initial_points_perturb = tf.transpose(top_k_tiled_perturb, [1, 0, 2])  # [1, V, D]
        # Use this mask to zero out gradients of non-perturbed points to ensure that
        # they don't get perturbed by gradient-based optimisation.
        perturb_grad_mask = tf.cast(initial_points_perturb, tf.float64)

        lower_lim = space.lower[None, None, ...]
        upper_lim = space.upper[None, None, ...]

        optimised_points = tf.Variable(
            initial_points, name="optimised_points", trainable=True, dtype=tf.float64
        )
        adam_queried_locations = optimised_points

        # Form loss function to be *minimised*
        def loss_fun(opt_points):
            # Since Adam doesn't support constrained optimisation, we add a penalty term
            # to the loss function to penalise points that are outside the bounds of the
            # search space.
            x_beyond_upper_lim = tf.nn.relu(opt_points - upper_lim)  # [1, B, D]
            x_beyond_lower_lim = tf.nn.relu(lower_lim - opt_points)  # [1, B, D]
            x_beyond_upper_lim = tf.reduce_sum(x_beyond_upper_lim, axis=-1)  # [1, B]
            x_beyond_lower_lim = tf.reduce_sum(x_beyond_lower_lim, axis=-1)  # [1, B]
            return -target_func(opt_points) + 1e10 * (
                x_beyond_upper_lim + x_beyond_lower_lim
            )  # We're trying to *maximise* 'target_func', but Adam *minimises* it

        opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1000)
        for i in range(150): 
            with tf.GradientTape() as tape:
                tape.watch(optimised_points)
                y = loss_fun(optimised_points)
                adam_queried_locations = tf.concat(
                    [adam_queried_locations, optimised_points], axis=0
                )
                grad = tape.gradient(y, optimised_points)
                # Mask gradient so that dimensions which shouldn't be perturbed aren't.
                masked_grad = grad * perturb_grad_mask
                opt.apply_gradients([(masked_grad, optimised_points)])

        clipped_adam_queried_locations = tf.clip_by_value(
            adam_queried_locations, clip_value_min=lower_lim, clip_value_max=upper_lim
        )
        adam_queried_values = target_func(clipped_adam_queried_locations)
        _, clipped_adam_queried_locations_top_k_indices = tf.math.top_k(
            tf.transpose(adam_queried_values), k=1
        )  # [V, num_optimization_runs]
        clipped_adam_queried_locations = tf.transpose(
            clipped_adam_queried_locations, [1, 0, 2]
        )  # [V, num_initial_samples, D]
        best_batch_adam_queried_locations = tf.gather(
            clipped_adam_queried_locations,
            clipped_adam_queried_locations_top_k_indices,
            batch_dims=1,
        )  # [V, 1, D]
        best_batch_adam_queried_locations = tf.transpose(
            best_batch_adam_queried_locations, [1, 0, 2]
        )  # [1, V, D]

        # We're trying to *maximise* 'target_func', so check to ensure that
        # gradient-based optimiztion has increased the value of 'target_func'.
        optimised_vals = target_func(best_batch_adam_queried_locations)
        initial_vals = target_func(initial_points)
        assert tf.reduce_all(optimised_vals >= initial_vals)
        assert tf.reduce_all(best_batch_adam_queried_locations >= lower_lim)
        assert tf.reduce_all(best_batch_adam_queried_locations <= upper_lim)
        return best_batch_adam_queried_locations[0]

    return al_adam_optimize_continuous


def generate_random_search_optimizer(
    num_samples: int = NUM_SAMPLES_MIN,
) -> ConstrainedTrustRegionAcquisitionOptimizer[SearchSpace]:
    """
    Generate an acquisition optimizer that samples `num_samples` random points across the space.
    The default is to sample at `NUM_SAMPLES_MIN` locations.

    We advise the user to either use the default `NUM_SAMPLES_MIN` for `num_samples`, or
    `NUM_SAMPLES_DIM` times the dimensionality of the search space, whichever is smaller.

    :param num_samples: The number of random points to sample.
    :return: The acquisition optimizer.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    def optimize_random(
        space: SearchSpace,
        target_func: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
        x_min=None,
        perturbation_prob=None,
        batchify_joint: bool = False,
        batch_size: Optional[int] = None,
    ) -> TensorType:
        """
        A random search :const:`AcquisitionOptimizer` defined for
        any :class:'SearchSpace' with a :meth:`sample`. If we have a :class:'DiscreteSearchSpace'
        with fewer than `num_samples` points, then we query all the points in the space.

        When this functions receives an acquisition-integer tuple as its `target_func`,it
        optimizes each of the individual V functions making up `target_func`, i.e.
        evaluating `num_samples` samples for each of the individual V functions making up
        target_func.

        :param space: The space over which to search.
        :param target_func: The function to maximise, with input shape [..., V, D] and output shape
                [..., V].
        :x_min: Location of best point found in the search space so far.
        :perturbation_prob: Probability of perturbing a dimension of the
        best value found so far. Eriksson et al. 2019
        (https://arxiv.org/pdf/1910.01739.pdf) reason that once we're in
        high-dimensional spaces we shouldn't perturb all dimensions of the best
        discovered point so far when choosing a new point to query.
        :batchify_joint: Whether to use :func:`batchify_joint` to jointly optimize the
        batch
        :batch_size: The number of points in the batch (needed if jointly optimizing)
        :return: The V points in ``space`` that maximises ``target_func``, with shape [V, D].
        """
        points = space.sample(num_samples)[:, None, :]
        return _get_max_discrete_points(points, target_func, x_min, perturbation_prob, batchify_joint, batch_size)

    return optimize_random


def generate_sobol_random_search_optimizer(
    num_samples: int = NUM_SAMPLES_MIN,
) -> AcquisitionOptimizer[Box]:
    """
    Generate an acquisition optimizer that samples `num_samples` random points across the space using sobol indices.
    The default is to sample at `NUM_SAMPLES_MIN` locations.

    We advise the user to either use the default `NUM_SAMPLES_MIN` for `num_samples`, or
    `NUM_SAMPLES_DIM` times the dimensionality of the search space, whichever is greater.

    :param num_samples: The number of random points to sample.
    :return: The acquisition optimizer.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    def optimize_sobol_random(
        space: Box,
        target_func: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
        x_min=None,
        perturbation_prob=None,
    ) -> TensorType:
        """
        A random search :const:`AcquisitionOptimizer` defined for
        a `Box` search space using Sobol indices.

        When this functions receives an acquisition-integer tuple as its `target_func`,it
        optimizes each of the individual V functions making up `target_func`, i.e.
        evaluating `num_samples` samples for each of the individual V functions making up
        target_func.

        :param space: The space over which to search.
        :param target_func: The function to maximise, with input shape [..., V, D] and output shape
                [..., V].
        :x_min: Location of best point found in the search space so far.
        :perturbation_prob: Probability of perturbing a dimension of the
        best value found so far. Eriksson et al. 2019
        (https://arxiv.org/pdf/1910.01739.pdf) reason that once we're in
        high-dimensional spaces we shouldn't perturb all dimensions of the best
        discovered point so far when choosing a new point to query.
        :return: The V points in ``space`` that maximises ``target_func``, with shape [V, D].
        """
        points = space.sample_sobol(num_samples)[:, None, :]  # [..., 1, D]
        return _get_max_discrete_points(points, target_func, x_min, perturbation_prob)

    return optimize_sobol_random
