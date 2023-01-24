import numpy as np
import tensorflow as tf
import trieste
from trieste.objectives import Hartmann6
from trieste.types import TensorType
from trieste.models.gpflow import (
    SparseVariational,
    build_svgp,
    KMeansInducingPointSelector,
)
from trieste.models.optimizer import BatchOptimizer
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition import ParallelContinuousThompsonSampling
from trieste.acquisition.optimizer import automatic_optimizer_selector
from trieste.acquisition.utils import split_acquisition_function_calls

tf.config.run_functions_eagerly(True)

hartmann_6 = Hartmann6.objective
search_space = Hartmann6.search_space

def noisy_hartmann_6(
    x: TensorType,
) -> TensorType:  # contaminate observations with Gaussian noise
    return hartmann_6(x) + tf.random.normal([len(x), 1], 0, 1, tf.float64)


if __name__ == "__main__":
    np.random.seed(1793)
    tf.random.set_seed(1793)


    num_initial_data_points = 15
    initial_query_points = search_space.sample(num_initial_data_points)
    observer = trieste.objectives.utils.mk_observer(noisy_hartmann_6)
    initial_data = observer(initial_query_points)

    gpflow_model = build_svgp(
        initial_data, search_space, likelihood_variance=0.01, num_inducing_points=50
    )

    inducing_point_selector = KMeansInducingPointSelector(search_space)

    model = SparseVariational(
        gpflow_model,
        num_rff_features=1_000,
        inducing_point_selector=inducing_point_selector,
        optimizer=BatchOptimizer(
            tf.optimizers.Adam(0.1), max_iter=100, batch_size=50, compile=True
        ),
    )

    num_query_points = 10

    acq_rule = EfficientGlobalOptimization(
        builder=ParallelContinuousThompsonSampling(),
        num_query_points=num_query_points,
        optimizer=split_acquisition_function_calls(
            automatic_optimizer_selector, split_size=100_000
        ),
    )

    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

    num_steps = 5
    result = bo.optimize(
        num_steps, initial_data, model, acq_rule, track_state=False
    )
    dataset = result.try_get_final_dataset()