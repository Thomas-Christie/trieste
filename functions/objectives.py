import tensorflow as tf


def linear_objective(x: tf.Tensor) -> tf.Tensor:
    """
    :param x: locations to evaluate objective at of shape [N, 2]
    :return: objective values at locations "x" with shape [N, 1]
    """
    x0 = x[..., :1]
    x1 = x[..., 1:]
    objective = tf.cast(x0 + x1, tf.float64)
    tf.debugging.assert_rank(objective, 2)
    return objective


def goldstein_price(x: tf.Tensor) -> tf.Tensor:
    """
    :param x: locations to evaluate objective at of shape [N, 2]
    :return: objective values at locations "x" with shape [N, 1]
    Rescaled and centered Goldstein-Price function from https://arxiv.org/pdf/1605.09466.pdf
    """
    x0 = x[..., :1]
    x1 = x[..., 1:]
    a_one = tf.square(4 * x0 + 4 * x1 - 3)
    a_two = 75 - 56 * (x0 + x1) + 3 * tf.square(4 * x0 - 2) + 6 * (4 * x0 - 2) * (4 * x1 - 2) + 3 * tf.square(
        4 * x1 - 2)
    a = a_one * a_two
    b_one = tf.square(8 * x0 - 12 * x1 + 2)
    b_two = - 14 - 128 * x0 + 12 * tf.square(4 * x0 - 2) + 192 * x1 - 36 * (4 * x0 - 2) * (4 * x1 - 2) + 27 * tf.square(
        4 * x1 - 2)
    b = b_one * b_two
    objective = (tf.math.log((1 + a) * (30 + b)) - 8.6928) / 2.4269
    tf.debugging.assert_rank(objective, 2)
    return objective


def lockwood_objective_trajectory(x: tf.Tensor) -> tf.Tensor:
    """
    Behaves like a trajectory in the case that we're modelling a known objective function for the Lockwood problem.
    :param x: locations to evaluate objective at of shape [N, B, D]
    :return: objective values at location "x" with shape [N, B, 1]
    """
    return tf.reduce_sum(x, axis=2, keepdims=True)