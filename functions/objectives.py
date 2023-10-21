import math
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

def ackley_10(x: tf.Tensor) -> tf.Tensor:
    """
    The Ackley test function over :math:`[0, 1]^10`. This function has
    many local minima and a global minima. See https://www.sfu.ca/~ssurjano/ackley.html
    for details. This is based on the problem presented in
    https://arxiv.org/pdf/2002.08526.pdf, which evaluates the function over the domain
    [-5, 10].

    :param x: The points at which to evaluate the function, with shape [N, 10].
    :return: The function values at ``x``, with shape [N, 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    x = -5.0 + 15.0 * x 

    exponent_1 = -0.2 * tf.math.sqrt((1 / 10.0) * tf.reduce_sum(x**2, -1))
    exponent_2 = (1 / 10.0) * tf.reduce_sum(tf.math.cos(2.0 * math.pi * x), -1)

    objective = tf.expand_dims((
        -20.0 * tf.math.exp(exponent_1)
        - tf.math.exp(exponent_2)
        + 20.0
        + tf.cast(tf.math.exp(1.0), dtype=x.dtype)
    ), -1) / 10.0
    tf.debugging.assert_rank(objective, 2)
    return objective

def keane_bump_30(x: tf.Tensor) -> tf.Tensor:
    """
    The Keane bump test function over :math:`[0, 1]^30`. Taken from 
    https://arxiv.org/pdf/2002.08526.pdf.

    :param x: The points at which to evaluate the function, with shape [N, 30].
    :return The function values at ``x``, with shape [N, 1].
    """
    x = 10.0 * x
    numerator = tf.reduce_sum(tf.math.cos(x) ** 4, axis=-1, keepdims=True) - 2.0 * tf.reduce_prod(tf.math.cos(x) ** 2, axis=-1, keepdims=True)
    denominator = tf.math.sqrt(tf.reduce_sum(tf.range(1, 31, dtype=x.dtype) * (x ** 2), axis=-1, keepdims=True)) 
    objective = (-1.0 * tf.abs(numerator / denominator)) / 0.1
    tf.debugging.assert_rank(objective, 2)
    return objective