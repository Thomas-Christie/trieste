from math import pi
import tensorflow as tf

def toy_constraint_one(x: tf.Tensor) -> tf.Tensor:
    """
    :param x: locations to evaluate constraint at of shape [N, 2]
    :return: constraint values at locations "x" with shape [N, 1]
    """
    x0 = x[..., :1]
    x1 = x[..., 1:]
    constraint = tf.cast(1.5 - x0 - 2*x1 - 0.5 * tf.sin(2 * pi * (x0 ** 2 - 2 * x1)), tf.float64)
    tf.debugging.assert_rank(constraint, 2)
    return constraint

def toy_constraint_two(x: tf.Tensor) -> tf.Tensor:
    """
    :param x: locations to evaluate constraint at of shape [N, 2]
    :return: constraint values at locations "x" with shape [N, 1]
    """
    x0 = x[..., :1]
    x1 = x[..., 1:]
    constraint = tf.cast(x0 ** 2 + x1 ** 2 - 1.5, tf.float64)
    tf.debugging.assert_rank(constraint, 2)
    return constraint

def dual_ascent_test_constraint(x: tf.Tensor) -> tf.Tensor:
    """
    :param x: locations to evaluate constraint at of shape [N, 2]
    :return: constraint values at locations "x" with shape [N, 1]
    """
    # Taken from https://www.cis.upenn.edu/~cis5150/ws-book-IIb.pdf
    x0 = x[..., :1]
    x1 = x[..., 1:]
    constraint = tf.cast(2 * x0 - x1 - 5, tf.float64)
    tf.debugging.assert_rank(constraint, 2)
    return constraint

