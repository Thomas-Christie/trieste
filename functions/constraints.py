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

def centered_branin(x: tf.Tensor) -> tf.Tensor:
    """
    :param x: locations to evaluate constraint at of shape [N, 2]
    :return: constraint values at locations "x" with shape [N, 1]
    From https://arxiv.org/pdf/1605.09466.pdf
    """
    x0 = x[..., :1]
    x1 = x[..., 1:]
    constraint = (15 - tf.square(15 * x1  - (5/(4 * (pi ** 2))) * tf.square(15 * x0 - 5) + (5/pi) * (15 * x0 - 5) - 6) - 10 * (1 - 1/(8*pi)) * tf.math.cos(15 * x0 - 5)) / 100
    tf.debugging.assert_rank(constraint, 2)
    return constraint

def parr_constraint(x: tf.Tensor) -> tf.Tensor:
    """
    :param x: locations to evaluate constraint at of shape [N, 2]
    :return: constraint values at locations "x" with shape [N, 1]
    From https://arxiv.org/pdf/1605.09466.pdf
    """
    x0 = x[..., :1]
    x1 = x[..., 1:]
    subsection = (4 - 2.1 * tf.square(2 * x0 - 1) + tf.pow(2 * x0 - 1, 4)/3) * tf.square(2 * x0 - 1)
    constraint = (4 - subsection - (2 * x0 - 1) * (2 * x1 - 1) - 16 * (tf.square(x1) - x1) * tf.square(2 * x1 - 1) - 3 * tf.math.sin(12 * (1 - x0)) - 3 * tf.math.sin(12 * (1 - x1))) / 10
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

