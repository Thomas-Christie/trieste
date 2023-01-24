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

def dual_ascent_test_objective(x: tf.Tensor) -> tf.Tensor:
    """
    :param x: locations to evaluate objective at of shape [N, 2]
    :return: objective values at locations "x" with shape [N, 1]
    """
    # Taken from https://www.cis.upenn.edu/~cis5150/ws-book-IIb.pdf
    x0 = x[..., :1]
    x1 = x[..., 1:]
    objective =  tf.cast(0.5 * (x0 ** 2 + x1 ** 2), tf.float64)
    tf.debugging.assert_rank(objective, 2)
    return objective
