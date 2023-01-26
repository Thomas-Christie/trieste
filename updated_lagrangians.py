import tensorflow as tf
from typing import Callable, List, Optional


def augmented_lagrangian(objective: Callable,
                         inequality_constraints: Optional[List[Callable]],
                         equality_constraints: Optional[List[Callable]],
                         equality_lambd: Optional[tf.Tensor],
                         inequality_lambd: Optional[tf.Tensor],
                         penalty: tf.Variable,
                         x: tf.Tensor) -> tf.Tensor:
    """
    Form augmented Lagrangian from given objective and constraints
    :param objective: Objective function
    :param equality_constraints: List of equality constraint functions
    :param inequality_constraints: List of inequality constraint functions
    :param equality_lambd: Array of Lagrange multipliers for equality constraints of shape ["num_eq_constraints", 1]
    :param inequality_lambd: Array of Lagrange multipliers for inequality constraints of shape ["num_ineq_constraints", 1]
    :param penalty: Penalty parameter for violated constraints
    :param x: Array of points at which to evaluate the lagrangian of shape [N, 1, M]
    :return:
    """
    objective_vals = objective(x)
    objective_vals = tf.squeeze(objective_vals, -2)
    if len(equality_constraints) > 0:
        equality_constraint_vals = tf.stack([constraint(x) for constraint in equality_constraints])
        equality_lambd_scaled = tf.reshape(tf.einsum("ij,ikj->k", equality_lambd, equality_constraint_vals), [-1, 1])
        equality_penalty_scaled = (1 / (2 * penalty)) * tf.reduce_sum(tf.square(equality_constraint_vals), axis=0)

    else:
        equality_lambd_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        equality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)

    if len(inequality_constraints) > 0:
        inequality_constraint_vals = tf.stack([constraint(x) for constraint in inequality_constraints])
        inequality_constraint_vals = tf.squeeze(inequality_constraint_vals, -2)
        slack_vals = tf.stack([obtain_slacks(inequality_constraints[i], x, inequality_lambd[i], penalty) for i in range(len(inequality_constraints))])
        assert(slack_vals.shape == inequality_constraint_vals.shape)
        inequality_plus_slack = inequality_constraint_vals + slack_vals
        inequality_lambd_scaled = tf.reshape(tf.einsum("ij,ikj->k", inequality_lambd, inequality_plus_slack), [-1, 1])
        inequality_penalty_scaled = (1 / (2 * penalty)) * tf.reduce_sum(tf.square(inequality_plus_slack), axis=0)
    else:
        inequality_lambd_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
        inequality_penalty_scaled = tf.zeros(objective_vals.shape, dtype=tf.float64)
    return - objective_vals - equality_lambd_scaled - inequality_lambd_scaled - equality_penalty_scaled - inequality_penalty_scaled


def obtain_slacks(inequality_constraint: Callable,
                  x: tf.Tensor,
                  inequality_lambd: tf.Tensor,
                  penalty: tf.Variable) -> tf.Tensor:
    """
    Obtain optimal slack values for augmented Lagrangian
    :param inequality_constraint: Inequality constraint function to find slack values for
    :param x: Input values at which slack values should be obtained of shape [N, M]
    :param inequality_lambd: Lagrangian multiplier for given inequality constraint
    :param penalty: Penalty for constraint violation
    :return: Optimal slack values at each x location, of shape [N, 1]
    """
    slack_vals = - inequality_lambd[0] * penalty - inequality_constraint(x)
    slack_vals = tf.squeeze(slack_vals, -2)
    slack_vals_non_neg = tf.nn.relu(slack_vals)
    tf.debugging.assert_shapes([(slack_vals_non_neg, (..., 1))])
    return slack_vals_non_neg


def obtain_single_slack(inequality_constraint_val: tf.Tensor,
                        inequality_lambd: tf.Tensor,
                        penalty: tf.Variable) -> tf.Tensor:
    """
    Obtain optimal slack values for augmented Lagrangian
    :param inequality_constraint: Inequality constraint function to find slack values for
    :param x: Input values at which slack values should be obtained of shape [N, M]
    :param inequality_lambd: Lagrangian multiplier for given inequality constraint
    :param penalty: Penalty for constraint violation
    :return: Optimal slack values at each x location, of shape [N, 1]
    """
    slack_vals = - inequality_lambd[0] * penalty - inequality_constraint_val
    slack_vals_non_neg = tf.nn.relu(slack_vals)
    tf.debugging.assert_shapes([(slack_vals_non_neg, (..., 1))])
    return slack_vals_non_neg
