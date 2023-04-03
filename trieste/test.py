import tensorflow as tf
import trieste
from trieste.acquisition.optimizer import generate_continuous_optimizer, generate_al_continuous_optimizer
from trieste.acquisition.function.new_constrained_thompson_sampling import BatchThompsonSamplingAugmentedLagrangian
from trieste.acquisition.rule import ALEfficientGlobalOptimization, EfficientGlobalOptimization
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.space import Box
from functions import constraints
from functions import objectives
import pickle

NUM_INITIAL_SAMPLES = 5
BATCH_SIZE = 1
NUM_BO_ITERS = 50
OBJECTIVE = "OBJECTIVE"
INEQUALITY_CONSTRAINT_ONE = "INEQUALITY_CONSTRAINT_ONE"
INEQUALITY_CONSTRAINT_TWO = "INEQUALITY_CONSTRAINT_TWO"
EQUALITY_CONSTRAINT_ONE = "EQUALITY_CONSTRAINT_ONE"
EQUALITY_CONSTRAINT_TWO = "EQUALITY_CONSTRAINT_TWO"
EPSILON = 0.001


def create_model(data):
    gpr = build_gpr(data, search_space, likelihood_variance=1e-7)
    return GaussianProcessRegression(gpr, num_rff_features=500)


# Original Toy Problem
if __name__ == "__main__":
    for run in range(1):
        search_space = Box([0.0, 0.0], [1.0, 1.0])
        observer = trieste.objectives.utils.mk_multi_observer(
            OBJECTIVE=objectives.linear_objective,
            INEQUALITY_CONSTRAINT_ONE=constraints.toy_constraint_one,
            INEQUALITY_CONSTRAINT_TWO=constraints.toy_constraint_two)

        initial_inputs = search_space.sample(NUM_INITIAL_SAMPLES)
        initial_data = observer(initial_inputs)
        initial_models = trieste.utils.map_values(create_model, initial_data)

        inequality_lambda = {INEQUALITY_CONSTRAINT_ONE: tf.Variable([[[0.0]]], dtype=tf.float64),
                             INEQUALITY_CONSTRAINT_TWO: tf.Variable([[[0.0]]], dtype=tf.float64)}
        initial_penalty = tf.Variable([[[0.5]]], dtype=tf.float64)

        # save_path = f"../results/30-03-23/lsq/data/run_{run}"
        augmented_lagrangian = BatchThompsonSamplingAugmentedLagrangian(OBJECTIVE, "INEQUALITY", None, inequality_lambda,
                                                                        None, BATCH_SIZE, initial_penalty, EPSILON, True, search_space,
                                                                        False, save_lambda=False, num_bo_iters=NUM_BO_ITERS)

        # rule = EfficientGlobalOptimization(augmented_lagrangian, optimizer=generate_continuous_optimizer(),
        #                                    num_query_points=BATCH_SIZE)
        rule = ALEfficientGlobalOptimization(augmented_lagrangian, optimizer=generate_al_continuous_optimizer(),
                                             num_query_points=BATCH_SIZE)
        bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
        data = bo.optimize(NUM_BO_ITERS, initial_data, initial_models, rule, track_state=True).try_get_final_datasets()
        # with open(f"../results/30-03-23/lsq/data/run_{run}_data.pkl", "wb") as fp:
        #     pickle.dump(data, fp)

# Harder problem with Goldstein-Price Objective
# if __name__ == "__main__":
#     search_space = Box([0.0, 0.0], [1.0, 1.0])
#     observer = trieste.objectives.utils.mk_multi_observer(
#         OBJECTIVE=objectives.goldstein_price,
#         INEQUALITY_CONSTRAINT_ONE=constraints.toy_constraint_one,
#         EQUALITY_CONSTRAINT_ONE=constraints.centered_branin,
#         EQUALITY_CONSTRAINT_TWO=constraints.parr_constraint)
#
#     initial_inputs = search_space.sample(NUM_INITIAL_SAMPLES)
#     initial_data = observer(initial_inputs)
#
#     # constraint_one_satisfied = (tf.squeeze(initial_data[INEQUALITY_CONSTRAINT_ONE].observations) <= 0)
#     # constraint_two_satisfied = (tf.abs(tf.squeeze(initial_data[EQUALITY_CONSTRAINT_ONE].observations)) <= 0 + EPSILON)
#     # constraint_three_satisfied = (tf.abs(tf.squeeze(initial_data[EQUALITY_CONSTRAINT_TWO].observations)) <= 0 + EPSILON)
#     # all_satisfied = tf.logical_and(constraint_one_satisfied, constraint_two_satisfied)
#     # all_satisfied = tf.logical_and(all_satisfied, constraint_three_satisfied)
#     # at_least_one_violated = tf.logical_not(all_satisfied)
#     #
#     # initial_penalty = None
#     # if tf.reduce_sum(tf.cast(at_least_one_violated, tf.int32)) == 0:
#     #     # If valid everywhere, set initial penalty to one
#     #     initial_penalty = 1
#     # else:
#     #     invalid_ineq_one_squared = tf.square(tf.nn.relu(
#     #         tf.squeeze(initial_data[INEQUALITY_CONSTRAINT_ONE].observations)[at_least_one_violated]))
#     #     invalid_eq_one_squared = tf.square(
#     #         tf.squeeze(initial_data[EQUALITY_CONSTRAINT_ONE].observations)[at_least_one_violated])
#     #     invalid_eq_two_squared = tf.square(
#     #         tf.squeeze(initial_data[EQUALITY_CONSTRAINT_TWO].observations)[at_least_one_violated])
#     #     sum_squared = invalid_ineq_one_squared + invalid_eq_one_squared + invalid_eq_two_squared
#     #     min_sum_squared = tf.math.reduce_min(sum_squared)
#     #     if tf.reduce_sum(tf.cast(all_satisfied, tf.int32)) == 0:
#     #         denominator = 2 * tfp.stats.percentile(initial_data[OBJECTIVE].observations, 50.0, interpolation='midpoint')
#     #         initial_penalty = min_sum_squared / denominator
#     #     else:
#     #         best_valid_objective = tf.math.reduce_min(initial_data[OBJECTIVE].observations[all_satisfied])
#     #         denominator = 2 * best_valid_objective
#     #         initial_penalty = min_sum_squared / denominator
#     #
#     # print(f"Original Initial Penalty: {tf.abs(initial_penalty)}")
#     initial_models = trieste.utils.map_values(create_model, initial_data)
#
#     # inequality_lambda = tf.constant([[2.0], [2.0]], dtype=tf.float64)
#     inequality_lambda = {INEQUALITY_CONSTRAINT_ONE: tf.Variable([[[0.0]]], dtype=tf.float64)}
#     equality_lambda = {EQUALITY_CONSTRAINT_ONE: tf.Variable([[[0.0]]], dtype=tf.float64),
#                        EQUALITY_CONSTRAINT_TWO: tf.Variable([[[0.0]]], dtype=tf.float64)}
#
#     # initial_penalty = tf.Variable([[[tf.abs(initial_penalty)]]], dtype=tf.float64)
#     initial_penalty = None
#
#     # save_path = f"../results/22-02-23/run_data/run_0"
#     augmented_lagrangian = BatchThompsonSamplingAugmentedLagrangian(OBJECTIVE, "INEQUALITY", "EQUALITY",
#                                                                     inequality_lambda, equality_lambda,
#                                                                     BATCH_SIZE, None, EPSILON, True, search_space,
#                                                                     plot=False, save_lambda=False,
#                                                                     num_bo_iters=NUM_BO_ITERS)
#
#     rule = EfficientGlobalOptimization(augmented_lagrangian, optimizer=generate_continuous_optimizer(),
#                                        num_query_points=BATCH_SIZE)
#     bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
#     data = bo.optimize(NUM_BO_ITERS, initial_data, initial_models, rule, track_state=True, fit_initial_model=False).try_get_final_datasets()
#     # with open(f"../results/22-02-23/run_data/run_0_data.pkl", "wb") as fp:
#     #     pickle.dump(data, fp)
