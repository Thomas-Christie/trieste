import tensorflow as tf
import gpflow
import trieste
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.acquisition.function.constrained_thompson_sampling import ThompsonSamplingAugmentedLagrangian
from trieste.acquisition.function.updated_constrained_thompson_sampling import UpdatedThompsonSamplingAugmentedLagrangian
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.space import Box
from functions import constraints
from functions import objectives

NUM_INITIAL_SAMPLES = 5
OBJECTIVE = "OBJECTIVE"
INEQUALITY_CONSTRAINT_ONE = "INEQUALITY_CONSTRAINT_ONE"
INEQUALITY_CONSTRAINT_TWO = "INEQUALITY_CONSTRAINT_TWO"


def create_model(data):
    gpr = build_gpr(data, search_space, likelihood_variance=1e-7)
    # gpr = gpflow.models.GPR(data.astuple(), kernel=gpflow.kernels.SquaredExponential())
    return GaussianProcessRegression(gpr, num_rff_features=500)

if __name__ == "__main__":
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    # objective_observer = trieste.objectives.utils.mk_observer(objective=objectives.linear_objective, key=OBJECTIVE)
    observer = trieste.objectives.utils.mk_multi_observer(
        OBJECTIVE=objectives.linear_objective,
        INEQUALITY_CONSTRAINT_ONE=constraints.toy_constraint_one,
        INEQUALITY_CONSTRAINT_TWO=constraints.toy_constraint_two)


    initial_inputs = search_space.sample(NUM_INITIAL_SAMPLES)
    initial_data = observer(initial_inputs)
    initial_models = trieste.utils.map_values(create_model, initial_data)

    # inequality_lambda = tf.constant([[2.0], [2.0]], dtype=tf.float64)
    inequality_lambda = {INEQUALITY_CONSTRAINT_ONE: tf.Variable(0.0, dtype=tf.float64),
                         INEQUALITY_CONSTRAINT_TWO: tf.Variable(0.0, dtype=tf.float64)}
    initial_penalty = tf.Variable(0.5, dtype=tf.float64)

    augmented_lagrangian = UpdatedThompsonSamplingAugmentedLagrangian(OBJECTIVE, "INEQUALITY", None, inequality_lambda, None,
                                                               initial_penalty, 0.001, search_space, True)

    rule = EfficientGlobalOptimization(augmented_lagrangian, optimizer=generate_continuous_optimizer())
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
    data = bo.optimize(50, initial_data, initial_models, rule, track_state=True).try_get_final_dataset()
