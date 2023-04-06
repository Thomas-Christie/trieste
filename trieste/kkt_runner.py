import pickle
import trieste
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.acquisition.function.kkt_expected_improvement import KKTExpectedImprovement,\
    KKTThompsonSamplingExpectedImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization, KKTEfficientGlobalOptimization
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.space import Box
from functions import constraints
from functions import objectives

NUM_INITIAL_SAMPLES = 10
BATCH_SIZE = 1
NUM_BO_ITERS = 100
OBJECTIVE = "OBJECTIVE"
INEQUALITY_CONSTRAINT_ONE = "INEQUALITY_CONSTRAINT_ONE"
INEQUALITY_CONSTRAINT_TWO = "INEQUALITY_CONSTRAINT_TWO"
EQUALITY_CONSTRAINT_ONE = "EQUALITY_CONSTRAINT_ONE"
EQUALITY_CONSTRAINT_TWO = "EQUALITY_CONSTRAINT_TWO"
EI_EPSILON = 0.001
INITIAL_ALPHA = 0.2
ALPHA_LOWER_BOUND = 0.01


def create_model(data):
    gpr = build_gpr(data, search_space, likelihood_variance=1e-7)
    return GaussianProcessRegression(gpr, num_rff_features=500)

# Original Toy Problem
# if __name__ == "__main__":
#     search_space = Box([0.0, 0.0], [1.0, 1.0])
#     observer = trieste.objectives.utils.mk_multi_observer(
#         OBJECTIVE=objectives.linear_objective,
#         INEQUALITY_CONSTRAINT_ONE=constraints.toy_constraint_one,
#         INEQUALITY_CONSTRAINT_TWO=constraints.toy_constraint_two)
#
#     initial_inputs = search_space.sample(NUM_INITIAL_SAMPLES)
#     initial_data = observer(initial_inputs)
#
#     initial_models = trieste.utils.map_values(create_model, initial_data)
#
#     kkt_expected_improvement = KKTExpectedImprovement(OBJECTIVE, "INEQUALITY", None, 0.001, search_space, False)
#
#     rule = EfficientGlobalOptimization(kkt_expected_improvement, optimizer=generate_continuous_optimizer())
#     # rule = EfficientGlobalOptimization(kkt_expected_improvement, optimizer=generate_continuous_optimizer(num_initial_samples=10000,
#     #                                                                                                      num_optimization_runs=100))
#     bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
#     data = bo.optimize(NUM_BO_ITERS, initial_data, initial_models, rule, track_state=True).try_get_final_datasets()
#     with open(f"../results/13-03-23/data.pkl", "wb") as fp:
#         pickle.dump(data, fp)


# Run with updated EGO algorithm, modified for KKT EI acquisition function
if __name__ == "__main__":
    for run in range(20):
        search_space = Box([0.0, 0.0], [1.0, 1.0])
        observer = trieste.objectives.utils.mk_multi_observer(
            OBJECTIVE=objectives.linear_objective,
            INEQUALITY_CONSTRAINT_ONE=constraints.toy_constraint_one,
            INEQUALITY_CONSTRAINT_TWO=constraints.toy_constraint_two)

        initial_inputs = search_space.sample(NUM_INITIAL_SAMPLES)
        initial_data = observer(initial_inputs)

        initial_models = trieste.utils.map_values(create_model, initial_data)

        kkt_expected_improvement = KKTThompsonSamplingExpectedImprovement(OBJECTIVE, "INEQUALITY", None, 0.01, search_space, False)

        rule = EfficientGlobalOptimization(kkt_expected_improvement, optimizer=generate_continuous_optimizer())

        bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
        data = bo.optimize(NUM_BO_ITERS, initial_data, initial_models, rule, track_state=True).try_get_final_datasets()

        kkt_bo = trieste.kkt_bayesian_optimizer.KKTBayesianOptimizer(observer, search_space)
        # data = kkt_bo.optimize(NUM_BO_ITERS, initial_data, initial_models, rule, track_state=True, initial_alpha=INITIAL_ALPHA, alpha_lower_bound=ALPHA_LOWER_BOUND).try_get_final_datasets()
        # with open(f"../results/14-03-23/multi_run_experiment_two/run_{run}_data.pkl", "wb") as fp:
        #     pickle.dump(data, fp)

# GSBP Problem
# if __name__ == "__main__":
#     for run in range(20):
#         search_space = Box([0.0, 0.0], [1.0, 1.0])
#         observer = trieste.objectives.utils.mk_multi_observer(
#             OBJECTIVE=objectives.goldstein_price,
#             INEQUALITY_CONSTRAINT_ONE=constraints.toy_constraint_one,
#             EQUALITY_CONSTRAINT_ONE=constraints.centered_branin,
#             EQUALITY_CONSTRAINT_TWO=constraints.parr_constraint)
#
#         initial_inputs = search_space.sample(NUM_INITIAL_SAMPLES)
#         initial_data = observer(initial_inputs)
#
#         initial_models = trieste.utils.map_values(create_model, initial_data)
#
#         kkt_expected_improvement = KKTExpectedImprovement(OBJECTIVE, "INEQUALITY", "EQUALITY", 0.001, search_space, False)
#
#         rule = KKTEfficientGlobalOptimization(kkt_expected_improvement, optimizer=generate_continuous_optimizer(),
#                                               epsilon=EI_EPSILON)
#
#         kkt_bo = trieste.kkt_bayesian_optimizer.KKTBayesianOptimizer(observer, search_space)
#         data = kkt_bo.optimize(NUM_BO_ITERS, initial_data, initial_models, rule, track_state=True,
#                                initial_alpha=INITIAL_ALPHA,
#                                alpha_lower_bound=ALPHA_LOWER_BOUND).try_get_final_datasets()
#         # with open(f"../results/14-03-23/multi_run_experiment_two/run_{run}_data.pkl", "wb") as fp:
#         #     pickle.dump(data, fp)
