import trieste
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.acquisition.function.kkt_expected_improvement import KKTExpectedImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.space import Box
from functions import constraints
from functions import objectives

NUM_INITIAL_SAMPLES = 6
BATCH_SIZE = 1
NUM_BO_ITERS = 20
OBJECTIVE = "OBJECTIVE"
INEQUALITY_CONSTRAINT_ONE = "INEQUALITY_CONSTRAINT_ONE"
INEQUALITY_CONSTRAINT_TWO = "INEQUALITY_CONSTRAINT_TWO"
EQUALITY_CONSTRAINT_ONE = "EQUALITY_CONSTRAINT_ONE"
EQUALITY_CONSTRAINT_TWO = "EQUALITY_CONSTRAINT_TWO"
EPSILON = 0.01


def create_model(data):
    gpr = build_gpr(data, search_space, likelihood_variance=1e-7)
    return GaussianProcessRegression(gpr, num_rff_features=500)

# Original Toy Problem
if __name__ == "__main__":
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    observer = trieste.objectives.utils.mk_multi_observer(
        OBJECTIVE=objectives.linear_objective,
        INEQUALITY_CONSTRAINT_ONE=constraints.toy_constraint_one,
        INEQUALITY_CONSTRAINT_TWO=constraints.toy_constraint_two)

    initial_inputs = search_space.sample(NUM_INITIAL_SAMPLES)
    initial_data = observer(initial_inputs)

    initial_models = trieste.utils.map_values(create_model, initial_data)

    kkt_expected_improvement = KKTExpectedImprovement(OBJECTIVE, "INEQUALITY", None, 0.001, search_space, False)

    rule = EfficientGlobalOptimization(kkt_expected_improvement, optimizer=generate_continuous_optimizer())
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
    data = bo.optimize(NUM_BO_ITERS, initial_data, initial_models, rule, track_state=True).try_get_final_datasets()
    # with open(f"../results/03-03-23/kkt_6_initial_samples/run_{run}_data.pkl", "wb") as fp:
    #     pickle.dump(data, fp)