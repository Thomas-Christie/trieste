import gymnasium as gym
from gymnasium.utils.step_api_compatibility import step_api_compatibility
import numpy as np
import tensorflow as tf
import trieste

from trieste.types import TensorType, Tag
from trieste.data import Dataset
from typing import Mapping

STEPS_LIMIT = 1000  # Max number of steps to run the Lunar environment for
TIMEOUT_REWARD = -100  # Reward to assign to a run that exceeds the step limit

# Based on Trieste documentation implementation at
# https://secondmind-labs.github.io/trieste/0.13.2/notebooks/openai_gym_lunar_lander.html 
# which is in turn taken from https://github.com/uber-research/TuRBO and based on
# https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py#L726 
def heuristic_controller(s, w):
    angle_targ = s[0] * w[0] + s[2] * w[1]
    if angle_targ > w[2]:
        angle_targ = w[2]
    if angle_targ < -w[2]:
        angle_targ = -w[2]
    hover_targ = w[3] * np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
    hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

    if s[6] or s[7]:
        angle_todo = w[8]
        hover_todo = -(s[3]) * w[9]

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
        a = 2
    elif angle_todo < -w[11]:
        a = 3
    elif angle_todo > +w[11]:
        a = 1
    return a


# Based on Trieste documentation implementation at
# https://secondmind-labs.github.io/trieste/0.13.2/notebooks/openai_gym_lunar_lander.html 
def run_lander(env: gym.Env, param_configs: TensorType, seed: int) -> TensorType:
    param_configs = 2.0 * param_configs  # For GPs the parameters are confined to [0, 1] so rescale to [0.0, 2.0] as mentioned at https://github.com/uber-research/TuRBO 
    
    rewards = []
    for param_config in param_configs.numpy():
        total_reward = 0
        steps = 0
        s, _ = env.reset(seed=seed)

        while True:
            if steps > STEPS_LIMIT:
                total_reward -= TIMEOUT_REWARD  # TODO: IMPORTANT - Should we have this? Or just return TIMEOUT_REWARD?
                break

            a = heuristic_controller(s, param_config)
            s, r, terminated, truncated, _ = step_api_compatibility(env.step(a), True)
            total_reward += r

            steps += 1
            if terminated or truncated:
                break
        rewards.append(total_reward)

    return tf.constant(rewards, dtype=tf.float64)[..., None]

def lunar_lander_observer(num_envs: int, env: gym.Env, query_points: TensorType) -> Mapping[Tag, Dataset]:
    """
    Takes input values of shape [N, 12] and returns tagged Dataset with objective and constraint
    observations at given query points. 'num_envs' is the number of different
    environments to run (i.e. one for each constraint as in https://arxiv.org/pdf/2002.08526.pdf).
    """
    tagged_observations = {}
    sum_rewards = tf.zeros((query_points.shape[0], 1), dtype=tf.float64)
    for env_num in range(num_envs):
        observations = run_lander(env, query_points, 42+env_num)
        sum_rewards += observations / 100.0 
        inequality_observations = (-1.0 * (observations - 200.0)) / 100.0  # Inequality constraint considered satisfied if reward is greater than 200
        tagged_observations[f"INEQUALITY_CONSTRAINT_{env_num}"] = Dataset(query_points, inequality_observations)

    mean_rewards = - sum_rewards / num_envs
    tagged_observations["OBJECTIVE"] = Dataset(query_points, mean_rewards)
    return tagged_observations



if __name__ == "__main__":
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    
    search_space = trieste.space.Box([0.0] * 12, [1.0] * 12)
    sample_w = search_space.sample(3)
    # a = tf.constant([[0.97736803, 0.73969325, 0.30885848, 0.40588711, 0.8175496,  0.93936169,
    #                   0.1362836,  0.54899272, 0.02122464, 0.85753745, 0.16552496,
    #                   0.02558125]], dtype=tf.float64)
    # a = tf.constant([[0.5472385104100799 , 0.4025476022107832 , 0.585036877787459  ,
    #    0.3469967235170917 , 0.4926346816255221 , 0.24439541606940574,
    #    0.3838592532813656 , 0.9036197620873828 , 0.5975460664605121 ,
    #    0.07169203429576296, 0.10388661184069026, 0.09984863077855238]], dtype=tf.float64)
    # a = tf.constant([[0.23945592, 0.62433714, 0.68457257, 0.466923, 0.8467446,  0.44547555,
    #                   0.26109718, 0.76523342, 0.52852851, 0.28296125, 0.05204201,
    #                   0.10942715]], dtype=tf.float64)
    a = tf.constant([[0.417509849022173  , 0.8151445852240101 , 0.5542238672033017 ,
       0.02824296592134945, 0.6887390072109282 , 0.48174070023392157,
       0.29544770952650284, 0.4353022417483064 , 0.14989989084384164,
       0.37192989382743114, 0.1432089953810865 , 0.271542013550477  ]], dtype=tf.float64)
    
    obs = lunar_lander_observer(10, env, a)
    for key, val in obs.items():
        print(f"{key}: {val.observations}")