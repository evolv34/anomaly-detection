import gym
import pandas as pd
from gym.spaces import Discrete, Box


class FraudDetectEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        print("Fraud Detect Gym Trainer env")
        self.obs = pd.DataFrame(columns=[])
        self.target = pd.DataFrame(columns=[])
        self.path = ""
        self.target_col = ""
        self.obs_index = 0
        self.action_space = Discrete(2)
        self.info = {}
        self.observation_space = Box(low=0, high=1, shape=(0, 0))

    def init(self, path: str, target_col: str):
        print(f"loading dataset from path = [{path}]")
        print(f"target col = [{target_col}]")
        self.path = path
        self.target_col = target_col
        self.obs_index = 0
        self.action_space = Discrete(2)
        self.info = {}

        self.obs = pd.read_csv(path)
        self.target = self.obs[target_col]

        self.obs = self.obs.drop(labels=[self.target_col], axis=1)
        self.observation_space = Box(low=0, high=1, shape=self.obs.shape)

    # fetch next observation - DONE
    # calculate action (target value for that action) - DONE
    # calculate reward - DONE
    # return obs, action and reward - DONE
    def step(self, action: int):
        current_step = self.obs_index
        self.obs_index = self.obs_index + 1

        reward = 0
        # if abs((action - self.target[current_step])) < 0.05:
        reward = action - self.target[current_step]
        # else:
        #     reward = -1

        done = False
        if self.obs_index == self.obs.shape[0]:
            done = True
        return self.obs.iloc[current_step], reward, done, self.info

    def reset(self):
        print(f"resetting env")
        self.init(self.path, self.target_col)

        current_step = self.obs_index
        self.obs_index = self.obs_index + 1
        return self.obs.iloc[current_step]

    def render(self, mode='human'):
        print(f"rendering env")
