from initialization import *
import numpy as np
import math

import gym
from gym import spaces

class DroneEnv(gym.Env):
    def __init__(self):
        self.drone = Drone([0, 0])
        self.wildfire = Wildfire([125, 126])
        self.action_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self):
        self.drone = Drone([0, 0])
        self.wildfire = Wildfire([125, 126])
        return self._get_obs()

    def step(self, action):
        self.drone.update(action)
        self.wildfire.spread(1.0)
        
        reward = self._calculate_reward()
        done = self._is_done()
        
        return self._get_obs(), reward, done, {}

    def _is_done(self):
        # Condition 1: Drone is close to the wildfire (within 5 units)
        if np.linalg.norm(self.drone.position - self.wildfire.position) < 5.0:
            return True
        # Condition 2: Drone is too far from the origin (over 500 units)
        if np.linalg.norm(self.drone.position) > 1000:
            return True
        return False

    def _get_obs(self):
        return np.concatenate([self.drone.position, self.wildfire.position])

    def _calculate_reward(self):
        distance = np.linalg.norm(self.drone.position - self.wildfire.position)
        if self._is_done() and np.linalg.norm(self.drone.position) > 1000:
            return -100000000  # Penalty for going out of bounds
        elif distance < 5.0:
            return 1000000000  # High reward for being close to the wildfire
        else:
              # Negative reward proportional to distance from the wildfire
            self.drone.timeTaken(self.drone)
            a=self.drone.time
            currentPosition=np.linalg.norm(self.drone.position)
            ratio=currentPosition/distance
            if(ratio>1):
                return -(ratio*distance)-(1/ratio)*a
            else:
                return -(ratio*a)-(1/ratio)*distance
            

            
