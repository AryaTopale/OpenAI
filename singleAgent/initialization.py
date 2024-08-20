import numpy as np
import torch
import math
import tensorflow as tf

#Initialization of Drone

class Drone:
    def __init__(self, position):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.acceleration = np.zeros(2, dtype=np.float32)
        self.time=0

    def update(self, action, dt=1.0):
        self.acceleration = np.clip(action, -20, 20)  # Ensure acceleration is within [-10, 10]
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt + 0.5 * self.acceleration * dt**2   
        
    def timeTaken(self):
        self.time=self.time+1;
     
#Class for Obstacle

class Wildfire:
    def __init__(self, position, intensity=1.0, spread_rate=0.1):
        self.position = np.array(position, dtype=np.float32)
        self.intensity = intensity
        self.spread_rate = spread_rate
        self.radius = 0.0

    def spread(self, dt):
        self.radius += self.spread_rate * dt
        self.intensity = max(0, self.intensity / (1 + self.spread_rate * dt))



#Class for Reward Model

# class RewardModel:
#     def __init__(self,drone,wildfire):
#         self.drone=drone
#         self.wildfire=wildfire
#     def distanceToFire(self):
#         drone_x,drone_y=self.drone.position
#         fire_x,fire_y=self.wildfire.position
#         return math.sqrt((drone_x-fire_x)**2+(drone_y-fire_y)**2)
#     def calculateReward(self):
#         distancetoFire=self.distanceToFire()
#         radius=self.wildfire.radius
#         if distancetoFire<radius:
#             difference=abs(distancetoFire-radius)
#             scale=3
#             reward= -scale*(difference)
#         else:
#             epsilon=10.0;
#             if distancetoFire<epsilon:
#                 reward=1000000
#             else:
#                 reward=-(distancetoFire)
#         return reward
                
                
                
            
        
        


