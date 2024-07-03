import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the Environment

# Step 1b: Patch reward model: exponential decay
class Patch:
    def __init__(self, initial_yield, decay_rate):
        self.initial_yield = initial_yield
        self.decay_rate = decay_rate
        self.time = 0
        self.harvesting = False

    def start_harvesting(self):
        self.harvesting = True

    def get_reward(self):
        if self.harvesting:
            reward = self.initial_yield * np.exp(-self.decay_rate * self.time)
            self.time += 1
            return reward
        else:
            return self.initial_yield

# Step 1c: Agent’s choice model: softmax with intercept
class Agent:
    def __init__(self, beta, intercept=0):
        self.beta = beta
        self.intercept = intercept

    def choose_action(self, rewards):
        rewards = np.array(rewards)
        adjusted_rewards = rewards + self.intercept
        stay_probability = np.exp(self.beta * adjusted_rewards[-1]) / np.sum(np.exp(self.beta * adjusted_rewards))
        return np.random.choice([0, 1], p=[1-stay_probability, stay_probability])  # 0: stay, 1: leave
