import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the Environment

# Step 1b: Patch reward model: exponential decay
class Patch:
    def __init__(self, initial_yield, decay_rate, decay_type='exponential'):
        self.initial_yield = initial_yield
        self.decay_rate = decay_rate
        self.decay_type = decay_type
        self.time = 0
        self.harvesting = False

    def start_harvesting(self):
        self.harvesting = True

    def get_reward(self):
        if self.harvesting:
            if self.decay_type == 'exponential':
                reward = self.initial_yield * np.exp(-self.decay_rate * self.time)
            elif self.decay_type == 'linear':
                reward = max(0, self.initial_yield - self.decay_rate * self.time)
            else:
                raise ValueError("Invalid decay type. Use 'exponential' or 'linear'.")
            self.time += 1
            return reward
        else:
            return self.initial_yield

# Step 1c: Agent’s choice model: softmax with intercept
class Agent:
    def __init__(self, beta, intercept=0):
        self.beta = beta
        self.intercept = intercept

    def leave_proba(self, reward):
        """ Compute the probability of leaving given the reward. """
        return 1 / (1 + np.exp(self.intercept + self.beta * reward))

    def choose_action(self, reward):
        leave_proba = 1 / (1 + np.exp(self.intercept + self.beta * reward))
        return np.random.choice([0, 1], p=[1-leave_proba, leave_proba])  # 0: stay, 1: leave
    
    # Epsilon-greedy policy
    def choose_action_epsilon(self, reward, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice([0, 1])
        else:
            return self.choose_action(reward)
