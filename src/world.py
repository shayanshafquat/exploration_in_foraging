import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

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
    def __init__(self, beta=1.0, intercept=0, omega=0.5):
        self.beta = beta
        self.intercept = intercept
        self.omega = omega

    # Softmax policy
    def leave_proba_softmax(self, reward):
        """ Compute the probability of leaving given the reward using softmax. """
        return 1 / (1 + np.exp(self.intercept + self.beta * reward))

    def choose_action_softmax(self, reward):
        leave_proba = self.leave_proba_softmax(reward)
        return np.random.choice([0, 1], p=[1-leave_proba, leave_proba])  # 0: stay, 1: leave
    
    # Epsilon-greedy policy
    def choose_action_epsilon(self, reward, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice([0, 1])
        else:
            leave_proba = self.leave_proba_softmax(reward)
            return 1 if leave_proba > 0.5 else 0

    # Mellowmax policy
    def mellowmax_operator(self, Q_values):
        """ Calculate the mellowmax operator for given Q-values. """
        return np.log(np.mean(np.exp(self.omega * Q_values))) / self.omega

    def optimize_beta(self, reward, mm_value):
        """ Optimize beta_hat using Brent's method. """
        def root_function(beta_hat):
            term1 = np.exp(-beta_hat * mm_value) * (-mm_value)
            term2 = np.exp(beta_hat * (reward - mm_value)) * (reward - mm_value)
            return term1 + term2

        beta_hat = brentq(root_function, -10, 10)
        return beta_hat

    def choose_action_mellowmax(self, reward):
        """ Choose action using mellowmax policy. """
        Q_stay = reward
        Q_leave = 0
        Q_values = np.array([Q_stay, Q_leave])

        mm_value = self.mellowmax_operator(Q_values)
        beta_hat = self.optimize_beta(reward, mm_value)
        # print(beta_hat)
        leave_proba = 1 / (1 + np.exp(beta_hat * reward))
        return np.random.choice([0, 1], p=[1-leave_proba, leave_proba])  # 0: stay, 1: leave