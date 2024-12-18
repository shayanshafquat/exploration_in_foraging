import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.optimize import root, fsolve

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
        self.time = 0

    def get_reward(self):
        if self.harvesting:
            if self.decay_type == 'exponential':
                reward = max(0, self.initial_yield * np.exp(-self.decay_rate * self.time))
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
    def __init__(self,
                  policy_type='softmax',
                  beta=None, intercept=None, 
                  omega=None, 
                  epsilon=None, 
                  mellowmax_type='add',
                  use_intercept=True):
        
        self.policy_type = policy_type
        self.mellowmax_type = mellowmax_type
        self.use_intercept = use_intercept

        if self.policy_type == 'softmax':
            if beta is None:
                raise ValueError("For softmax policy, beta must be provided.")
            self.beta = beta
            self.intercept = intercept if use_intercept else 0
        elif self.policy_type == 'mellowmax':
            if omega is None:
                raise ValueError("For mellowmax policy, omega must be provided.")
            self.omega = omega
            if not use_intercept and mellowmax_type == 'denom':
                self.intercept = 1
            else:
                self.intercept = intercept if use_intercept else 0
        elif self.policy_type == 'epsilon-greedy':
            if epsilon is None:
                raise ValueError("For epsilon-greedy policy, epsilon must be provided.")
            self.epsilon = epsilon
        else:
            raise ValueError("Unsupported policy type: choose either 'softmax', 'mellowmax', or 'epsilon-greedy'.")

    # Softmax policy
    def leave_proba_softmax(self, reward):
        """ Compute the probability of leaving given the reward using softmax. """
        return 1 / (1 + np.exp(self.intercept + self.beta * reward))

    def choose_action_softmax(self, reward):
        leave_proba = self.leave_proba_softmax(reward)
        if isinstance(leave_proba, np.ndarray) and len(leave_proba) == 1:
            leave_proba = leave_proba[0]
        return np.random.choice([0, 1], p=[1-leave_proba, leave_proba])  # 0: stay, 1: leave
    
    # Epsilon-greedy policy
    def choose_action_epsilon(self, reward):
        """ Choose action using epsilon-greedy policy. """
        if np.random.random() < self.epsilon:
            # With probability epsilon, choose a random action (explore)
            return np.random.choice([0, 1])  # 0: stay, 1: leave
        else:
            # With probability 1 - epsilon, choose the default action (exploit)
            return 0  # Default action is to stay

    # Mellowmax policy with intercept
    def mellowmax_operator(self, Q_values):
        """ Calculate the mellowmax operator for given Q-values. """
        if self.mellowmax_type == 'add':
            return np.log(np.mean(np.exp(self.omega * Q_values) + self.intercept)) / self.omega 
        elif self.mellowmax_type == 'denom':
            return np.log(np.mean(np.exp(self.omega * Q_values) / self.intercept)) / self.omega
        else:
            raise ValueError("Unsupported mellowmax type: choose either 'add' or 'denom'.")

    def optimize_beta(self, reward, mm_value):
        """ Optimize beta_hat using Brent's method. """
        def root_function(beta_hat):
            term1 = np.exp(-beta_hat * mm_value) * (-mm_value)
            term2 = np.exp(beta_hat * (reward - mm_value)) * (reward - mm_value)
            return term1 + term2

        # beta_hat = brentq(root_function, 0, 1)
        initial_guess = 0  # Start at 0 or another reasonable guess
        beta_hat = fsolve(root_function, initial_guess)
        return beta_hat[0]

        # # Use root to find the root of the function
        # initial_guess = 0.3  # A reasonable initial guess, could also be adjusted based on context
        # result = root(root_function, initial_guess)
        # if result.success:
        #     return result.x[0]  # root returns an array, so return the first element
        # else:
        #     raise ValueError("Root finding failed to converge")


    def leave_proba_mellowmax(self, reward):
        """ Compute the probability of leaving given the reward using mellowmax with intercept. """
        Q_stay = reward
        Q_leave = 0
        Q_values = np.array([Q_stay, Q_leave])

        mm_value = self.mellowmax_operator(Q_values)
        beta_hat = self.optimize_beta(reward, mm_value)
        leave_proba = 1 / (1 + np.exp(beta_hat * reward))
        return leave_proba

    def choose_action_mellowmax(self, reward):
        """ Choose action using mellowmax policy. """
        leave_proba = self.leave_proba_mellowmax(reward)
        return np.random.choice([0, 1], p=[1-leave_proba, leave_proba])  # 0: stay, 1: leave

    def get_leave_probability(self, reward):
        """ Get the leave probability based on the selected policy type. """
        if self.policy_type == 'softmax':
            return self.leave_proba_softmax(reward)
        elif self.policy_type == 'mellowmax':
            return self.leave_proba_mellowmax(reward)
        elif self.policy_type == 'epsilon-greedy':
            return self.leave_proba_softmax(reward)
        else:
            raise ValueError("Unsupported policy type: choose either 'softmax', 'mellowmax', or 'epsilon-greedy'.")