import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, norm
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from collections import deque

from utils import prepare_avg_rewards

class Agent:
    def __init__(self, decay_rate_means, decay_rate_stds, observation_var=0.00001):
        self.decay_rate_means = decay_rate_means
        self.decay_rate_stds = decay_rate_stds
        self.observation_var = observation_var
        self.beliefs = {(i+1, env): {'mean': decay_rate_means[i], 'std': decay_rate_stds[i], 'count': 0}
                        for i in range(len(decay_rate_means)) for env in [1, 2]}
        self.mean_history = {(i+1, env): [] for i in range(len(decay_rate_means)) for env in [1, 2]}
        self.std_history = {(i+1, env): [] for i in range(len(decay_rate_means)) for env in [1, 2]}

    def sample_decay_rate(self, patch_type, env):
        belief = self.beliefs[(patch_type, env)]
        return np.random.normal(belief['mean'], belief['std'])

    def update_belief(self, patch_type, env, observed_decay_rate):
        belief = self.beliefs[(patch_type, env)]
        prior_mean = belief['mean']
        prior_var = belief['std'] ** 2
        belief['count'] += 1
        count = belief['count']

        if self.observation_var > 0:
            # Adjust the observation variance to decrease with more observations
            adjusted_observation_var = self.observation_var / count
            
            # Bayesian update for the mean and variance
            posterior_mean = (prior_var * observed_decay_rate + adjusted_observation_var * prior_mean) / (prior_var + adjusted_observation_var)
            posterior_var = (prior_var * adjusted_observation_var) / (prior_var + adjusted_observation_var)
        else:
            # Simple averaging for the mean if observation variance is zero
            posterior_mean = (prior_mean * (count - 1) + observed_decay_rate) / count
            posterior_var = prior_var * (count - 1) / count  # Decrease the uncertainty by the number of observations

        # Update the Agent's belief with the new mean and reduced variance
        self.beliefs[(patch_type, env)]['mean'] = posterior_mean
        self.beliefs[(patch_type, env)]['std'] = np.sqrt(posterior_var)

        # Log the updated belief
        self.mean_history[(patch_type, env)].append(posterior_mean)
        self.std_history[(patch_type, env)].append(np.sqrt(posterior_var))

    def decide_leave(self, reward, average_reward):
        return reward < average_reward

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
                reward = max(0.1, self.initial_yield * np.exp(-self.decay_rate * self.time))
            elif self.decay_type == 'linear':
                reward = max(0.1, self.initial_yield - self.decay_rate * self.time)
            else:
                raise ValueError("Invalid decay type. Use 'exponential' or 'linear'.")
            self.time += 1
            return reward
        else:
            return self.initial_yield

# class Simulation:
#     def __init__(self, decay_rate_means, decay_rate_stds, avg_rewards, observation_var=0.01, learning_rate=0.1, method='bayesian', random_seed=None):
#         self.decay_rate_means = decay_rate_means
#         self.decay_rate_stds = decay_rate_stds
#         self.avg_rewards = avg_rewards
#         self.observation_var = observation_var
#         self.learning_rate = learning_rate
#         self.method = method
#         self.patch_types = self.initialize_env()

#         if random_seed is not None:
#             np.random.seed(random_seed)

#     def initialize_env(self):
#         patch_types = [
#             {'type': 1, 'initial_yield': 32.5, 'decay_rate_mean': 0.075},
#             {'type': 2, 'initial_yield': 45, 'decay_rate_mean': 0.075},
#             {'type': 3, 'initial_yield': 57.5, 'decay_rate_mean': 0.075}
#         ]
#         return patch_types

#     def get_patch_info(self, patch_type):
#         for patch in self.patch_types:
#             if patch['type'] == patch_type:
#                 return patch
#         raise ValueError("Patch type not found")

#     # def simulate_subject(self, subject_df, agent, avg_reward, env, n_max=1000):
#     #     patch_sequence = []
#     #     leave_times = []

#     #     for _, trial in subject_df.iterrows():
#     #         patch_info = self.get_patch_info(trial['patch'])
#     #         patch = Patch(
#     #             patch_info['initial_yield'], 
#     #             patch_info['decay_rate_mean']
#     #         )
#     #         patch.start_harvesting()
            
#     #         # Sample decay rate based on the updated belief
#     #         sampled_decay_rate = agent.sample_decay_rate(trial['patch'], env)
#     #         patch.decay_rate = sampled_decay_rate  # Apply the sampled decay rate to the patch

#     #         for t in range(1, n_max + 1):
#     #             reward = patch.get_reward()

#     #             if agent.decide_leave(reward, avg_reward) or t>=n_max:
#     #                 observed_decay_rate = -np.log(reward / patch_info['initial_yield']) / t if t > 0 else sampled_decay_rate
#     #                 agent.update_belief(trial['patch'], env, observed_decay_rate)
#     #                 patch_sequence.append(trial['patch'])
#     #                 leave_times.append(t)
#     #                 break

#     #     return patch_sequence, leave_times
#     def simulate_subject(self, subject_df, agent, avg_reward, env, n_max=1000):
#         patch_sequence = []
#         leave_times = []

#         for _, trial in subject_df.iterrows():
#             patch_info = self.get_patch_info(trial['patch'])
#             patch = Patch(
#                 patch_info['initial_yield'], 
#                 patch_info['decay_rate_mean']
#             )
#             patch.start_harvesting()
            
#             # Sample decay rate based on the updated belief
#             sampled_decay_rate = agent.sample_decay_rate(trial['patch'], env)
#             patch.decay_rate = sampled_decay_rate  # Apply the sampled decay rate to the patch

#             for t in range(1, n_max + 1):
#                 if agent.method == 'fixed_noise':
#                     # Calculate reward using the fixed noise method
#                     reward = agent.get_reward_with_noise(
#                         initial_yield=patch_info['initial_yield'], 
#                         true_decay_rate=sampled_decay_rate, 
#                         time=t
#                     )
#                 else:
#                     # Default reward calculation
#                     reward = patch.get_reward()

#                 # Calculate expected reward using the fixed decay parameter if using Rescorla-Wagner
#                 if agent.method == 'rescorla_wagner':
#                     fixed_decay_rate = 0.075  # Set the fixed decay parameter here
#                     expected_reward = agent.calculate_expected_reward(
#                         initial_yield=patch_info['initial_yield'], 
#                         decay_rate=fixed_decay_rate, 
#                         time=t
#                     )
#                 else:
#                     expected_reward = avg_reward  # Use average reward for other methods

#                 if agent.decide_leave(reward, avg_reward) or t >= n_max:
#                     if agent.method not in ['fixed_noise', 'random_walk', 'no_mean_change']:
#                         observed_decay_rate = -np.log(reward / patch_info['initial_yield']) / t if t > 0 else sampled_decay_rate
#                     else:
#                         observed_decay_rate = sampled_decay_rate
                    
#                     # Update belief based on the specific method
#                     agent.update_belief(trial['patch'], env, observed_decay_rate, reward, expected_reward)
                    
#                     patch_sequence.append(trial['patch'])
#                     leave_times.append(t)
#                     break

#         return patch_sequence, leave_times

#     def run_simulation(self, df_trials, subject_id=None):
#         simulated_leave_times = []
#         if subject_id is not None:
#             subject_df = df_trials[df_trials['sub'] == subject_id]
#             patch_sequence = []
#             agent = MultipleAgent(
#                 decay_rate_means=self.decay_rate_means, 
#                 decay_rate_stds=self.decay_rate_stds, 
#                 method=self.method, 
#                 observation_var=self.observation_var,
#                 learning_rate=self.learning_rate
#                 )
#             for env in subject_df['env'].unique():
#                 avg_reward = self.avg_rewards[(subject_id, env)]
#                 subject_env_df = subject_df[subject_df['env'] == env]
#                 seq, leave_times = self.simulate_subject(subject_env_df, agent, avg_reward, env)
#                 patch_sequence.extend(seq)
#                 simulated_leave_times.extend(leave_times)
#             return agent, patch_sequence, simulated_leave_times
#         else:
#             for subject in df_trials['sub'].unique():
#                 subject_df = df_trials[df_trials['sub'] == subject]
#                 for env in subject_df['env'].unique():
#                     avg_reward = self.avg_rewards[(subject, env)]
#                     agent = MultipleAgent(
#                         decay_rate_means=self.decay_rate_means, 
#                         decay_rate_stds=self.decay_rate_stds, 
#                         method=self.method, 
#                         observation_var=self.observation_var,
#                         learning_rate=self.learning_rate
#                         )
#                     subject_env_df = subject_df[subject_df['env'] == env]
#                     _, leave_times = self.simulate_subject(subject_env_df, agent, avg_reward, env)
#                     simulated_leave_times.extend(leave_times)
#             df_trials['simulated_leaveT'] = simulated_leave_times
#             df_trials['simulated_dmLeave'] = df_trials['simulated_leaveT'] - df_trials['meanLT']
#             return df_trials

class Simulation:
    def __init__(self, decay_rate_means, decay_rate_stds, avg_rewards, observation_var=0.01, learning_rate=0.1, ema_alpha=0.1, noise_std=0.01, noise_decay_rate=0.95, step_size=0.001, n_components=2, random_seed=None, method='ema'):
        self.decay_rate_means = decay_rate_means
        self.decay_rate_stds = decay_rate_stds
        self.avg_rewards = avg_rewards
        self.observation_var = observation_var
        self.learning_rate = learning_rate  # For Rescorla-Wagner
        self.ema_alpha = ema_alpha  # For EMA
        self.noise_std = noise_std  # For Fixed Noise
        self.noise_decay_rate = noise_decay_rate  # Rate at which noise decreases
        self.step_size = step_size  # For Random Walk
        self.n_components = n_components  # For GMM
        self.method = method  # Method for belief updating
        self.patch_types = self.initialize_env()

        if random_seed is not None:
            np.random.seed(random_seed)

    def initialize_env(self):
        patch_types = [
            {'type': 1, 'initial_yield': 32.5, 'decay_rate_mean': 0.075},
            {'type': 2, 'initial_yield': 45, 'decay_rate_mean': 0.075},
            {'type': 3, 'initial_yield': 57.5, 'decay_rate_mean': 0.075}
        ]
        return patch_types

    def get_patch_info(self, patch_type):
        for patch in self.patch_types:
            if patch['type'] == patch_type:
                return patch
        raise ValueError("Patch type not found")

    def create_agent(self, method):
        """
        Create and configure a MultipleAgent instance based on the selected method.
        """
        if method == 'bayesian':
            return MultipleAgent(
                decay_rate_means=self.decay_rate_means, 
                decay_rate_stds=self.decay_rate_stds, 
                method=method,
                observation_var=self.observation_var
            )
        elif method == 'rescorla_wagner':
            return MultipleAgent(
                decay_rate_means=self.decay_rate_means, 
                decay_rate_stds=self.decay_rate_stds, 
                method=method,
                learning_rate=self.learning_rate
            )
        elif method == 'ema':
            return MultipleAgent(
                decay_rate_means=self.decay_rate_means, 
                decay_rate_stds=self.decay_rate_stds, 
                method=method,
                ema_alpha=self.ema_alpha
            )
        # elif method == 'gmm':
        #     return MultipleAgent(
        #         decay_rate_means=self.decay_rate_means, 
        #         decay_rate_stds=self.decay_rate_stds, 
        #         method=method,
        #         n_components=self.n_components
        #     )
        elif method == 'fixed_noise':
            return MultipleAgent(
                decay_rate_means=self.decay_rate_means, 
                decay_rate_stds=self.decay_rate_stds, 
                method=method,
                noise_std=self.noise_std,
                noise_decay_rate=self.noise_decay_rate
            )
        elif method == 'random_walk':
            return MultipleAgent(
                decay_rate_means=self.decay_rate_means, 
                decay_rate_stds=self.decay_rate_stds, 
                method=method,
                step_size=self.step_size
            )
        elif method == 'hierarchical':
            return MultipleAgent(
                decay_rate_means=self.decay_rate_means, 
                decay_rate_stds=self.decay_rate_stds, 
                method=method,
                learning_rate=self.learning_rate
            )
        elif method == 'no_mean_change':
            return MultipleAgent(
                decay_rate_means=self.decay_rate_means, 
                decay_rate_stds=self.decay_rate_stds, 
                method=method
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

    def simulate_subject(self, subject_df, agent, avg_reward, env, n_max=1000):
        patch_sequence = []
        leave_times = []

        for _, trial in subject_df.iterrows():
            patch_info = self.get_patch_info(trial['patch'])
            patch = Patch(
                patch_info['initial_yield'], 
                patch_info['decay_rate_mean']
            )
            patch.start_harvesting()
            
            # Sample decay rate based on the updated belief
            sampled_decay_rate = agent.sample_decay_rate(trial['patch'], env)
            patch.decay_rate = sampled_decay_rate  # Apply the sampled decay rate to the patch

            for t in range(1, n_max + 1):
                if agent.method == 'fixed_noise':
                    # Calculate reward using the fixed noise method
                    reward = agent.get_reward_with_noise(
                        initial_yield=patch_info['initial_yield'], 
                        true_decay_rate=0.075, 
                        time=t,
                        env=env
                    )
                else:
                    # Default reward calculation
                    reward = patch.get_reward()
                    agent.record_reward(trial['patch'], env, reward)

                # Calculate expected reward using the fixed decay parameter if using Rescorla-Wagner
                if agent.method == 'rescorla_wagner':
                    fixed_decay_rate = 0.075  # Set the fixed decay parameter here
                    expected_reward = agent.calculate_expected_reward(
                        initial_yield=patch_info['initial_yield'], 
                        decay_rate=fixed_decay_rate, 
                        time=t
                    )
                else:
                    expected_reward = avg_reward  # Use average reward for other methods

                if agent.decide_leave(reward, avg_reward) or t >= n_max:
                    if agent.method not in ['fixed_noise', 'random_walk', 'no_mean_change']:
                        observed_decay_rate = -np.log(reward / patch_info['initial_yield']) / t if t > 0 else sampled_decay_rate
                    else:
                        observed_decay_rate = sampled_decay_rate
                    
                    # Update belief based on the specific method
                    agent.update_belief(trial['patch'], env, observed_decay_rate, reward, expected_reward)
                    
                    patch_sequence.append(trial['patch'])
                    leave_times.append(t)
                    break

        return patch_sequence, leave_times

    def run_simulation(self, df_trials, subject_id=None):
        simulated_leave_times = []
        if subject_id is not None:
            subject_df = df_trials[df_trials['sub'] == subject_id]
            patch_sequence = []
            agent = self.create_agent(method=self.method)
            for env in subject_df['env'].unique():
                avg_reward = self.avg_rewards[(subject_id, env)]
                subject_env_df = subject_df[subject_df['env'] == env]
                seq, leave_times = self.simulate_subject(subject_env_df, agent, avg_reward, env)
                patch_sequence.extend(seq)
                simulated_leave_times.extend(leave_times)
            return agent, patch_sequence, simulated_leave_times
        else:
            for subject in df_trials['sub'].unique():
                subject_df = df_trials[df_trials['sub'] == subject]
                for env in subject_df['env'].unique():
                    avg_reward = self.avg_rewards[(subject, env)]
                    agent = self.create_agent(method=self.method)
                    subject_env_df = subject_df[subject_df['env'] == env]
                    _, leave_times = self.simulate_subject(subject_env_df, agent, avg_reward, env)
                    simulated_leave_times.extend(leave_times)
            df_trials['simulated_leaveT'] = simulated_leave_times
            df_trials['simulated_dmLeave'] = df_trials['simulated_leaveT'] - df_trials['meanLT']
            return df_trials

class MultipleAgent:
    def __init__(self, decay_rate_means, decay_rate_stds, method='bayesian', observation_var=0.00001, noise_std=0.01, noise_decay_rate = 0.95, learning_rate=0.1, ema_alpha=0.1, n_components=2):
        self.decay_rate_means = decay_rate_means
        self.decay_rate_stds = decay_rate_stds
        self.observation_var = observation_var
        self.noise_std = noise_std
        self.noise_decay_rate = noise_decay_rate
        self.learning_rate = learning_rate
        self.ema_alpha = ema_alpha
        self.n_components = n_components
        self.method = method  # The method to be used for belief updates

        self.beliefs = {(i+1, env): {'mean': decay_rate_means[i], 'std': decay_rate_stds[i], 'count': 0}
                        for i in range(len(decay_rate_means)) for env in [1, 2]}
        self.mean_history = {(i+1, env): [] for i in range(len(decay_rate_means)) for env in [1, 2]}
        self.std_history = {(i+1, env): [] for i in range(len(decay_rate_means)) for env in [1, 2]}
        self.reward_history = {(i+1, env): [] for i in range(len(decay_rate_means)) for env in [1, 2]}
        self.observation_counts = {env: 0 for env in [1, 2]}

    def sample_decay_rate(self, patch_type, env):
        belief = self.beliefs[(patch_type, env)]
        return np.random.normal(belief['mean'], belief['std'])

    def calculate_expected_reward(self, initial_yield, decay_rate, time):
        """
        Calculate the expected reward using the fixed decay parameter.
        """
        return max(0.1, initial_yield * np.exp(-decay_rate * time))

    def update_belief(self, patch_type, env, observed_decay_rate=None, observed_reward=None, expected_reward=None):
        if self.method == 'bayesian':
            self.update_belief_bayesian(patch_type, env, observed_decay_rate)
        elif self.method == 'rescorla_wagner':
            self.update_belief_rescorla_wagner(patch_type, env, observed_reward, expected_reward)
        elif self.method == 'ema':
            self.update_belief_ema(patch_type, env, observed_decay_rate)
        elif self.method == 'gmm':
            self.update_belief_gmm(patch_type, env)
        elif self.method == 'fixed_noise':
            # No update to the belief, we directly generate noisy rewards instead
            pass
        elif self.method == 'random_walk':
            self.update_belief_random_walk(patch_type, env)
        elif self.method == 'hierarchical':
            self.update_belief_hierarchical(patch_type, env, observed_decay_rate)
        elif self.method == 'no_mean_change':
            self.update_belief_no_mean_change(patch_type, env, observed_decay_rate)

    # Original Bayesian Update Method
    def update_belief_bayesian(self, patch_type, env, observed_decay_rate):
        belief = self.beliefs[(patch_type, env)]
        prior_mean = belief['mean']
        prior_var = belief['std'] ** 2
        belief['count'] += 1
        count = belief['count']

        if self.observation_var > 0:
            # Adjust the observation variance to decrease with more observations
            adjusted_observation_var = self.observation_var / count
            
            # Bayesian update for the mean and variance
            posterior_mean = (prior_var * observed_decay_rate + adjusted_observation_var * prior_mean) / (prior_var + adjusted_observation_var)
            posterior_var = (prior_var * adjusted_observation_var) / (prior_var + adjusted_observation_var)
        else:
            # Simple averaging for the mean if observation variance is zero
            posterior_mean = (prior_mean * (count - 1) + observed_decay_rate) / count
            posterior_var = prior_var * (count - 1) / count  # Decrease the uncertainty by the number of observations

        # Update the Agent's belief with the new mean and reduced variance
        self.beliefs[(patch_type, env)]['mean'] = posterior_mean
        self.beliefs[(patch_type, env)]['std'] = np.sqrt(posterior_var)

        # Log the updated belief
        self.mean_history[(patch_type, env)].append(posterior_mean)
        self.std_history[(patch_type, env)].append(np.sqrt(posterior_var))

    # Method 1: Rescorla-Wagner Model
    def update_belief_rescorla_wagner(self, patch_type, env, observed_reward, expected_reward):
        belief = self.beliefs[(patch_type, env)]
        prediction_error = observed_reward - expected_reward
        updated_mean = belief['mean'] + self.learning_rate * prediction_error

        # Ensure decay rate stays positive
        updated_mean = max(updated_mean, 0)
        
        belief['mean'] = updated_mean
        self.mean_history[(patch_type, env)].append(updated_mean)

    # Method 2: Exponential Moving Average (EMA)
    def update_belief_ema(self, patch_type, env, observed_decay_rate):
        belief = self.beliefs[(patch_type, env)]
        updated_mean = self.ema_alpha * observed_decay_rate + (1 - self.ema_alpha) * belief['mean']
        
        belief['mean'] = updated_mean
        self.mean_history[(patch_type, env)].append(updated_mean)

    # # Method 3: Gaussian Mixture Model (GMM)
    # def update_belief_gmm(self, patch_type, env):
    #     belief = self.beliefs[(patch_type, env)]
    #     reward_data = self.reward_history[(patch_type, env)]
    #     print(reward_data)
    #     if len(reward_data) >= self.n_components:  # Only fit GMM if enough data is available
    #         gmm = GaussianMixture(n_components=self.n_components)
    #         gmm.fit(np.array(reward_data).reshape(-1, 1))
    #         means = gmm.means_.flatten()
    #         weights = gmm.weights_.flatten()

    #         # Update the belief mean as a weighted average of GMM components
    #         updated_mean = np.dot(means, weights)
    #         updated_std = np.sqrt(np.dot(weights, (means - updated_mean)**2 + gmm.covariances_.flatten()))
    #         print(updated_mean, updated_std)
    #         belief['mean'] = updated_mean
    #         belief['std'] = updated_std

    #         self.beliefs[(patch_type, env)]['mean'] = updated_mean
    #         self.beliefs[(patch_type, env)]['std'] = updated_std

    #         if (patch_type, env) not in self.mean_history:
    #             self.mean_history[(patch_type, env)] = []
    #         if (patch_type, env) not in self.std_history:
    #             self.std_history[(patch_type, env)] = []
    #         if (patch_type, env) not in self.beliefs[(patch_type, env)]:
    #             self.beliefs[(patch_type, env)]['count'] = 0

    #         self.mean_history[(patch_type, env)].append(belief['mean'])
    #         self.std_history[(patch_type, env)].append(belief['std'])

    def record_reward(self, patch_type, env, reward):
        self.reward_history[(patch_type, env)].append(reward)

    # Method 4: Fixed Decay Parameter with Noise
    def get_reward_with_noise(self, initial_yield, true_decay_rate, time, env):
        self.observation_counts[env] += 1
        current_noise_std = self.noise_std * (self.noise_decay_rate ** self.observation_counts[env])

        # Calculate the reward with decreasing noise
        noise = np.random.normal(0, current_noise_std)
        return max(0.1, initial_yield * np.exp(-(true_decay_rate + noise) * time))

    # # Method 5: Random Walk on Decay Parameter
    # def update_belief_random_walk(self, patch_type, env, step_size=0.001):
    #     belief = self.beliefs[(patch_type, env)]
    #     prior_mean = belief['mean']
    #     random_walk = np.random.normal(0, step_size)
        
    #     # Update with a small random walk
    #     new_mean = prior_mean + random_walk
        
    #     # Keep the mean within reasonable bounds
    #     new_mean = max(0, new_mean)

    #     self.beliefs[(patch_type, env)]['mean'] = new_mean
    #     self.mean_history[(patch_type, env)].append(new_mean)

    # Method 6: Two-Level Bayesian Model
    def update_belief_hierarchical(self, patch_type, env, observed_decay_rate, learning_rate=0.1):
        belief = self.beliefs[(patch_type, env)]
        prior_mean = belief['mean']
        
        # Update environment-level belief while subject-level remains stable
        updated_mean = prior_mean + learning_rate * (observed_decay_rate - prior_mean)
        belief['mean'] = updated_mean

        self.mean_history[(patch_type, env)].append(updated_mean)

    # Method 7: No Update to the Mean Decay Parameter
    def update_belief_no_mean_change(self, patch_type, env, observed_decay_rate):
        belief = self.beliefs[(patch_type, env)]
        belief['count'] += 1
        
        # Only reduce the variance, do not change the mean
        prior_var = belief['std'] ** 2
        new_var = prior_var * belief['count'] / (belief['count'] + 1)
        belief['std'] = np.sqrt(new_var)

        self.std_history[(patch_type, env)].append(np.sqrt(new_var))

    # Existing Methods (Decide Leave, etc.)
    def decide_leave(self, reward, average_reward):
        return reward < average_reward
    
class AgentWithMemory:
    def __init__(self, decay_rate_means, decay_rate_stds, method='bayesian', observation_var=0.00001, noise_std=0.01, learning_rate=0.1, ema_alpha=0.1, n_components=2, memory_size=10):
        self.decay_rate_means = decay_rate_means
        self.decay_rate_stds = decay_rate_stds
        self.observation_var = observation_var
        self.noise_std = noise_std
        self.learning_rate = learning_rate
        self.ema_alpha = ema_alpha
        self.n_components = n_components
        self.method = method
        self.memory_size = memory_size

        self.beliefs = {(i+1, env): {'mean': decay_rate_means[i], 'std': decay_rate_stds[i], 'count': 0}
                        for i in range(len(decay_rate_means)) for env in [1, 2]}
        self.mean_history = {(i+1, env): [] for i in range(len(decay_rate_means)) for env in [1, 2]}
        self.std_history = {(i+1, env): [] for i in range(len(decay_rate_means)) for env in [1, 2]}
        self.reward_history = {(i+1, env): deque(maxlen=self.memory_size) for i in range(len(decay_rate_means)) for env in [1, 2]}

    def sample_decay_rate(self, patch_type, env):
        belief = self.beliefs[(patch_type, env)]
        return np.random.normal(belief['mean'], belief['std'])

    def update_belief(self, patch_type, env, observed_decay_rate=None, observed_reward=None, expected_reward=None):
        if self.method == 'bayesian':
            self.update_belief_bayesian(patch_type, env)
        elif self.method == 'rescorla_wagner':
            self.update_belief_rescorla_wagner(patch_type, env, observed_reward, expected_reward)
        elif self.method == 'ema':
            self.update_belief_ema(patch_type, env)
        elif self.method == 'gmm':
            self.update_belief_gmm(patch_type, env)
        elif self.method == 'fixed_noise':
            self.get_reward_with_noise(patch_type, env, observed_decay_rate)
        elif self.method == 'random_walk':
            self.update_belief_random_walk(patch_type, env)
        elif self.method == 'hierarchical':
            self.update_belief_hierarchical(patch_type, env)
        elif self.method == 'no_mean_change':
            self.update_belief_no_mean_change(patch_type, env)

    def update_belief_bayesian(self, patch_type, env):
        reward_history = np.array(self.reward_history[(patch_type, env)])
        if len(reward_history) > 0:
            observed_decay_rate = -np.log(reward_history[-1] / reward_history[0]) / len(reward_history)
            belief = self.beliefs[(patch_type, env)]
            prior_mean = belief['mean']
            prior_var = belief['std'] ** 2
            belief['count'] += 1
            count = belief['count']

            if self.observation_var > 0:
                adjusted_observation_var = self.observation_var / count
                posterior_mean = (prior_var * observed_decay_rate + adjusted_observation_var * prior_mean) / (prior_var + adjusted_observation_var)
                posterior_var = (prior_var * adjusted_observation_var) / (prior_var + adjusted_observation_var)
            else:
                posterior_mean = (prior_mean * (count - 1) + observed_decay_rate) / count
                posterior_var = prior_var * (count - 1) / count

            self.beliefs[(patch_type, env)]['mean'] = posterior_mean
            self.beliefs[(patch_type, env)]['std'] = np.sqrt(posterior_var)

            self.mean_history[(patch_type, env)].append(posterior_mean)
            self.std_history[(patch_type, env)].append(np.sqrt(posterior_var))

    def update_belief_rescorla_wagner(self, patch_type, env, observed_reward=None, expected_reward=None):
        reward_history = np.array(self.reward_history[(patch_type, env)])
        if len(reward_history) > 0:
            prediction_error = observed_reward - expected_reward if observed_reward is not None else reward_history[-1] - np.mean(reward_history[:-1])
            belief = self.beliefs[(patch_type, env)]
            updated_mean = belief['mean'] + self.learning_rate * prediction_error

            updated_mean = max(updated_mean, 0)
            belief['mean'] = updated_mean
            self.mean_history[(patch_type, env)].append(updated_mean)

    def update_belief_ema(self, patch_type, env):
        reward_history = np.array(self.reward_history[(patch_type, env)])
        if len(reward_history) > 0:
            observed_decay_rate = -np.log(reward_history[-1] / reward_history[0]) / len(reward_history)
            belief = self.beliefs[(patch_type, env)]
            updated_mean = self.ema_alpha * observed_decay_rate + (1 - self.ema_alpha) * belief['mean']

            belief['mean'] = updated_mean
            self.mean_history[(patch_type, env)].append(updated_mean)

    def update_belief_gmm(self, patch_type, env):
        reward_history = np.array(self.reward_history[(patch_type, env)]).reshape(-1, 1)
        if len(reward_history) >= self.n_components:
            gmm = GaussianMixture(n_components=self.n_components)
            gmm.fit(reward_history)
            means = gmm.means_.flatten()
            weights = gmm.weights_.flatten()

            updated_mean = np.dot(means, weights)
            updated_std = np.sqrt(np.dot(weights, (means - updated_mean)**2 + gmm.covariances_.flatten()))

            belief = self.beliefs[(patch_type, env)]
            belief['mean'] = updated_mean
            belief['std'] = updated_std

            self.mean_history[(patch_type, env)].append(updated_mean)
            self.std_history[(patch_type, env)].append(updated_std)

    def record_reward(self, patch_type, env, reward):
        self.reward_history[(patch_type, env)].append(reward)

    def update_belief_random_walk(self, patch_type, env, step_size=0.001):
        belief = self.beliefs[(patch_type, env)]
        random_walk = np.random.normal(0, step_size)
        belief['mean'] += random_walk
        belief['mean'] = max(belief['mean'], 0)
        self.mean_history[(patch_type, env)].append(belief['mean'])

    def update_belief_hierarchical(self, patch_type, env):
        belief = self.beliefs[(patch_type, env)]
        reward_history = np.array(self.reward_history[(patch_type, env)])
        if len(reward_history) > 0:
            observed_decay_rate = -np.log(reward_history[-1] / reward_history[0]) / len(reward_history)
            updated_mean = belief['mean'] + self.learning_rate * (observed_decay_rate - belief['mean'])
            belief['mean'] = updated_mean
            self.mean_history[(patch_type, env)].append(updated_mean)

    def update_belief_no_mean_change(self, patch_type, env):
        belief = self.beliefs[(patch_type, env)]
        belief['count'] += 1
        prior_var = belief['std'] ** 2
        new_var = prior_var * belief['count'] / (belief['count'] + 1)
        belief['std'] = np.sqrt(new_var)
        self.std_history[(patch_type, env)].append(belief['std'])
                                                          
    def decide_leave(self, reward, average_reward):
        return reward < average_reward