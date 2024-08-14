import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from utils import prepare_avg_rewards

class Agent:
    def __init__(self, decay_rate_means, decay_rate_stds, observation_var=0.01):
        self.decay_rate_means = decay_rate_means
        self.decay_rate_stds = decay_rate_stds
        self.observation_var = observation_var
        self.beliefs = {(i+1, env): {'mean': decay_rate_means[i], 'std': decay_rate_stds[i]}
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
        observation_var = self.observation_var

        posterior_mean = (prior_var * observed_decay_rate + observation_var * prior_mean) / (prior_var + observation_var)
        posterior_var = (prior_var * observation_var) / (prior_var + observation_var)

        self.beliefs[(patch_type, env)]['mean'] = posterior_mean
        self.beliefs[(patch_type, env)]['std'] = np.sqrt(posterior_var)

        # print(f"Updating history for Patch {patch_type}, Environment {env}:")
        # print(f"  Mean before update: {self.mean_history[(patch_type, env)]}")
        # print(f"  Std before update: {self.std_history[(patch_type, env)]}")
        
        self.mean_history[(patch_type, env)].append(self.beliefs[(patch_type, env)]['mean'])
        self.std_history[(patch_type, env)].append(self.beliefs[(patch_type, env)]['std'])

        # print(f"  Mean after update: {self.mean_history[(patch_type, env)]}")
        # print(f"  Std after update: {self.std_history[(patch_type, env)]}")
        # print('---')

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
                reward = max(0, self.initial_yield * np.exp(-self.decay_rate * self.time))
            elif self.decay_type == 'linear':
                reward = max(0, self.initial_yield - self.decay_rate * self.time)
            else:
                raise ValueError("Invalid decay type. Use 'exponential' or 'linear'.")
            self.time += 1
            return reward
        else:
            return self.initial_yield
        
class Simulation:
    def __init__(self, decay_rate_means, decay_rate_stds, avg_rewards):
        self.decay_rate_means = decay_rate_means
        self.decay_rate_stds = decay_rate_stds
        self.avg_rewards = avg_rewards
        self.patch_types = self.initialize_env()

    def initialize_env(self):
        patch_types = [
            {'type': 1, 'initial_yield': 32.5, 'decay_rate_mean': self.decay_rate_means[0], 'decay_rate_std': self.decay_rate_stds[0]},
            {'type': 2, 'initial_yield': 45, 'decay_rate_mean': self.decay_rate_means[1], 'decay_rate_std': self.decay_rate_stds[1]},
            {'type': 3, 'initial_yield': 57.5, 'decay_rate_mean': self.decay_rate_means[2], 'decay_rate_std': self.decay_rate_stds[2]}
        ]
        return patch_types

    def get_patch_info(self, patch_type):
        for patch in self.patch_types:
            if patch['type'] == patch_type:
                return patch
        raise ValueError("Patch type not found")

    def simulate_subject(self, subject_df, agent, avg_reward, env, n_max=1000):
        print(f"Simulating subject in environment {env}")
        patch_sequence = []
        leave_times = []
        for _, trial in subject_df.iterrows():
            patch_info = self.get_patch_info(trial['patch'])
            patch = Patch(
                patch_info['initial_yield'], 
                patch_info['decay_rate_mean']
            )
            patch.start_harvesting()

            sampled_decay_rate = agent.sample_decay_rate(trial['patch'], env)
            # print(f"Sampled decay rate for patch {trial['patch']} in environment {env}: {sampled_decay_rate}")
            patch.decay_rate = sampled_decay_rate

            for t in range(1, n_max + 1):
                reward = patch.get_reward()

                if agent.decide_leave(reward, avg_reward):
                    observed_decay_rate = -np.log(reward / patch_info['initial_yield']) / t if t > 0 else sampled_decay_rate
                    # print(f"Observed decay rate for patch {trial['patch']} in environment {env} at time {t}: {observed_decay_rate}")
                    agent.update_belief(trial['patch'], env, observed_decay_rate)
                    patch_sequence.append(trial['patch'])
                    leave_times.append(t)
                    break

        return patch_sequence, leave_times

    def run_simulation(self, df_trials, subject_id=None):
        simulated_leave_times = []
        if subject_id is not None:
            subject_df = df_trials[df_trials['sub'] == subject_id]
            patch_sequence = []
            agent = Agent(decay_rate_means=self.decay_rate_means, decay_rate_stds=self.decay_rate_stds)
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
                    agent = Agent(decay_rate_means=self.decay_rate_means, decay_rate_stds=self.decay_rate_stds)
                    subject_env_df = subject_df[subject_df['env'] == env]
                    _, leave_times = self.simulate_subject(subject_env_df, agent, avg_reward, env)
                    simulated_leave_times.extend(leave_times)
            df_trials['simulated_leaveT'] = simulated_leave_times
            df_trials['simulated_dmLeave'] = df_trials['simulated_leaveT'] - df_trials['meanLT']
            return df_trials