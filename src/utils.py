import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from world import Patch, Agent
from matplotlib.lines import Line2D

matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

def plot_model_fit_comparisons_to_subject_leaving_times(mean_times, sub_col_name, model_col_name):
    title = sub_col_name.split('_')[0]
    # Map environment numbers to labels
    env_labels = {1: 'Rich', 2: 'Poor'}
    env_colors = {1: 'blue', 2: 'green'}
    environments = [1, 2]

    # Plot comparison
    plt.figure(figsize=(6, 3))

    for i, env in enumerate(environments, 1):
        env_data = mean_times[mean_times['env'] == env]
        
        plt.subplot(1, len(environments), i)
        plt.scatter(env_data[sub_col_name], env_data[model_col_name], alpha=0.6, color=env_colors[env])
        plt.plot([0, env_data[sub_col_name].max()], [0, env_data[sub_col_name].max()], 'k--')
        
        # Calculate correlation coefficient
        r, _ = pearsonr(env_data[sub_col_name], env_data[model_col_name])
        
        plt.xlabel(f'{title} Subject Leaving Times (s)')
        plt.ylabel(f'{title} Model Leaving Times (s)')
        plt.title(f'{env_labels[env]} Environment\nr = {r:.2f}')
        plt.xlim(0, env_data[sub_col_name].max() + 10)
        plt.ylim(0, env_data[model_col_name].max() + 10)

    plt.tight_layout()
    plt.show()

def prepare_avg_rewards(df_trials, subject_ids, proportions, mvt_model):
    avg_rewards = {}
    for subject_id in subject_ids:
        subject_proportions = proportions.loc[subject_id]
        for env in subject_proportions.index.get_level_values('env').unique():
            patch_proportions = subject_proportions.loc[env].values
            avg_reward = mvt_model.get_average_reward_rate(patch_proportions)
            avg_rewards[(subject_id, env)] = avg_reward
    return avg_rewards

def decay_param_evolution(agent):
    # Assuming agent_2 is the Agent instance from the simulation for Subject 2
    # Plot the trajectory of the mean and standard deviation of the decay rate for Subject 2
    fig, axes = plt.subplots(2, 3, figsize=(12, 5))

    patch_types = [1, 2, 3]
    patch_labels = ['Low', 'Mid', 'High']
    env_labels = {1: 'Rich', 2: 'Poor'}
    env_rows = {1: 0, 2: 1}  # Rich environment in the first row, Poor in the second row

    for patch_type in patch_types:
        for env in [1, 2]:
            mean_values = agent.mean_history[(patch_type, env)]
            std_values = agent.std_history[(patch_type, env)]
        
            if not mean_values:  # Check if the data exists
                print(f'No data for Patch {patch_type}, Environment {env}')
                continue
        
            ax = axes[env_rows[env], patch_type-1]  # Select the correct subplot

        # Time steps based on the length of the history
            time_steps = range(1, len(mean_values) + 1)
        
            ax.plot(time_steps, mean_values, label=f'{patch_labels[patch_type-1]} Patch {env_labels[env]} Mean', color='blue' if env == 1 else 'green')
            ax.fill_between(time_steps,
                        [mean - std for mean, std in zip(mean_values, std_values)],
                        [mean + std for mean, std in zip(mean_values, std_values)],
                        color='blue' if env == 1 else 'green', alpha=0.2)
        
            ax.set_title(f'{patch_labels[patch_type-1]} Patch ({env_labels[env]} Env)')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Decay Rate')
            ax.legend()

    plt.suptitle('Mean and SD of Decay Rate Trajectory for Subject 2', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def compare_mean_to_subject(df_trials_with_simulation, MVT_rich, MVT_poor, method):
    subject_grouped_observed = df_trials_with_simulation.groupby(['sub', 'patch', 'env'])['leaveT'].mean().unstack(level=-1)
    subject_grouped_simulated = df_trials_with_simulation.groupby(['sub', 'patch', 'env'])['simulated_leaveT'].mean().unstack(level=-1)

    mean_leave_observed = subject_grouped_observed.groupby('patch').mean()
    sem_leave_observed = subject_grouped_observed.groupby('patch').sem()

    mean_leave_simulated = subject_grouped_simulated.groupby('patch').mean()
    sem_leave_simulated = subject_grouped_simulated.groupby('patch').sem()

    patch_types = ['Low', 'Mid', 'High']
    patch_indices = np.arange(len(patch_types))

    plt.figure(figsize=(4, 4))

    # Adjusted labels with apostrophe
    plt.errorbar(patch_indices, mean_leave_observed[1], yerr=sem_leave_observed[1], fmt='--o', color='blue')
    plt.errorbar(patch_indices, mean_leave_observed[2], yerr=sem_leave_observed[2], fmt='--o', color='green')

    plt.errorbar(patch_indices, mean_leave_simulated[1], yerr=sem_leave_simulated[1], fmt='-o', color='blue')
    plt.errorbar(patch_indices, mean_leave_simulated[2], yerr=sem_leave_simulated[2], fmt='-o', color='green')

    plt.plot(patch_indices, MVT_rich, 'o--', color='grey', linewidth=1, markersize=4)
    plt.plot(patch_indices, MVT_poor, 'o--', color='grey', linewidth=1, markersize=4)

    plt.xticks(patch_indices, patch_types, fontsize=14)
    plt.xlabel('Patch Type', fontsize=14)
    plt.ylabel('Mean Leaving Time (s)', fontsize=14)
    # plt.title('Subjects', fontsize=14)
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label="Rich", linestyle='--'),
        Line2D([0], [0], color='green', lw=2, label="Poor", linestyle='--'),
        Line2D([0], [0], color='black', lw=2, label="Model", linestyle='-')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=14)

    # Remove top and right borders
    sns.despine()
    plt.tight_layout()
    save_path = f'../plots/uncertainty/{method}_sd.png'
    plt.savefig(save_path, dpi=300)
    plt.show()

def compare_variability_to_subject(df_trials_with_simulation, method):
    subject_grouped_observed = df_trials_with_simulation.groupby(['sub', 'patch', 'env'])['leaveT'].std().unstack(level=-1)
    subject_grouped_simulated = df_trials_with_simulation.groupby(['sub', 'patch', 'env'])['simulated_leaveT'].std().unstack(level=-1)

    mean_sd_observed = subject_grouped_observed.groupby('patch').mean()
    sem_sd_observed = subject_grouped_observed.groupby('patch').sem()

    mean_sd_simulated = subject_grouped_simulated.groupby('patch').mean()
    sem_sd_simulated = subject_grouped_simulated.groupby('patch').sem()

    patch_types = ['Low', 'Mid', 'High']
    patch_indices = np.arange(len(patch_types))

    plt.figure(figsize=(4, 4))

    # Adjusted labels and linestyle for subject data
    plt.errorbar(patch_indices, mean_sd_observed[1], yerr=sem_sd_observed[1], fmt='--o', color='blue')
    plt.errorbar(patch_indices, mean_sd_observed[2], yerr=sem_sd_observed[2], fmt='--o', color='green')

    # Model data
    plt.errorbar(patch_indices, mean_sd_simulated[1], yerr=sem_sd_simulated[1], fmt='-o', color='blue')
    plt.errorbar(patch_indices, mean_sd_simulated[2], yerr=sem_sd_simulated[2], fmt='-o', color='green')

    plt.xticks(patch_indices, patch_types, fontsize=14)
    plt.xlabel('Patch Type', fontsize=14)
    plt.ylabel(r'SD$_{\mathrm{leave}}$ (s)', fontsize=14)
    # plt.title('Standard Deviation in Leaving Times by Patch Type', fontsize=14)
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label="Rich", linestyle='--'),
        Line2D([0], [0], color='green', lw=2, label="Poor", linestyle='--'),
        Line2D([0], [0], color='black', lw=2, label="Model", linestyle='-')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=14)

    # Remove top and right borders
    sns.despine()

    plt.tight_layout()
    save_path = f'../plots/uncertainty/{method}_mean.png'
    plt.savefig(save_path, dpi=300)
    plt.show()

def calculate_aic_bic(n, rss, k):
    # Calculate log-likelihood
    ll = -n / 2 * (np.log(2 * np.pi * rss / n) + 1)
    
    # Calculate AIC and BIC
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    
    return aic, bic

# Function to calculate AIC and BIC for each subject
def calculate_subject_aic_bic(df, pred_col_name, k):
    results = []
    subjects = df['sub'].unique()
    
    for sub in subjects:
        sub_data = df[df['sub'] == sub]
        n = len(sub_data)
        rss = np.sum((sub_data['actual_mean_leaveT'] - sub_data[pred_col_name]) ** 2)
        aic, bic = calculate_aic_bic(n, rss, k)
        results.append({
            'sub': sub,
            'aic': aic,
            'bic': bic
        })
        
    results_df = pd.DataFrame(results)
    return results_df

def calculate_leave_statistics(policy_type, parameter, intercept, patch, mellowmax_type=None, max_timesteps=200, use_intercept=True):
    if policy_type == 'epsilon_greedy':
        epsilon = parameter
        expected_leave_time = 1 / epsilon
        variance_leave_time = (1 - epsilon) / (epsilon ** 2)
        std_leave_time = np.sqrt(variance_leave_time)
        return expected_leave_time, std_leave_time
    
    agent_kwargs = {
        'policy_type': policy_type,
        'beta': parameter if policy_type == 'softmax' else None,
        'intercept': intercept,
        'omega': parameter if policy_type == 'mellowmax' else None,
        'use_intercept': use_intercept
    }
    if policy_type == 'mellowmax':
        agent_kwargs['mellowmax_type'] = mellowmax_type

    agent = Agent(**agent_kwargs)
    patch.start_harvesting()
    expected_leave_time = 0.0
    expected_leave_time_squared = 0.0  # To accumulate n^2 * p(leave at n)
    cumulative_prob = 1.0

    for n in range(1, max_timesteps + 1):
        reward = patch.get_reward()
        prob_leave_now = agent.get_leave_probability(reward)
        p_leave_n = prob_leave_now * cumulative_prob
        
        # Expected leave time E(leave)
        expected_leave_time += n * p_leave_n
        
        # Accumulate expected leave time squared
        expected_leave_time_squared += (n ** 2) * p_leave_n
        
        # Update cumulative probability
        cumulative_prob *= (1 - prob_leave_now)
    
    # Compute variance from E(X^2) - (E(X))^2
    variance_leave_time = expected_leave_time_squared - (expected_leave_time ** 2)
    std_leave_time = np.sqrt(variance_leave_time)
    
    return expected_leave_time, std_leave_time