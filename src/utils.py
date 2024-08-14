import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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