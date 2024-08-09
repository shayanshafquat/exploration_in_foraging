import sys
import scipy.io
import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
from world import Patch, Agent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the patches
patches = {
    'low': Patch(32.5, 0.075, 'exponential'),
    'med': Patch(45, 0.075, 'exponential'),
    'high': Patch(57.5, 0.075, 'exponential')
}

def load_data():
    logging.info("Loading data...")
    data = scipy.io.loadmat('../leheron_trialbytrial/leheron_blockSwitchIndex.mat')
    block_order_df = pd.read_csv('../leheron_trialbytrial/leheron_blockOrder.csv')
    df_trials = pd.read_csv("../leheron_trialbytrial/leheron_trialbytrial.csv")
    return df_trials, block_order_df, data

def group_data(df_trials):
    logging.info("Grouping trial data by subject, environment, and patch...")
    return df_trials.groupby(['sub', 'env', 'patch']).agg(
        mean_leaveT=('leaveT', 'mean'),
        count=('leaveT', 'count')
    ).reset_index()

def simulate_agent_in_patch(policy_type, parameter, intercept, patch, mellowmax_type=None, max_timesteps=200):
    agent_kwargs = {
        'policy_type': policy_type,
        'beta': parameter if policy_type == 'softmax' else None,
        'intercept': intercept,
        'omega': parameter if policy_type == 'mellowmax' else None,
    }
    if policy_type == 'mellowmax':
        agent_kwargs['mellowmax_type'] = mellowmax_type

    agent = Agent(**agent_kwargs)

    patch.start_harvesting()
    cumulative_prob = 1.0
    expected_leave_time = 0.0

    for n in range(1, max_timesteps + 1):
        reward = patch.get_reward()
        prob_leave_now = agent.get_leave_probability(reward)
        prob_leave_n = prob_leave_now * cumulative_prob
        expected_leave_time += n * prob_leave_n
        cumulative_prob *= (1 - prob_leave_now)

    return expected_leave_time

def objective(params, df, patches, policy_type, mellowmax_type=None, fix_intercept=None, fix_parameter=None):
    if fix_parameter is not None:
        parameter = fix_parameter
        intercept = params[0]
    elif fix_intercept is not None:
        parameter = params[0]
        intercept = fix_intercept
    else:
        parameter, intercept = params

    total_error = 0
    total_count = df['count'].sum()
    patch_mapping = {1: 'low', 2: 'med', 3: 'high'}
    
    for _, row in df.iterrows():
        patch_type, actual_mean, count = row['patch'], row['mean_leaveT'], row['count']
        patch_name = patch_mapping[patch_type]
        patch = patches[patch_name]
        predicted_mean = simulate_agent_in_patch(policy_type, parameter, intercept, patch, mellowmax_type)
        error = count * (predicted_mean - actual_mean) ** 2
        total_error += error
        
    rmse = np.sqrt(total_error / total_count)
    return rmse

def fit_parameters_for_subject(df_sub, patches, policy_type, mellowmax_type=None):
    initial_guess = [0.3, -3] if policy_type == 'softmax' else [0.3, 0]
    result = minimize(objective, initial_guess, args=(df_sub, patches, policy_type, mellowmax_type), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    fitted_parameter, fitted_intercept = result.x
    return fitted_parameter, fitted_intercept

def calculate_weighted_means(df_sub_env, fitted_parameter, fitted_intercept, patches, policy_type, mellowmax_type=None):
    weighted_predicted_leaveT = 0
    weighted_actual_leaveT = 0
    total_count = df_sub_env['count'].sum()
    patch_mapping = {1: 'low', 2: 'med', 3: 'high'}
    
    for _, row in df_sub_env.iterrows():
        patch_name = patch_mapping[row['patch']]
        patch = patches[patch_name]
        predicted_leave_time = simulate_agent_in_patch(policy_type, fitted_parameter, fitted_intercept, patch, mellowmax_type)
        weighted_predicted_leaveT += row['count'] * predicted_leave_time
        weighted_actual_leaveT += row['count'] * row['mean_leaveT']

    weighted_predicted_leaveT /= total_count
    weighted_actual_leaveT /= total_count
    
    return weighted_predicted_leaveT, weighted_actual_leaveT

def run_case_1(grouped_df, patches, policy_type, mellowmax_type=None):
    logging.info(f"Running Case 1: Fix Parameter and Fix Intercept for {policy_type} policy...")
    results = []
    subject_params = {}
    
    for sub in grouped_df['sub'].unique():
        df_sub = grouped_df[grouped_df['sub'] == sub]
        fitted_parameter, fitted_intercept = fit_parameters_for_subject(df_sub, patches, policy_type, mellowmax_type)
        subject_params[sub] = (fitted_parameter, fitted_intercept)
        
        for env in df_sub['env'].unique():
            df_sub_env = df_sub[df_sub['env'] == env]
            if not df_sub_env.empty:
                weighted_predicted_leaveT, weighted_actual_leaveT = calculate_weighted_means(df_sub_env, fitted_parameter, fitted_intercept, patches, policy_type, mellowmax_type)
                results.append({
                    'sub': sub,
                    'env': env,
                    'case': 'fix_parameter_fix_c',
                    'policy_type': policy_type,
                    'fitted_parameter': fitted_parameter,
                    'fitted_intercept': fitted_intercept,
                    'predicted_leaveT': weighted_predicted_leaveT,
                    'actual_mean_leaveT': weighted_actual_leaveT
                })
    
    return results, subject_params

def run_case_2(grouped_df, patches, subject_params, policy_type, mellowmax_type=None):
    logging.info(f"Running Case 2: Vary Parameter and Vary Intercept for {policy_type} policy...")
    results = []
    
    for sub in grouped_df['sub'].unique():
        for env in grouped_df['env'].unique():
            df_sub_env = grouped_df[(grouped_df['sub'] == sub) & (grouped_df['env'] == env)]
            if not df_sub_env.empty:
                initial_guess = list(subject_params[sub])
                result = minimize(objective, initial_guess, args=(df_sub_env, patches, policy_type, mellowmax_type), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
                fitted_parameter, fitted_intercept = result.x
                weighted_predicted_leaveT, weighted_actual_leaveT = calculate_weighted_means(df_sub_env, fitted_parameter, fitted_intercept, patches, policy_type, mellowmax_type)
                results.append({
                'sub': sub,
                'env': env,
                'case': 'vary_parameter_vary_c',
                'policy_type': policy_type,
                'fitted_parameter': fitted_parameter,
                'fitted_intercept': fitted_intercept,
                'predicted_leaveT': weighted_predicted_leaveT,
                'actual_mean_leaveT': weighted_actual_leaveT
            })
    return results

def run_case_3(grouped_df, patches, subject_params, policy_type, mellowmax_type=None):
    logging.info(f"Running Case 3: Vary Parameter and Fix Intercept for {policy_type} policy…")
    results = []
    
    for sub in grouped_df['sub'].unique():
        fixed_intercept = subject_params[sub][1]
        for env in grouped_df['env'].unique():
            df_sub_env = grouped_df[(grouped_df['sub'] == sub) & (grouped_df['env'] == env)]
            if not df_sub_env.empty:
                initial_guess = [subject_params[sub][0]]
                result = minimize(objective, initial_guess, args=(df_sub_env, patches, policy_type, mellowmax_type, fixed_intercept), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
                fitted_parameter = result.x[0]
                weighted_predicted_leaveT, weighted_actual_leaveT = calculate_weighted_means(df_sub_env, fitted_parameter, fixed_intercept, patches, policy_type, mellowmax_type)
                results.append({
                    'sub': sub,
                    'env': env,
                    'case': 'vary_parameter_fix_c',
                    'policy_type': policy_type,
                    'fitted_parameter': fitted_parameter,
                    'fitted_intercept': fixed_intercept,
                    'predicted_leaveT': weighted_predicted_leaveT,
                    'actual_mean_leaveT': weighted_actual_leaveT
                })

    return results

def run_case_4(grouped_df, patches, subject_params, policy_type, mellowmax_type=None):
    logging.info(f"Running Case 4: Fix Parameter and Vary Intercept for {policy_type} policy…")
    results = []
    
    for sub in grouped_df['sub'].unique():
        fixed_parameter = subject_params[sub][0]
        for env in grouped_df['env'].unique():
            df_sub_env = grouped_df[(grouped_df['sub'] == sub) & (grouped_df['env'] == env)]
            if not df_sub_env.empty:
                initial_guess = [subject_params[sub][1]]
                result = minimize(objective, initial_guess, args=(df_sub_env, patches, policy_type, mellowmax_type, None, fixed_parameter), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
                fitted_intercept = result.x[0]
                weighted_predicted_leaveT, weighted_actual_leaveT = calculate_weighted_means(df_sub_env, fixed_parameter, fitted_intercept, patches, policy_type, mellowmax_type)
                results.append({
                    'sub': sub,
                    'env': env,
                    'case': 'fix_parameter_vary_c',
                    'policy_type': policy_type,
                    'fitted_parameter': fixed_parameter,
                    'fitted_intercept': fitted_intercept,
                    'predicted_leaveT': weighted_predicted_leaveT,
                    'actual_mean_leaveT': weighted_actual_leaveT
                })

    return results

# def log_likelihood(params, df, patches, policy_type, mellowmax_type=None):
#     """Calculate the negative log-likelihood for a given set of parameters."""
#     omega, intercept = params

#     total_log_likelihood = 0
#     patch_mapping = {1: 'low', 2: 'med', 3: 'high'}  # Assuming these are the patch type mappings

#     # Iterate over each trial in df_trials
#     for _, row in df.iterrows():
#         sub, patch_type, env, leaveT, mean_leaveT = row['sub'], row['patch'], row['env'], row['leaveT'], row['meanLT']
#         patch = patches[patch_mapping[patch_type]]  # Get the corresponding patch

#         patch.start_harvesting()  # Reset the patch harvesting process
#         agent = Agent(policy_type=policy_type, omega=omega, intercept=intercept, mellowmax_type=mellowmax_type)

#         # Simulate the harvesting until the observed leave time
#         for t in range(int(leaveT)):
#             reward = patch.get_reward()

#             if t + 1 == int(leaveT):
#                 leave_proba = agent.get_leave_probability(reward)
#                 total_log_likelihood += np.log(leave_proba)
#             else:
#                 stay_proba = 1 - agent.get_leave_probability(reward)
#                 total_log_likelihood += np.log(stay_proba)

#     return -total_log_likelihood  # Negative because we are minimizing

# def main():
#     # Load data
#     df_trials, block_order_df, data = load_data()

#     # Define the patches with their respective initial yields and decay rates
#     patches = {
#         'low': Patch(32.5, 0.075, 'exponential'),
#         'med': Patch(45, 0.075, 'exponential'),
#         'high': Patch(57.5, 0.075, 'exponential')
#     }

#     # Policy type and mellowmax type
#     policy_type = 'mellowmax'  # Change this to 'softmax' or 'mellowmax' as needed
#     mellowmax_type = 'add' if policy_type == 'mellowmax' else None  # Change this to 'add' or 'denom' as needed (only relevant for mellowmax)

#     # Choose fitting method: 'analytical' or 'log_likelihood'
#     fitting_method = 'log_likelihood'  # or 'analytical'

#     # Prepare data based on fitting method
#     if fitting_method == 'analytical':
#         grouped_df = df_trials.groupby(['sub', 'env', 'patch']).agg(
#             mean_leaveT=('leaveT', 'mean'),
#             count=('leaveT', 'count')
#         ).reset_index()
#     else:
#         grouped_df = df_trials

#     # Run all cases
#     case_1_results, subject_params = run_case_1(grouped_df, patches, policy_type, mellowmax_type)
#     case_2_results = run_case_2(grouped_df, patches, subject_params, policy_type, mellowmax_type)
#     case_3_results = run_case_3(grouped_df, patches, subject_params, policy_type, mellowmax_type)
#     case_4_results = run_case_4(grouped_df, patches, subject_params, policy_type, mellowmax_type)

#     # Combine all results
#     logging.info("Combining results...")
#     all_results = case_1_results + case_2_results + case_3_results + case_4_results
#     results_df = pd.DataFrame(all_results)

#     # Save the results to CSV
#     logging.info("Saving results to CSV...")
#     results_df.to_csv(f'optimization_results_{policy_type}_{mellowmax_type}_{fitting_method}.csv', index=False)

#     logging.info("Optimization completed successfully!")

def main():
    # Load data
    df_trials, block_order_df, data = load_data()

    # Group data
    grouped_df = group_data(df_trials)

    # Policy type and mellowmax type
    policy_type = 'mellowmax'  # Change this to 'softmax' or 'mellowmax' as needed
    mellowmax_type = 'add' if policy_type == 'mellowmax' else None  # Change this to 'add' or 'denom' as needed (only relevant for mellowmax)

    # Run all cases
    case_1_results, subject_params = run_case_1(grouped_df, patches, policy_type, mellowmax_type)
    case_2_results = run_case_2(grouped_df, patches, subject_params, policy_type, mellowmax_type)
    case_3_results = run_case_3(grouped_df, patches, subject_params, policy_type, mellowmax_type)
    case_4_results = run_case_4(grouped_df, patches, subject_params, policy_type, mellowmax_type)

    # Combine all results
    logging.info("Combining results...")
    all_results = case_1_results + case_2_results + case_3_results + case_4_results
    results_df = pd.DataFrame(all_results)

    # Save the results to CSV
    logging.info("Saving results to CSV...")
    results_df.to_csv(f'optimization_results_{policy_type}_{mellowmax_type}.csv', index=False)

    logging.info("Optimization completed successfully!")

if __name__ == "__main__":
    main()
