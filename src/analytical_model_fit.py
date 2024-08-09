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

def simulate_agent_in_patch(beta, intercept, patch, max_timesteps=300):
    agent = Agent(policy_type='softmax', beta=beta, intercept=intercept)
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

def objective(params, df, patches, fix_intercept=None, fix_beta=None):
    if fix_beta is not None:
        beta = fix_beta
        intercept = params[0]
    elif fix_intercept is not None:
        beta = params[0]
        intercept = fix_intercept
    else:
        beta, intercept = params

    total_error = 0
    total_count = df['count'].sum()
    patch_mapping = {1: 'low', 2: 'med', 3: 'high'}
    
    for _, row in df.iterrows():
        patch_type, actual_mean, count = row['patch'], row['mean_leaveT'], row['count']
        patch_name = patch_mapping[patch_type]
        patch = patches[patch_name]
        predicted_mean = simulate_agent_in_patch(beta, intercept, patch)
        error = count * (predicted_mean - actual_mean) ** 2
        total_error += error
        
    rmse = np.sqrt(total_error / total_count)
    return rmse

def fit_parameters_for_subject(df_sub, patches):
    initial_guess = [0.3, -3]
    result = minimize(objective, initial_guess, args=(df_sub, patches), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    fitted_beta, fitted_intercept = result.x
    return fitted_beta, fitted_intercept

def calculate_weighted_means(df_sub_env, fitted_beta, fitted_intercept, patches):
    weighted_predicted_leaveT = 0
    weighted_actual_leaveT = 0
    total_count = df_sub_env['count'].sum()
    patch_mapping = {1: 'low', 2: 'med', 3: 'high'}
    
    for _, row in df_sub_env.iterrows():
        patch_name = patch_mapping[row['patch']]
        patch = patches[patch_name]
        predicted_leave_time = simulate_agent_in_patch(fitted_beta, fitted_intercept, patch)
        weighted_predicted_leaveT += row['count'] * predicted_leave_time
        weighted_actual_leaveT += row['count'] * row['mean_leaveT']

    weighted_predicted_leaveT /= total_count
    weighted_actual_leaveT /= total_count
    
    return weighted_predicted_leaveT, weighted_actual_leaveT

def run_case_1(grouped_df, patches):
    logging.info("Running Case 1: Fix Beta and Fix Intercept...")
    results = []
    subject_params = {}
    
    for sub in grouped_df['sub'].unique():
        df_sub = grouped_df[grouped_df['sub'] == sub]
        fitted_beta, fitted_intercept = fit_parameters_for_subject(df_sub, patches)
        subject_params[sub] = (fitted_beta, fitted_intercept)
        
        for env in df_sub['env'].unique():
            df_sub_env = df_sub[df_sub['env'] == env]
            if not df_sub_env.empty:
                weighted_predicted_leaveT, weighted_actual_leaveT = calculate_weighted_means(df_sub_env, fitted_beta, fitted_intercept, patches)
                results.append({
                    'sub': sub,
                    'env': env,
                    'case': 'fix_beta_fix_c',
                    'fitted_beta': fitted_beta,
                    'fitted_intercept': fitted_intercept,
                    'predicted_leaveT': weighted_predicted_leaveT,
                    'actual_mean_leaveT': weighted_actual_leaveT
                })
    
    return results, subject_params

def run_case_2(grouped_df, patches, subject_params):
    logging.info("Running Case 2: Vary Beta and Vary Intercept...")
    results = []
    
    for sub in grouped_df['sub'].unique():
        for env in grouped_df['env'].unique():
            df_sub_env = grouped_df[(grouped_df['sub'] == sub) & (grouped_df['env'] == env)]
            if not df_sub_env.empty:
                initial_guess = list(subject_params[sub])
                result = minimize(objective, initial_guess, args=(df_sub_env, patches), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
                fitted_beta, fitted_intercept = result.x
                weighted_predicted_leaveT, weighted_actual_leaveT = calculate_weighted_means(df_sub_env, fitted_beta, fitted_intercept, patches)
                results.append({
                    'sub': sub,
                    'env': env,
                    'case': 'vary_beta_vary_c',
                    'fitted_beta': fitted_beta,
                    'fitted_intercept': fitted_intercept,
                    'predicted_leaveT': weighted_predicted_leaveT,
                    'actual_mean_leaveT': weighted_actual_leaveT
                })
    
    return results

def run_case_3(grouped_df, patches, subject_params):
    logging.info("Running Case 3: Vary Beta and Fix Intercept...")
    results = []
    
    for sub in grouped_df['sub'].unique():
        fixed_intercept = subject_params[sub][1]
        for env in grouped_df['env'].unique():
            df_sub_env = grouped_df[(grouped_df['sub'] == sub) & (grouped_df['env'] == env)]
            if not df_sub_env.empty:
                initial_guess = [subject_params[sub][0]]
                result = minimize(objective, initial_guess, args=(df_sub_env, patches, fixed_intercept), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
                fitted_beta = result.x[0]
                weighted_predicted_leaveT, weighted_actual_leaveT = calculate_weighted_means(df_sub_env, fitted_beta, fixed_intercept, patches)
                results.append({
                    'sub': sub,
                    'env': env,
                    'case': 'vary_beta_fix_c',
                    'fitted_beta': fitted_beta,
                    'fitted_intercept': fixed_intercept,
                    'predicted_leaveT': weighted_predicted_leaveT,
                    'actual_mean_leaveT': weighted_actual_leaveT
                })
    
    return results

def run_case_4(grouped_df, patches, subject_params):
    logging.info("Running Case 4: Fix Beta and Vary Intercept...")
    results = []
    
    for sub in grouped_df['sub'].unique():
        fixed_beta = subject_params[sub][0]
        for env in grouped_df['env'].unique():
            df_sub_env = grouped_df[(grouped_df['sub'] == sub) & (grouped_df['env'] == env)]
            if not df_sub_env.empty:
                initial_guess = [subject_params[sub][1]]
                result = minimize(objective, initial_guess, args=(df_sub_env, patches, None, fixed_beta), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
                fitted_intercept = result.x[0]
                weighted_predicted_leaveT, weighted_actual_leaveT = calculate_weighted_means(df_sub_env, fixed_beta, fitted_intercept, patches)
                results.append({
                    'sub': sub,
                    'env': env,
                    'case': 'fix_beta_vary_c',
                    'fitted_beta': fixed_beta,
                    'fitted_intercept': fitted_intercept,
                    'predicted_leaveT': weighted_predicted_leaveT,
                    'actual_mean_leaveT': weighted_actual_leaveT
                })
    
    return results

def main():
    # Load data
    df_trials, block_order_df, data = load_data()

    # Group data
    grouped_df = group_data(df_trials)

    # Run all cases
    case_1_results, subject_params = run_case_1(grouped_df, patches)
    case_2_results = run_case_2(grouped_df, patches, subject_params)
    case_3_results = run_case_3(grouped_df, patches, subject_params)
    case_4_results = run_case_4(grouped_df, patches, subject_params)

    # Combine all results
    logging.info("Combining results...")
    all_results = case_1_results + case_2_results + case_3_results + case_4_results
    results_df = pd.DataFrame(all_results)

    # Save the results to CSV
    logging.info("Saving results to CSV...")
    results_df.to_csv('optimization_results_softmax.csv', index=False)

    logging.info("Optimization completed successfully!")

if __name__ == "__main__":
    main()
