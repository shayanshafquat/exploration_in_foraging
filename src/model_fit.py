import sys
import scipy.io
import numpy as np
import pandas as pd
import scipy.optimize as opt
import logging

# Import custom modules
sys.path.append('../src')
from mvt_brr import MVTModel
from world import Patch, Agent
from simulation import Simulation

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    logging.info("Loading data...")
    data = scipy.io.loadmat('../leheron_trialbytrial/leheron_blockSwitchIndex.mat')
    block_order_df = pd.read_csv('../leheron_trialbytrial/leheron_blockOrder.csv')
    df_trials = pd.read_csv("../leheron_trialbytrial/leheron_trialbytrial.csv")
    logging.info("Data loaded successfully.")
    return df_trials

def error_calculation(patch_type_leave_times, patch_type_observed_leave):
    mean_simulated_leave = {patch_type: np.mean(times) for patch_type, times in patch_type_leave_times.items()}
    mean_observed_leave = {patch_type: np.mean(times) for patch_type, times in patch_type_observed_leave.items()}
    squared_errors = [(mean_simulated_leave[patch_type] - mean_observed_leave[patch_type]) ** 2 for patch_type in patch_type_leave_times]
    rmse_error = np.sqrt(np.sum(squared_errors))
    return rmse_error

def simulation(params, patch_types, observed_leave, constant_intercept=None):
    beta, intercept = params if constant_intercept is None else (params, constant_intercept)
    agent = Agent(beta=beta, intercept=intercept)
    sim = Simulation(decay_rate=0.075, model='softmax')

    patch_type_leave_times = {patch_type: [] for patch_type in set(patch_types)}
    patch_type_observed_leave = {patch_type: [] for patch_type in set(patch_types)}

    for i, patch_type in enumerate(patch_types):
        patch_id = sim.patch_types[patch_type-1]
        leave_times = sim.simulate(patch_id, agent, n_runs=5)
        patch_type_leave_times[patch_type].extend([np.mean(leave_times)])
        patch_type_observed_leave[patch_type].append(observed_leave[i])

    rmse = error_calculation(patch_type_leave_times, patch_type_observed_leave)
    return rmse

def optimize_params(df_trials, constant_intercept=-3):
    logging.info("Starting optimization process...")
    environments = df_trials['env'].unique()
    results = []

    for env in environments:
        logging.info(f"Processing environment: {env}")
        env_data = df_trials[df_trials['env'] == env]
        participants = env_data['sub'].unique()

        for participant in participants:
            logging.info(f"Processing participant: {participant}")
            participant_data = env_data[env_data['sub'] == participant]
            patches = participant_data['patch'].values
            leave_times = participant_data['leaveT'].values

            initial_beta = 0.3
            initial_params = [0.3, -3]

            res_beta_vary = opt.minimize(simulation, initial_beta, method='nelder-mead', args=(patches, leave_times, constant_intercept), options={'xatol': 1e-6, 'disp': True})
            res_both_vary = opt.minimize(simulation, initial_params, method='nelder-mead', args=(patches, leave_times), options={'xatol': 1e-6, 'disp': True})

            results.append({
                'environment': env,
                'participant': participant,
                'beta_vary': res_beta_vary.x[0],
                'intercept_constant': constant_intercept,
                'rmse_beta': res_beta_vary.fun,
                'beta_both_vary': res_both_vary.x[0],
                'intercept_both_vary': res_both_vary.x[1],
                'rmse_both': res_both_vary.fun
            })

    results_df = pd.DataFrame(results)
    logging.info("Optimization process completed.")
    return results_df

def main():
    df_trials = load_data()
    results_df = optimize_params(df_trials)
    results_df.to_csv('optimization_results.csv', index=False)
    logging.info("Results saved to optimization_results.csv")

if __name__ == "__main__":
    main()