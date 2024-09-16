import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mvt_brr import MVTModel
from world import Patch, Agent
from utils import calculate_leave_statistics

# Set global font size for all text in plots
matplotlib.rc('font', size=15)
matplotlib.rc('axes', titlesize=15)
matplotlib.rc('axes', labelsize=15)
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('legend', fontsize=15)

colors = ['#FFC300', '#FF5733', '#C70039']  # Light orange to dark orange

class Simulation:
    def __init__(self, decay_rate, model, beta_values=None, intercept_values=None, epsilon_values=None, omega_values=None):
        self.decay_rate = decay_rate
        self.beta_values = beta_values if model == 'softmax' else None
        self.intercept_values = intercept_values if model == 'softmax' else None
        self.epsilon_values = epsilon_values if model == 'epsilon_greedy' else None
        self.omega_values = omega_values if model == 'mellowmax' else None
        self.model = model
        self.patch_types, self.rich_proportions, self.poor_proportions = self.initialize_env()

    def initialize_env(self):
        patch_types = [
            {'type': 'Low', 'initial_yield': 32.5, 'decay_rate': self.decay_rate},
            {'type': 'Mid', 'initial_yield': 45, 'decay_rate': self.decay_rate},
            {'type': 'High', 'initial_yield': 57.5, 'decay_rate': self.decay_rate}
        ]
        rich_proportions = [0.2, 0.3, 0.5]
        poor_proportions = [0.5, 0.3, 0.2]
        return patch_types, rich_proportions, poor_proportions

    def compute_analytical_stats(self, patch_id, parameter, intercept, mellowmax_type='add'):
        patch = Patch(patch_id['initial_yield'], patch_id['decay_rate'])
        expected_leave_time, std_leave_time = calculate_leave_statistics(
            policy_type=self.model,
            parameter=parameter,
            intercept=intercept,
            patch=patch,
            mellowmax_type=mellowmax_type,
            use_intercept=True
        )
        return expected_leave_time, std_leave_time  # Return variance

    def plot_softmax_metric_vs_beta_and_intercept(self, metric='mean', save_plots=False):
        if metric not in ['mean', 'std']:
            raise ValueError("Metric must be either 'mean' or 'std'.")

        fig, axes = plt.subplots(2, 1, figsize=(4, 8))

        # Plot 1: Metric vs. Beta (with intercept = -2)
        intercept_fixed = -2
        betas = np.logspace(-2, 0, 100)
        for patch_id, color in zip(self.patch_types, colors):
            metrics = []
            for beta in betas:
                mean_time, std_time = self.compute_analytical_stats(patch_id, beta, intercept_fixed)
                if metric == 'mean':
                    metrics.append(mean_time)
                elif metric == 'std':
                    metrics.append(std_time)

            axes[0].plot(betas, metrics, label=f'{patch_id["type"]} Patch', color=color, linewidth=2)

        axes[0].set_xscale('log')
        axes[0].set_xlabel(r'$\beta$ (higher = exploit)', fontsize=15)
        axes[0].set_ylabel(f'{"Expected leave time (s)" if metric == "mean" else "SD of leave time (s)"}', fontsize=15)
        axes[0].legend()
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].text(0.05, 0.9, r'$c = -2$', transform=axes[0].transAxes, fontsize=15, verticalalignment='top')

        # Plot 2: Metric vs. |Intercept| (with beta = 0.4)
        beta_fixed = 0.4
        intercepts = -np.logspace(2, -2, 100)  # Generates values from -100 to -0.01 (logarithmic scale)
        for patch_id, color in zip(self.patch_types, colors):
            metrics = []
            for intercept in intercepts:
                mean_time, std_time = self.compute_analytical_stats(patch_id, beta_fixed, intercept)
                if metric == 'mean':
                    metrics.append(mean_time)
                elif metric == 'std':
                    metrics.append(std_time)

            axes[1].plot(intercepts, metrics, label=f'{patch_id["type"]} Patch', color=color, linewidth=2)

        axes[1].set_xscale('symlog', linthresh=1e-2)
        axes[1].set_xticks([-100, -10, -1, -0.01])
        axes[1].set_xticklabels([r'$-100$', r'$-10$', r'$-1$', r'$-0.01$'])
        axes[1].set_xlabel(r'$c$ (bias, higher = exploit)', fontsize=15)
        axes[1].set_ylabel(f'{"Expected leave time (s)" if metric == "mean" else "SD of leave time (s)"}', fontsize=15)
        axes[1].legend()
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].text(0.05, 0.9, r'$\beta = 0.4$', transform=axes[1].transAxes, fontsize=15, verticalalignment='top')

        plt.tight_layout()
        if save_plots:
            plt.savefig(f'../plots/softmax_{metric}_vs_beta_and_intercept.png')
        else:
            plt.show()
    
    def plot_mellowmax_leave_time_vs_omega(self, metric='mean', save_plots=False):
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))

        # Plot 1: Expected leave time vs. Beta (with intercept = -2)
        intercept_fixed = 0
        omegas = np.logspace(-2, 0, 100)
        for patch_id, color in zip(self.patch_types, colors):
            # leave_times = []
            metrics = []
            for omega in omegas:
                mean_time, std_time = self.compute_analytical_stats(patch_id, omega, intercept_fixed)
                # leave_times.append(mean_time)
                if metric == 'mean':
                    metrics.append(mean_time)
                elif metric == 'std':
                    metrics.append(std_time)
            # axes.plot(omegas, leave_times, label=f'{patch_id["type"]} Patch', color=color, linewidth=2)
            axes.plot(omegas, metrics, label=f'{patch_id["type"]}', color=color, linewidth=2)

        axes.set_xscale('log')
        axes.set_xlabel(r'$\omega$ (higher = exploit)', fontsize=15)
        # axes.set_ylabel('Expected leaving time (s)', fontsize=15)
        axes.set_ylabel(f'{"Expected leave time (s)" if metric == "mean" else "SD of leave time (s)"}', fontsize=15)
        axes.legend()
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.text(0.04, 9, fr'$c = {intercept_fixed}$', fontsize=15)

        # # Plot 2: Expected leave time vs. |Intercept| (with beta = 0.4)
        # omega_fixed = 0.5
        # intercepts = np.concatenate([
        #     -np.logspace(2, np.log10(0.01), 80),  # Generates values from -100 to -0.01 (logarithmic scale)
        #     np.linspace(-0.01, 10, 20)  # Extends the range from -0.01 to 10 linearly
        # ])
        # for patch_id, color in zip(self.patch_types, colors):
        #     leave_times = []
        #     for intercept in intercepts:
        #         mean_time, _ = self.compute_analytical_stats(patch_id, omega_fixed, intercept)
        #         leave_times.append(mean_time)
        #     axes.plot(intercepts, leave_times, label=f'{patch_id["type"]}', color=color, linewidth=2)

        # axes.set_xscale('symlog', linthresh=1e-2)
        # axes.set_xticks([-100, -1, 10])  # Set only three x-tick values
        # axes.set_xticklabels([r'$-100$', r'$-1$', r'$10$'])  # Corresponding labels
        # axes.set_xlabel(r'$c$ (bias, higher = exploit)', fontsize=15)
        # axes.set_ylabel('Expected leaving time (s)', fontsize=15)
        # axes.legend()
        # axes.spines['top'].set_visible(False)
        # axes.spines['right'].set_visible(False)
        # # axes.text(0.04, 4, r'$\omega = 0.2$', fontsize=15)

        plt.tight_layout()
        if save_plots:
            plt.savefig(f'../plots/mellowmax_{metric}_leave_time_vs_omega.png')
        else:
            plt.show()

def main():
    decay_rate = 0.075
    model = 'softmax'  # model choices are ['softmax', 'epsilon_greedy', 'mellowmax']
    
    # Parameters for the different models
    epsilon_values = [0.1, 0.2, 0.3]  # Only for epsilon-greedy
    omega_values = [0.25, 0.5, 0.75]  # Only for mellowmax
    beta_values = [0.25, 0.5, 0.75]  # Only for softmax
    intercept_values = [-1, 0, 1]  # Only for softmax

    mvt_model = MVTModel(decay_type='exponential')
    MVT_rich, MVT_poor = mvt_model.run()

    sim = Simulation(
        decay_rate, model, 
        beta_values=beta_values, 
        intercept_values=intercept_values, 
        epsilon_values=epsilon_values, 
        omega_values=omega_values
    )
    
    # sim.plot_softmax_leave_time_vs_beta_and_intercept(save_plots=True)
    sim.plot_softmax_metric_vs_beta_and_intercept(metric='std', save_plots=True)

    # sim.plot_mellowmax_leave_time_vs_omega(metric='std', save_plots=True)

if __name__ == "__main__":
    main()