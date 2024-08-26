import numpy as np
import matplotlib.pyplot as plt
from mvt_brr import MVTModel
from world import Patch, Agent
from utils import calculate_leave_statistics

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

    def compute_analytical_stats(self, patch_id, parameter, intercept, mellowmax_type=None):
        patch = Patch(patch_id['initial_yield'], patch_id['decay_rate'])
        expected_leave_time, std_leave_time = calculate_leave_statistics(
            policy_type=self.model,
            parameter=parameter,
            intercept=intercept,
            patch=patch,
            mellowmax_type=mellowmax_type
        )
        return expected_leave_time, std_leave_time ** 2  # Return variance

    def prepare_results(self):
        results = []

        if self.model == 'mellowmax':
            for omega in self.omega_values:
                rich_stats = []
                poor_stats = []

                for patch_id in self.patch_types:
                    expected_time, var_time = self.compute_analytical_stats(patch_id, omega, intercept=0, mellowmax_type='type1')
                    rich_stats.append((expected_time, var_time))

                for patch_id in self.patch_types:
                    expected_time, var_time = self.compute_analytical_stats(patch_id, omega, intercept=0, mellowmax_type='type2')
                    poor_stats.append((expected_time, var_time))

                results.append({
                    'omega': omega,
                    'rich_stats': rich_stats,
                    'poor_stats': poor_stats
                })

        elif self.model == 'softmax':
            for beta in self.beta_values:
                for intercept in self.intercept_values:
                    rich_stats = []
                    poor_stats = []

                    for patch_id in self.patch_types:
                        expected_time, var_time = self.compute_analytical_stats(patch_id, beta, intercept)
                        rich_stats.append((expected_time, var_time))

                    for patch_id in self.patch_types:
                        expected_time, var_time = self.compute_analytical_stats(patch_id, beta, intercept)
                        poor_stats.append((expected_time, var_time))

                    results.append({
                        'beta': beta,
                        'intercept': intercept,
                        'rich_stats': rich_stats,
                        'poor_stats': poor_stats
                    })
        
        elif self.model == 'epsilon_greedy':
            for epsilon in self.epsilon_values:
                rich_stats = []
                poor_stats = []

                for patch_id in self.patch_types:
                    expected_time, var_time = self.compute_analytical_stats(patch_id, epsilon, intercept=0)
                    rich_stats.append((expected_time, var_time))

                for patch_id in self.patch_types:
                    expected_time, var_time = self.compute_analytical_stats(patch_id, epsilon, intercept=0)
                    poor_stats.append((expected_time, var_time))

                results.append({
                    'epsilon': epsilon,
                    'rich_stats': rich_stats,
                    'poor_stats': poor_stats
                })

        return results

    def plot_results(self, results, MVT_rich, MVT_poor, save_plots=False):
        if self.model == 'mellowmax':
            self.plot_mellowmax_results(results, MVT_rich, MVT_poor, save_plots)
        elif self.model == 'softmax':
            self.plot_softmax_results(results, MVT_rich, MVT_poor, save_plots)
        elif self.model == 'epsilon_greedy':
            self.plot_epsilon_greedy_results(results, MVT_rich, MVT_poor, save_plots)

    def plot_mellowmax_results(self, results, MVT_rich, MVT_poor, save_plots):
        fig, ax = plt.subplots(figsize=(8, 6))

        for result in results:
            omega = result['omega']
            rich_stats = result['rich_stats']
            poor_stats = result['poor_stats']

            patch_types = ['Low', 'Mid', 'High']
            rich_means = [stat[0] for stat in rich_stats]
            poor_means = [stat[0] for stat in poor_stats]
            rich_errors = [np.sqrt(stat[1]) for stat in rich_stats]  # Standard deviation
            poor_errors = [np.sqrt(stat[1]) for stat in poor_stats]  # Standard deviation

            color = plt.cm.rainbow(omega / max(self.omega_values))
            ax.errorbar(patch_types, rich_means, yerr=rich_errors, marker='o', linestyle='-', color=color, label=f'Rich, Omega: {omega}', capsize=5)
            ax.errorbar(patch_types, poor_means, yerr=poor_errors, marker='o', linestyle='--', color=color, label=f'Poor, Omega: {omega}', capsize=5)
        
        ax.plot(patch_types, MVT_rich, marker='o', linestyle='-', color='black', label='Optimal Rich')
        ax.plot(patch_types, MVT_poor, marker='o', linestyle='--', color='black', label='Optimal Poor')

        ax.set_xlabel('Patch Types')
        ax.set_ylabel('Mean Leave Time')
        ax.set_title('Mellowmax Policy')
        ax.legend()

        plt.tight_layout()
        if save_plots:
            plt.savefig('../plots/mellowmax_results.png')
        else:
            plt.show()

    def plot_softmax_results(self, results, MVT_rich, MVT_poor, save_plots):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        color_map = plt.cm.rainbow
        beta_colors = {beta: color_map(i / (len(self.beta_values) - 1)) for i, beta in enumerate(self.beta_values)}
        intercept_colors = {intercept: color_map(i / (len(self.intercept_values) - 1)) for i, intercept in enumerate(self.intercept_values)}

        # Plot with constant intercept (choose a middle value)
        constant_intercept = self.intercept_values[len(self.intercept_values)//2]
        ax = axes[0]

        for beta in self.beta_values:
            result = next(r for r in results if r['beta'] == beta and r['intercept'] == constant_intercept)
            rich_stats = result['rich_stats']
            poor_stats = result['poor_stats']

            patch_types = ['Low', 'Mid', 'High']
            rich_means = [stat[0] for stat in rich_stats]
            poor_means = [stat[0] for stat in poor_stats]
            print(beta, rich_means, poor_means)
            rich_errors = [np.sqrt(stat[1]) for stat in rich_stats]  # Standard deviation
            poor_errors = [np.sqrt(stat[1]) for stat in poor_stats]  # Standard deviation
            print(beta, rich_errors, poor_errors)

            color = beta_colors[beta]
            ax.errorbar(patch_types, rich_means, yerr=rich_errors, marker='o', linestyle='-', color=color, label=f'Rich, Beta: {beta}', capsize=5)
            ax.errorbar(patch_types, poor_means, yerr=poor_errors, marker='o', linestyle='--', color=color, label=f'Poor, Beta: {beta}', capsize=5)
        ax.plot(patch_types, MVT_rich, marker='o', linestyle='-', color='black', label='Optimal Rich')
        ax.plot(patch_types, MVT_poor, marker='o', linestyle='--', color='black', label='Optimal Poor')

        ax.set_xlabel('Patch Types')
        ax.set_ylabel('Mean Leave Time')
        ax.set_title(f'Constant Intercept: {constant_intercept}, Varying Beta')
        ax.legend()

        # Plot with constant beta (choose a middle value)
        constant_beta = self.beta_values[len(self.beta_values)//2]
        ax = axes[1]

        for intercept in self.intercept_values:
            result = next(r for r in results if r['beta'] == constant_beta and r['intercept'] == intercept)
            rich_stats = result['rich_stats']
            poor_stats = result['poor_stats']

            patch_types = ['Low', 'Mid', 'High']
            rich_means = [stat[0] for stat in rich_stats]
            poor_means = [stat[0] for stat in poor_stats]
            rich_errors = [np.sqrt(stat[1]) for stat in rich_stats]  # Standard deviation
            poor_errors = [np.sqrt(stat[1]) for stat in poor_stats]  # Standard deviation

            color = intercept_colors[intercept]
            ax.errorbar(patch_types, rich_means, yerr=rich_errors, marker='o', linestyle='-', color=color, label=f'Rich, Intercept: {intercept}', capsize=5)
            ax.errorbar(patch_types, poor_means, yerr=poor_errors, marker='o', linestyle='--', color=color, label=f'Poor, Intercept: {intercept}', capsize=5)
        ax.plot(patch_types, MVT_rich, marker='o', linestyle='-', color='black', label='Optimal Rich')
        ax.plot(patch_types, MVT_poor, marker='o', linestyle='-', color='black', label='Optimal Poor')

        ax.set_xlabel('Patch Types')
        ax.set_ylabel('Mean Leave Time')
        ax.set_title(f'Constant Beta: {constant_beta}, Varying Intercept')
        ax.legend()

        plt.tight_layout()
        if save_plots:
            plt.savefig('../plots/softmax_results.png')
        else:
            plt.show()
    
    def plot_softmax_leave_time_vs_beta_and_intercept(self, save_plots=False):
        fig, axes = plt.subplots(2, 1, figsize=(6, 10))

        # Plot 1: Expected leave time vs. Beta (with intercept = -2)
        intercept_fixed = -2
        betas = np.logspace(-2, 0, 100)
        for patch_id in self.patch_types:
            leave_times = []
            for beta in betas:
                mean_time, _ = self.compute_analytical_stats(patch_id, beta, intercept_fixed)
                leave_times.append(mean_time)
            axes[0].plot(betas, leave_times, label=f'{patch_id["type"]} Patch', linewidth=2)

        axes[0].set_xscale('log')
        axes[0].set_xlabel(r'Beta (higher = exploit)')
        axes[0].set_ylabel('Expected leaving time (s)')
        axes[0].set_title(r'c = -2')
        axes[0].legend()

        # Plot 2: Expected leave time vs. |Intercept| (with beta = 0.3)
        beta_fixed = 0.1
        intercepts = -np.logspace(2, -2, 100)  # Generates values from -0.01 to -1 (logarithmic scale)
        for patch_id in self.patch_types:
            leave_times = []
            for intercept in intercepts:
                mean_time, _ = self.compute_analytical_stats(patch_id, beta_fixed, intercept)
                print(intercept, mean_time)
                leave_times.append(mean_time)
            axes[1].plot(intercepts, leave_times, label=f'{patch_id["type"]} Patch', linewidth=2)  # Plot |c|

        # axes[1].set_xscale('log')
        axes[1].set_xscale('symlog', linthresh=1e-2)
        axes[1].set_xticks([-100, -10, -1, -0.01])
        axes[1].set_xticklabels([r'$-100$', r'$-10$', r'$-1$', r'$-0.01$'])
        axes[1].set_xlabel(r'Bias (higher = exploit)')
        axes[1].set_ylabel('Expected leaving time (s)')
        axes[1].set_title(r'$\beta$ = 0.4')
        axes[1].legend()

        plt.tight_layout()
        if save_plots:
            plt.savefig('../plots/softmax_leave_time_vs_beta_and_intercept.png')
        else:
            plt.show()

    def plot_epsilon_greedy_results(self, results, MVT_rich, MVT_poor, save_plots=False):
        fig, ax = plt.subplots(figsize=(8, 6))

        for result in results:
            epsilon = result['epsilon']
            rich_stats = result['rich_stats']
            poor_stats = result['poor_stats']

            patch_types = ['Low', 'Mid', 'High']
            rich_means = [stat[0] for stat in rich_stats]
            poor_means = [stat[0] for stat in poor_stats]
            rich_errors = [np.sqrt(stat[1]) for stat in rich_stats]  # Standard deviation
            poor_errors = [np.sqrt(stat[1]) for stat in poor_stats]  # Standard deviation

            color = plt.cm.viridis(epsilon / max(self.epsilon_values))
            ax.errorbar(patch_types, rich_means, yerr=rich_errors, marker='o', linestyle='-', color=color, label=f'Rich, Epsilon: {epsilon}', capsize=5)
            ax.errorbar(patch_types, poor_means, yerr=poor_errors, marker='o', linestyle='--', color=color, label=f'Poor, Epsilon: {epsilon}', capsize=5)
        
        # Plot the MVT (Optimal) values
        ax.plot(patch_types, MVT_rich, marker='o', linestyle='-', color='black', label='Optimal Rich')
        ax.plot(patch_types, MVT_poor, marker='o', linestyle='--', color='black', label='Optimal Poor')

        ax.set_xlabel('Patch Types')
        ax.set_ylabel('Mean Leave Time')
        ax.set_title('Epsilon-Greedy Policy')
        ax.legend()

        plt.tight_layout()
        if save_plots:
            plt.savefig('../plots/epsilon_greedy_results.png')
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
    
    # results = sim.prepare_results()
    # sim.plot_results(results, MVT_rich, MVT_poor, save_plots=True)
    sim.plot_softmax_leave_time_vs_beta_and_intercept(save_plots=True)

if __name__ == "__main__":
    main()