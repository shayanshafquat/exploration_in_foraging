import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mvt_brr import MVTModel
from world import Patch, Agent

matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

class Simulation:
    def __init__(self, decay_rate, model, beta_values=None, intercept_values=None, epsilon_values=None, omega_values=None):
        self.decay_rate = decay_rate
        self.beta_values = beta_values if beta_values is not None else []
        self.intercept_values = intercept_values if intercept_values is not None else []
        self.model = model
        self.epsilon_values = epsilon_values if model == 'epsilon-greedy' else None
        self.omega_values = omega_values if model == 'mellowmax' else None
        self.patch_types = self.initialize_env()

    def initialize_env(self):
        patch_types = [
            {'type': 'Low', 'initial_yield': 32.5, 'decay_rate': self.decay_rate},
            {'type': 'Mid', 'initial_yield': 45, 'decay_rate': self.decay_rate},
            {'type': 'High', 'initial_yield': 57.5, 'decay_rate': self.decay_rate}
        ]
        return patch_types

    def simulate(self, patch_id, agent, n_runs=1000, n_max=1000):
        leave_times = []
        for _ in range(n_runs):
            patch = Patch(patch_id['initial_yield'], patch_id['decay_rate'])
            patch.start_harvesting()
            for t in range(1, n_max+1):
                reward = patch.get_reward()
                if self.model == 'epsilon-greedy':
                    action = agent.choose_action_epsilon(reward)
                elif self.model == 'mellowmax':
                    action = agent.choose_action_mellowmax(reward)
                else:
                    action = agent.choose_action_softmax(reward)
                if action == 1:
                    leave_times.append(t)
                    break
        return leave_times

    def compute_stats(self, leave_times):
        leave_times = np.array(leave_times)
        mean_leave_time = np.mean(leave_times)
        sem_leave_time = np.std(leave_times) / np.sqrt(len(leave_times))  # Calculate SEM
        return mean_leave_time, sem_leave_time

    def prepare_results(self):
        results = []

        if self.model == 'mellowmax':
            for omega in self.omega_values:
                for intercept in self.intercept_values:
                    agent = Agent(policy_type=self.model, omega=omega, intercept=intercept, mellowmax_type='add')

                    patch_stats = []
                    for patch_id in self.patch_types:
                        leave_times = self.simulate(patch_id, agent)
                        stats = self.compute_stats(leave_times)
                        patch_stats.append(stats)

                    results.append({
                        'omega': omega,
                        'intercept': intercept,
                        'patch_stats': patch_stats
                    })
        elif self.model == 'epsilon-greedy':
            for epsilon in self.epsilon_values:  # Example range from 0.01 to 1.0
                agent = Agent(policy_type=self.model, epsilon=epsilon)

                patch_stats = []
                for patch_id in self.patch_types:
                    leave_times = self.simulate(patch_id, agent)
                    stats = self.compute_stats(leave_times)
                    patch_stats.append(stats)

                results.append({
                    'epsilon': epsilon,
                    'patch_stats': patch_stats
                })
        else:
            for beta in self.beta_values:
                for intercept in self.intercept_values:
                    agent = Agent(beta=beta, intercept=intercept)

                    patch_stats = []
                    for patch_id in self.patch_types:
                        leave_times = self.simulate(patch_id, agent)
                        stats = self.compute_stats(leave_times)
                        patch_stats.append(stats)

                    results.append({
                        'beta': beta,
                        'intercept': intercept,
                        'patch_stats': patch_stats
                    })
        return results

    def plot_results(self, results, MVT_rich, MVT_poor, save_plots=False):
        if self.model == 'mellowmax':
            self.plot_mellowmax_results(results, MVT_rich, MVT_poor, save_plots)
        elif self.model == 'epsilon-greedy':
            self.plot_epsilon_greedy_results(results, MVT_rich, MVT_poor, save_plots)
        else:
            self.plot_softmax_results(results, MVT_rich, MVT_poor, save_plots)

    def plot_epsilon_greedy_results(self, results, MVT_rich, MVT_poor, save_plots=False):
        fig, ax = plt.subplots(figsize=(4, 4))

        epsilon_colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

        for i, result in enumerate(results):
            epsilon = result["epsilon"]
            y_value = 1 / epsilon  # Calculating the y-value as 1/epsilon

            patch_types = ['Low', 'Mid', 'High']
            y_values = [y_value, y_value, y_value]
            
            # Plot horizontal line at y = 1/epsilon
            color = 'green' if i == 0 else 'blue'
            ax.plot(patch_types, y_values, linestyle='-', marker='o', color=color, label=f'Epsilon: {epsilon:.2f}')

        # for i, result in enumerate(results):
        #     patch_stats = result['patch_stats']

        #     patch_types = ['Low', 'Mid', 'High']
        #     means = [stat[0] for stat in patch_stats]
        #     errors = [stat[1] for stat in patch_stats]

        #     color = epsilon_colors[i]
        #     ax.errorbar(patch_types, means, yerr=errors, marker='o', linestyle='-', color=color, 
        #                 label=f'Epsilon: {result["epsilon"]:.2f}', capsize=5)

        ax.plot(patch_types, MVT_rich, marker='o', linestyle='--', color='blue', alpha=0.8)
        ax.plot(patch_types, MVT_poor, marker='o', linestyle='--', color='green', alpha=0.8)

        ax.set_xlabel('Patch Types', fontsize=15)
        ax.set_ylabel('Expected Leave Time', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.legend(fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        if save_plots:
            plt.savefig('../plots/epsilon_greedy_results.png')
        else:
            plt.show()

    def plot_mellowmax_results(self, results, MVT_rich, MVT_poor, save_plots=False):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # Colormaps for varying omega and intercept
        omega_colors = plt.cm.viridis(np.linspace(0, 1, len(self.omega_values)))
        intercept_colors = plt.cm.viridis(np.linspace(0, 1, len(self.intercept_values)))

        # Plot with constant intercept (choose a middle value)
        constant_intercept = self.intercept_values[0]
        ax = axes[0]

        for i, omega in enumerate(self.omega_values):
            result = next(r for r in results if r['omega'] == omega and r['intercept'] == constant_intercept)
            patch_stats = result['patch_stats']

            patch_types = ['Low', 'Mid', 'High']
            means = [stat[0] for stat in patch_stats]
            errors = [stat[1] for stat in patch_stats]

            color = omega_colors[i]
            ax.errorbar(patch_types, means, yerr=errors, marker='o', linestyle='-', color=color, label=f'$\\omega$ = {omega}', capsize=5)

        ax.plot(patch_types, MVT_rich, marker='o', linestyle='-', color='grey')
        ax.plot(patch_types, MVT_poor, marker='o', linestyle='--', color='grey')

        ax.set_xlabel('Patch Types', fontsize=15)
        ax.set_ylabel('Mean Leave Time', fontsize=15)
        ax.legend(fontsize=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.5, 0.1, f'c = {constant_intercept}', fontsize=15, ha='center', transform=ax.transAxes)

        # Plot with constant omega (choose a middle value)
        constant_omega = self.omega_values[0]
        ax = axes[1]

        for i, intercept in enumerate(self.intercept_values):
            result = next(r for r in results if r['omega'] == constant_omega and r['intercept'] == intercept)
            patch_stats = result['patch_stats']

            patch_types = ['Low', 'Mid', 'High']
            means = [stat[0] for stat in patch_stats]
            errors = [stat[1] for stat in patch_stats]

            color = intercept_colors[i]
            ax.errorbar(patch_types, means, yerr=errors, marker='o', linestyle='-', color=color, label=f'c = {intercept}', capsize=5)

        ax.plot(patch_types, MVT_rich, marker='o', linestyle='-', color='grey')
        ax.plot(patch_types, MVT_poor, marker='o', linestyle='--', color='grey')

        ax.set_xlabel('Patch Types', fontsize=15)
        ax.set_ylabel('Mean Leave Time', fontsize=15)
        ax.legend(fontsize=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.5, 0.9, f'$\\omega$ = {constant_omega}', fontsize=15, ha='center', transform=ax.transAxes)

        plt.tight_layout()
        if save_plots:
            plt.savefig('../plots/mellowmax_results_simulation.png')
        else:
            plt.show()

    def plot_softmax_results(self, results, MVT_rich, MVT_poor, save_plots=False):
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))

        custom_cmap = LinearSegmentedColormap.from_list("blue_green", ["blue", "green"])
        # Gradient colors based on increasing beta and intercept values
        beta_gradient = custom_cmap(np.linspace(0, 1, len(self.beta_values)))
        intercept_gradient = custom_cmap(np.linspace(0, 1, len(self.intercept_values)))

        # Plot with varying beta, constant intercept
        constant_intercept = self.intercept_values[0]
        ax = axes[0]

        for i, beta in enumerate(self.beta_values):
            result = next(r for r in results if r['beta'] == beta and r['intercept'] == constant_intercept)
            patch_stats = result['patch_stats']

            patch_types = ['Low', 'Mid', 'High']
            means = [stat[0] for stat in patch_stats]
            errors = [stat[1] for stat in patch_stats]

            color = beta_gradient[i]
            ax.errorbar(patch_types, means, yerr=errors, marker='o', linestyle='-', color=color, label=f'Beta: {beta}', capsize=5)

        ax.plot(patch_types, MVT_rich, marker='o', linestyle='-', color='black', label='Optimal Rich')
        ax.plot(patch_types, MVT_poor, marker='o', linestyle='--', color='black', label='Optimal Poor')

        ax.set_xlabel('Patch Types', fontsize=15)
        ax.set_ylabel('Mean Leave Time', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.legend(fontsize=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Plot with varying intercept, constant beta
        constant_beta = self.beta_values[0]  # You requested to use beta=0.4
        ax = axes[1]

        for i, intercept in enumerate(self.intercept_values):
            result = next(r for r in results if r['beta'] == constant_beta and r['intercept'] == intercept)
            patch_stats = result['patch_stats']

            patch_types = ['Low', 'Mid', 'High']
            means = [stat[0] for stat in patch_stats]
            errors = [stat[1] for stat in patch_stats]

            color = intercept_gradient[i]
            ax.errorbar(patch_types, means, yerr=errors, marker='o', linestyle='-', color=color, label=f'Intercept: {intercept}', capsize=5)

        ax.plot(patch_types, MVT_rich, marker='o', linestyle='-', color='black', label='MVT Rich')
        ax.plot(patch_types, MVT_poor, marker='o', linestyle='--', color='black', label='MVT Poor')

        ax.set_xlabel('Patch Types', fontsize=15)
        ax.set_ylabel('Mean Leave Time', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.legend(fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        if save_plots:
            plt.savefig('../plots/softmax_epsilon_results.png')
        else:
            plt.show()

def main():
    decay_rate = 0.075
    model = 'epsilon-greedy'
    beta_values = [0.185, 0.22]  # Example beta values for demonstration
    # intercept_values = [-3.07, -2.38]  # Example intercept values for demonstration
    intercept_values = [-1, 0]
    omega_values = [0.2, 0.4]
    epsilon_values = [0.08, 0.1]

    # Assume MVTModel has the required implementation
    mvt_model = MVTModel(decay_type='exponential')
    MVT_rich, MVT_poor = mvt_model.run()

    sim = Simulation(decay_rate, model, beta_values=beta_values, intercept_values=intercept_values, epsilon_values=epsilon_values, omega_values=omega_values)
    results = sim.prepare_results()
    sim.plot_results(results, MVT_rich, MVT_poor, save_plots=True)

if __name__ == "__main__":
    main()