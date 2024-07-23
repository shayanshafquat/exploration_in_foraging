import numpy as np
import matplotlib.pyplot as plt
from mvt_brr import MVTModel
from world import Patch, Agent

class Simulation:
    def __init__(self, decay_rate, beta_values, intercept_values, model, epsilon=None):
        self.decay_rate = decay_rate
        self.beta_values = beta_values
        self.intercept_values = intercept_values
        self.model = model
        self.epsilon = epsilon if model == 'epsilon_greedy' else None
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

    def simulate(self, environment, agent, n_runs=1000, n_max=1000):
        leave_times = []
        for _ in range(n_runs):
            patch = Patch(environment['initial_yield'], environment['decay_rate'])
            patch.start_harvesting()
            for t in range(1, n_max+1):
                reward = patch.get_reward()
                if self.model == 'epsilon_greedy':
                    action = agent.choose_action_epsilon(reward, self.epsilon)
                else:
                    action = agent.choose_action(reward)
                if action == 1:
                    leave_times.append(t)
                    break
        return leave_times

    def compute_stats(self, leave_times):
        leave_times = np.array(leave_times)
        mean_leave_time = np.mean(leave_times)
        var_leave_time = np.var(leave_times)
        return mean_leave_time, var_leave_time

    def prepare_results(self):
        results = []

        for beta in self.beta_values:
            for intercept in self.intercept_values:
                agent = Agent(beta=beta, intercept=intercept)

                rich_stats = []
                poor_stats = []

                for env, proportion in zip(self.patch_types, self.rich_proportions):
                    leave_times = self.simulate(env, agent, n_runs=int(1000 * proportion))
                    stats = self.compute_stats(leave_times)
                    rich_stats.append(stats)

                for env, proportion in zip(self.patch_types, self.poor_proportions):
                    leave_times = self.simulate(env, agent, n_runs=int(1000 * proportion))
                    stats = self.compute_stats(leave_times)
                    poor_stats.append(stats)

                results.append({
                    'beta': beta,
                    'intercept': intercept,
                    'rich_stats': rich_stats,
                    'poor_stats': poor_stats
                })
        return results

    def plot_results(self, results, MVT_rich, MVT_poor):
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
            rich_errors = [np.sqrt(stat[1]) for stat in rich_stats]  # Standard deviation
            poor_errors = [np.sqrt(stat[1]) for stat in poor_stats]  # Standard deviation

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
        ax.plot(patch_types, MVT_poor, marker='o', linestyle='--', color='black', label='Optimal Poor')

        ax.set_xlabel('Patch Types')
        ax.set_ylabel('Mean Leave Time')
        ax.set_title(f'Constant Beta: {constant_beta}, Varying Intercept')
        ax.legend()

        plt.tight_layout()
        plt.show()

def main():
    decay_rate = 0.075
    beta_values = [0.25, 0.5, 0.75]
    intercept_values = [-1, 0, 1]
    epsilon = 0.05
    # model = 'epsilon_greedy'
    model = 'other'

    mvt_model = MVTModel(decay_type='exponential')
    MVT_rich, MVT_poor = mvt_model.run()

    sim = Simulation(decay_rate, beta_values, intercept_values, model, epsilon)
    results = sim.prepare_results()
    sim.plot_results(results, MVT_rich, MVT_poor)

if __name__ == "__main__":
    main()