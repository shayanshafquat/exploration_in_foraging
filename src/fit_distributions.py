import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class DistributionFitter:
    def __init__(self, data):
        """
        Initialize the fitter with data and available distributions.
        
        Parameters:
        data (array-like): The data to fit distributions to
        """
        self.data = np.array(data)
        self.distributions = {
            'normal': self._fit_normal,
            'lognormal': self._fit_lognormal,
            'gamma': self._fit_gamma,
            'exponential': self._fit_exponential,
            'weibull': self._fit_weibull,
            'uniform': self._fit_uniform,
            'bimodal_normal': self._fit_bimodal_normal,
            # 'bimodal_gamma': self._fit_bimodal_gamma
        }

    def _fit_normal(self):
        params = stats.norm.fit(self.data)
        logL = np.sum(stats.norm.logpdf(self.data, *params))
        return {'params': params, 'logL': logL, 'n_params': 2}

    def _fit_lognormal(self):
        params = stats.lognorm.fit(self.data)
        logL = np.sum(stats.lognorm.logpdf(self.data, *params))
        return {'params': params, 'logL': logL, 'n_params': 2}

    def _fit_gamma(self):
        params = stats.gamma.fit(self.data)
        logL = np.sum(stats.gamma.logpdf(self.data, *params))
        return {'params': params, 'logL': logL, 'n_params': 2}

    def _fit_exponential(self):
        params = stats.expon.fit(self.data)
        logL = np.sum(stats.expon.logpdf(self.data, *params))
        return {'params': params, 'logL': logL, 'n_params': 1}

    def _fit_weibull(self):
        params = stats.weibull_min.fit(self.data)
        logL = np.sum(stats.weibull_min.logpdf(self.data, *params))
        return {'params': params, 'logL': logL, 'n_params': 2}

    def _fit_uniform(self):
        params = stats.uniform.fit(self.data)
        logL = np.sum(stats.uniform.logpdf(self.data, *params))
        return {'params': params, 'logL': logL, 'n_params': 2}

    def _bimodal_normal_pdf(self, x, mu1, sigma1, mu2, sigma2, p):
        return p * stats.norm.pdf(x, mu1, sigma1) + (1-p) * stats.norm.pdf(x, mu2, sigma2)

    def _fit_bimodal_normal(self):
        def neg_log_likelihood(params):
            mu1, sigma1, mu2, sigma2, p = params
            if sigma1 <= 0 or sigma2 <= 0 or p < 0 or p > 1:
                return np.inf
            pdf_values = self._bimodal_normal_pdf(self.data, mu1, sigma1, mu2, sigma2, p)
            return -np.sum(np.log(pdf_values + 1e-10))

        # Initial guess based on data statistics
        initial_guess = [
            np.percentile(self.data, 25),  # mu1
            np.std(self.data)/2,           # sigma1
            np.percentile(self.data, 75),  # mu2
            np.std(self.data)/2,           # sigma2
            0.5                            # p
        ]
        
        bounds = [
            (min(self.data), max(self.data)),  # mu1
            (1e-4, None),                      # sigma1
            (min(self.data), max(self.data)),  # mu2
            (1e-4, None),                      # sigma2
            (0, 1)                             # p
        ]

        result = minimize(neg_log_likelihood, initial_guess, bounds=bounds)
        return {'params': result.x, 'logL': -result.fun, 'n_params': 5}

    # def _bimodal_gamma_pdf(self, x, a1, b1, a2, b2, p):
    #     return p * stats.gamma.pdf(x, a1, scale=b1) + (1-p) * stats.gamma.pdf(x, a2, scale=b2)

    # def _fit_bimodal_gamma(self):
    #     def neg_log_likelihood(params):
    #         a1, b1, a2, b2, p = params
    #         if a1 <= 0 or b1 <= 0 or a2 <= 0 or b2 <= 0 or p < 0 or p > 1:
    #             return np.inf
    #         pdf_values = self._bimodal_gamma_pdf(self.data, a1, b1, a2, b2, p)
    #         return -np.sum(np.log(pdf_values + 1e-10))

    #     # Initial guess
    #     initial_guess = [2, 1, 2, 1, 0.5]
    #     bounds = [(1e-4, None), (1e-4, None), (1e-4, None), (1e-4, None), (0, 1)]
        
    #     result = minimize(neg_log_likelihood, initial_guess, bounds=bounds)
    #     return {'params': result.x, 'logL': -result.fun, 'n_params': 5}

    def fit_all(self):
        """Fit all distributions and compute AIC and BIC scores."""
        n = len(self.data)
        results = {}
        
        for dist_name, fit_func in self.distributions.items():
            try:
                fit_result = fit_func()
                aic = 2 * fit_result['n_params'] - 2 * fit_result['logL']
                bic = np.log(n) * fit_result['n_params'] - 2 * fit_result['logL']
                
                results[dist_name] = {
                    'params': fit_result['params'],
                    'logL': fit_result['logL'],
                    'aic': aic,
                    'bic': bic
                }
            except:
                print(f"Failed to fit {dist_name} distribution")
                continue
                
        return results

def plot_scores(total_scores, score_type, patch_names):
    """
    Plot total scores (AIC or BIC) vs patch types for each distribution.
    
    Parameters:
    total_scores (DataFrame): DataFrame containing the scores
    score_type (str): Either 'aic' or 'bic'
    patch_names (dict): Dictionary mapping patch numbers to names
    """
    
    plt.figure(figsize=(8, 6))
    
    # Create plot
    for dist in total_scores['distribution'].unique():
        dist_data = total_scores[total_scores['distribution'] == dist]
        plt.plot(dist_data['patch_type'], dist_data[score_type], 
                marker='o', label=dist, linewidth=2)
    
    plt.xlabel('Patch Type', fontsize=12)
    plt.ylabel(f'Total {score_type.upper()}', fontsize=12)
    plt.title(f'Total {score_type.upper()} by Patch Type and Distribution', fontsize=14)
    plt.legend(title='Distribution', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Remove top and right spines
    sns.despine()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures/distribution_fits', exist_ok=True)
    
    # Save plot
    plt.savefig(f'figures/distribution_fits/total_{score_type.lower()}_comparison.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Load the data
    print(f"Current working directory: {os.getcwd()}")
    df = pd.read_csv("leheron_trialbytrial/leheron_trialbytrial.csv")
    
    # Dictionary to map patch numbers to names
    patch_names = {1: 'Low', 2: 'Medium', 3: 'High'}
    env_names = {1: 'Rich', 2: 'Poor'}
    
    # Create a DataFrame to store all fitting results
    all_results = []
    
    # Fit distributions for each subject and patch type
    for subject in df['sub'].unique():
        for patch in df['patch'].unique():
            for env in df['env'].unique():
                # Get leaving times for this subject, patch type and environment
                leave_times = df[
                    (df['sub'] == subject) & 
                    (df['patch'] == patch) & 
                    (df['env'] == env)
                ]['leaveT'].values
                
                if len(leave_times) == 0:
                    continue
                
                # Fit distributions
                fitter = DistributionFitter(leave_times)
                fit_results = fitter.fit_all()
                
                # Store results for each distribution
                for dist_name, results in fit_results.items():
                    all_results.append({
                        'subject': subject,
                        'patch_type': patch_names[patch],
                        'environment': env_names[env],
                        'distribution': dist_name,
                        'aic': results['aic'],
                        'bic': results['bic'],
                        'logL': results['logL'],
                        'params': str(results['params'])
                    })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate total AIC and BIC for each distribution and patch type
    total_scores = (results_df
        .groupby(['distribution', 'patch_type'])
        .agg({
            'aic': 'sum',
            'bic': 'sum'
        })
        .reset_index()
    )
    
    # Find best distribution for each patch type
    best_distributions = pd.DataFrame()
    for patch in patch_names.values():
        patch_scores = total_scores[total_scores['patch_type'] == patch]
        
        best_aic_dist = patch_scores.loc[patch_scores['aic'].idxmin()]
        best_bic_dist = patch_scores.loc[patch_scores['bic'].idxmin()]
        
        best_distributions = pd.concat([
            best_distributions,
            pd.DataFrame({
                'patch_type': [patch],
                'best_dist_aic': [best_aic_dist['distribution']],
                'aic_score': [best_aic_dist['aic']],
                'best_dist_bic': [best_bic_dist['distribution']],
                'bic_score': [best_bic_dist['bic']]
            })
        ])
    
    os.makedirs('results', exist_ok=True)

    # Save detailed results
    results_df.to_csv('results/all_distribution_fits.csv', index=False)
    best_distributions.to_csv('results/best_distributions.csv', index=False)
    
    # Print summary
    print("\nBest Distributions by Patch Type:")
    print(best_distributions)
    
    # Print detailed scores for each distribution
    print("\nTotal AIC scores by distribution and patch type:")
    aic_summary = total_scores.pivot(
        index='patch_type',
        columns='distribution',
        values='aic'
    )
    print(aic_summary)
    
    print("\nTotal BIC scores by distribution and patch type:")
    bic_summary = total_scores.pivot(
        index='patch_type',
        columns='distribution',
        values='bic'
    )
    print(bic_summary)
    
    # Create plots for AIC and BIC
    plot_scores(total_scores, 'aic', patch_names)
    plot_scores(total_scores, 'bic', patch_names)

if __name__ == "__main__":
    main() 