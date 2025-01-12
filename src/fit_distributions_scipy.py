import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib parameters
plt.style.use('seaborn')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

class DistributionAnalyzer:
    """Class to analyze and fit distributions to data using scipy's built-in methods."""
    
    def __init__(self):
        # Define distributions to test
        self.distributions = [
            stats.norm, stats.lognorm, stats.gamma, 
            stats.expon, stats.weibull_min, stats.uniform
        ]
        self.dist_names = [
            'normal', 'lognormal', 'gamma', 
            'exponential', 'weibull', 'uniform'
        ]

    def fit_distributions(self, data: np.ndarray) -> Dict:
        """
        Fit multiple distributions to the data and compute goodness-of-fit metrics.
        
        Args:
            data: Array of observations
            
        Returns:
            Dictionary containing fit results for each distribution
        """
        results = {}
        n = len(data)
        
        for dist, name in zip(self.distributions, self.dist_names):
            try:
                # Fit distribution to data
                params = dist.fit(data)
                
                # Calculate log-likelihood
                logL = np.sum(dist.logpdf(data, *params))
                
                # Calculate AIC and BIC
                k = len(params)  # number of parameters
                aic = 2 * k - 2 * logL
                bic = np.log(n) * k - 2 * logL
                
                results[name] = {
                    'params': params,
                    'logL': logL,
                    'aic': aic,
                    'bic': bic,
                    'distribution': dist
                }
            except Exception as e:
                print(f"Failed to fit {name} distribution: {str(e)}")
                continue
                
        return results

    def plot_distribution_fits(self, data: np.ndarray, results: Dict, 
                             title: str = "", save_path: str = None):
        """
        Plot histogram of data with fitted distributions.
        
        Args:
            data: Array of observations
            results: Dictionary of fit results
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of data
        plt.hist(data, bins='auto', density=True, alpha=0.6, color='gray', 
                label='Data')
        
        # Plot fitted distributions
        x = np.linspace(min(data), max(data), 200)
        for name, res in results.items():
            try:
                dist = res['distribution']
                params = res['params']
                y = dist.pdf(x, *params)
                plt.plot(x, y, label=f'{name} (AIC: {res["aic"]:.2f})')
            except:
                continue
        
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def analyze_leaving_times(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze leaving times for different patch types and environments.
    
    Args:
        df: DataFrame containing trial data
        
    Returns:
        Two DataFrames containing environment-level and patch-level results
    """
    analyzer = DistributionAnalyzer()
    env_results = []
    patch_results = []
    
    # Create output directories
    os.makedirs('figures/distribution_fits', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Environment-level analysis
    for subject in df['sub'].unique():
        # First calculate subject's overall mean and std for normalization
        subject_times = df[df['sub'] == subject]['leaveT'].values
        subject_mean = np.mean(subject_times)
        subject_std = np.std(subject_times)
        
        for patch in df['patch'].unique():
            for env in df['env'].unique():
                leave_times = df[
                    (df['sub'] == subject) & 
                    (df['patch'] == patch) & 
                    (df['env'] == env)
                ]['leaveT'].values
                
                if len(leave_times) < 5:  # Skip if too few observations
                    continue
                
                # Z-score normalize within subject
                normalized_times = (leave_times - subject_mean) / subject_std
                fit_results = analyzer.fit_distributions(normalized_times)
                
                # Plot distributions for selected subjects - Environment level
                if subject in [11]:  # Add more subjects if needed
                    analyzer.plot_distribution_fits(
                        normalized_times,
                        fit_results,
                        f'Subject {subject} - Patch {patch} - Environment {env} (Z-scored)',
                        f'figures/distribution_fits/subject{subject}_patch{patch}_env{env}_normalized_fits.png'
                    )
                    
                    # Also plot raw data for comparison
                    raw_fit_results = analyzer.fit_distributions(leave_times)
                    analyzer.plot_distribution_fits(
                        leave_times,
                        raw_fit_results,
                        f'Subject {subject} - Patch {patch} - Environment {env} (Raw)',
                        f'figures/distribution_fits/subject{subject}_patch{patch}_env{env}_raw_fits.png'
                    )
                
                # Save results
                for dist_name, res in fit_results.items():
                    env_results.append({
                        'subject': subject,
                        'patch_type': f'Patch {patch}',
                        'environment': f'Env {env}',
                        'distribution': dist_name,
                        'aic': res['aic'],
                        'bic': res['bic'],
                        'logL': res['logL']
                    })
    
    # Patch-level analysis (combining environments)
    for subject in df['sub'].unique():
        # First calculate subject's overall mean and std for normalization
        subject_times = df[df['sub'] == subject]['leaveT'].values
        subject_mean = np.mean(subject_times)
        subject_std = np.std(subject_times)
        
        for patch in df['patch'].unique():
            # Get leaving times for each environment separately
            env_times = {}
            for env in df['env'].unique():
                env_times[env] = df[
                    (df['sub'] == subject) & 
                    (df['patch'] == patch) & 
                    (df['env'] == env)
                ]['leaveT'].values
            
            # Skip if either environment has too few observations
            if any(len(times) < 5 for times in env_times.values()):
                continue
            
            # Z-score normalize all times within subject
            normalized_times = []
            raw_times = []
            for times in env_times.values():
                if len(times) > 0:
                    normalized_times.extend((times - subject_mean) / subject_std)
                    raw_times.extend(times)
            
            if len(normalized_times) < 5:
                continue
            
            normalized_times = np.array(normalized_times)
            raw_times = np.array(raw_times)
            
            fit_results = analyzer.fit_distributions(normalized_times)
            
            # Plot distributions for selected subjects - Patch level
            if subject in [11]:  # Add more subjects if needed
                analyzer.plot_distribution_fits(
                    normalized_times,
                    fit_results,
                    f'Subject {subject} - Patch {patch} (All Environments, Z-scored)',
                    f'figures/distribution_fits/subject{subject}_patch{patch}_normalized_fits.png'
                )
                
                # Also plot raw data for comparison
                raw_fit_results = analyzer.fit_distributions(raw_times)
                analyzer.plot_distribution_fits(
                    raw_times,
                    raw_fit_results,
                    f'Subject {subject} - Patch {patch} (All Environments, Raw)',
                    f'figures/distribution_fits/subject{subject}_patch{patch}_raw_fits.png'
                )
            
            # Save results
            for dist_name, res in fit_results.items():
                patch_results.append({
                    'subject': subject,
                    'patch_type': f'Patch {patch}',
                    'distribution': dist_name,
                    'aic': res['aic'],
                    'bic': res['bic'],
                    'logL': res['logL']
                })
    
    return pd.DataFrame(env_results), pd.DataFrame(patch_results)

def plot_summary_scores(results_df: pd.DataFrame, score_type: str, level: str):
    """
    Plot summary scores (AIC or BIC) for different distributions and patch types.
    
    Args:
        results_df: DataFrame containing the fit results
        score_type: 'aic' or 'bic'
        level: 'patch' or 'env' to indicate analysis level
    """
    # Calculate mean scores for each distribution and patch type
    summary = (results_df
        .groupby(['patch_type', 'distribution'])[score_type]
        .mean()
        .reset_index()
        .pivot(index='patch_type', columns='distribution', values=score_type)
    )
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    
    # Plot grouped bars
    bar_width = 0.8 / len(summary.columns)
    x = np.arange(len(summary.index))
    
    for i, (dist_name, scores) in enumerate(summary.items()):
        offset = (i - len(summary.columns)/2 + 0.5) * bar_width
        plt.bar(x + offset, scores, bar_width, label=dist_name, alpha=0.8)
    
    plt.xlabel('Patch Type')
    plt.ylabel(f'Mean {score_type.upper()}')
    plt.title(f'Mean {score_type.upper()} by Patch Type and Distribution ({level.title()} Level)')
    plt.xticks(x, summary.index)
    plt.legend(title='Distribution', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Remove top and right spines
    sns.despine()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'figures/distribution_fits/mean_{score_type.lower()}_{level}_comparison.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_score_distributions(results_df: pd.DataFrame, level: str):
    """
    Create histograms of AIC and BIC scores.
    
    Args:
        results_df: DataFrame containing the fit results
        level: 'patch' or 'env' to indicate analysis level
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot AIC and BIC distributions
    for metric, ax, title in zip(['aic', 'bic'], [ax1, ax2], 
                                ['AIC Scores', 'BIC Scores']):
        for patch_type in sorted(results_df['patch_type'].unique()):
            patch_data = results_df[results_df['patch_type'] == patch_type]
            
            # Filter out infinite values
            finite_scores = patch_data[metric][~np.isinf(patch_data[metric])]
            if len(finite_scores) > 0:
                sns.histplot(data=finite_scores, ax=ax, label=patch_type, 
                           alpha=0.5, bins=20)

        ax.set_xlabel(f'{metric.upper()} Score')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {title} by Patch Type\n({level.title()} Level)')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'figures/distribution_fits/{level}_score_distributions.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def main():
    """Main function to run the analysis."""
    # Load data
    df = pd.read_csv("leheron_trialbytrial/leheron_trialbytrial.csv")
    
    # Analyze distributions
    env_results_df, patch_results_df = analyze_leaving_times(df)
    
    # Save results
    env_results_df.to_csv('results/distribution_fits_env.csv', index=False)
    patch_results_df.to_csv('results/distribution_fits_patch.csv', index=False)
    
    # Create summary plots for both patch-level and environment-level analyses
    for results_df, level in [(patch_results_df, 'patch'), (env_results_df, 'env')]:
        # Plot AIC and BIC summaries
        plot_summary_scores(results_df, 'aic', level)
        plot_summary_scores(results_df, 'bic', level)
        
        # Plot score distributions
        plot_score_distributions(results_df, level)
    
    # Print summary statistics
    print("\nBest fitting distributions by AIC:")
    for level, results_df in [('Patch', patch_results_df), ('Environment', env_results_df)]:
        print(f"\n{level}-level analysis:")
        best_fits = results_df.loc[results_df.groupby('patch_type')['aic'].idxmin()]
        print(best_fits[['patch_type', 'distribution', 'aic']])

if __name__ == "__main__":
    main() 