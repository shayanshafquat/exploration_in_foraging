import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from scipy.stats import normaltest, skewtest
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

def compute_skewness(data: np.ndarray) -> Tuple[float, float]:
    """
    Compute skewness and perform skewness test.
    
    Args:
        data: Array of observations
    
    Returns:
        Tuple of (skewness value, p-value from test)
    """
    skewness = stats.skew(data)
    _, p_value = skewtest(data)
    return skewness, p_value

def assess_sample_size(data: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Assess if sample size is sufficient for distribution estimation.
    
    Args:
        data: Array of observations
        alpha: Significance level for power analysis
    
    Returns:
        Dictionary containing assessment results
    """
    n = len(data)
    
    # Rule of thumb checks
    min_sample_size = 30  # Minimum for CLT
    recommended_size = 100  # Recommended for distribution fitting
    
    # Effect size for normality (using d'Agostino-Pearson test)
    _, norm_p_value = normaltest(data)
    
    return {
        'n_samples': n,
        'sufficient_for_clt': n >= min_sample_size,
        'sufficient_for_fitting': n >= recommended_size,
        'normality_p_value': norm_p_value,
        'assessment': 'Sufficient' if n >= recommended_size else 
                     'Minimal' if n >= min_sample_size else 'Insufficient'
    }

def fit_population_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit distributions to population-level data for each patch x environment combination.
    First normalizes within subjects, then combines normalized data.
    
    Args:
        df: DataFrame containing trial data
    
    Returns:
        DataFrame containing fit results
    """
    results = []
    distributions = [
        ('normal', stats.norm),
        ('lognormal', stats.lognorm),
        ('gamma', stats.gamma),
        ('exponential', stats.expon),
        ('weibull', stats.weibull_min)
    ]
    
    # Create output directory
    os.makedirs('figures/population_fits', exist_ok=True)
    
    # Analyze each patch x environment combination
    for patch in sorted(df['patch'].unique()):
        for env in sorted(df['env'].unique()):
            # First normalize within each subject
            normalized_times_all = []
            
            for subject in df['sub'].unique():
                # Get subject's overall stats for normalization
                subject_times = df[df['sub'] == subject]['leaveT'].values
                if len(subject_times) < 5:  # Skip subjects with too few trials
                    continue
                subject_mean = np.mean(subject_times)
                subject_std = np.std(subject_times)
                
                # Get this subject's leaving times for this patch x env combination
                leave_times = df[
                    (df['sub'] == subject) & 
                    (df['patch'] == patch) & 
                    (df['env'] == env)
                ]['leaveT'].values
                
                if len(leave_times) > 0:
                    # Normalize within subject
                    subject_normalized = (leave_times - subject_mean) / subject_std
                    normalized_times_all.extend(subject_normalized)
            
            normalized_times_all = np.array(normalized_times_all)
            
            if len(normalized_times_all) < 30:  # Skip if too few observations
                continue
            
            # Compute skewness
            skewness, skew_p = compute_skewness(normalized_times_all)
            
            # Assess sample size
            size_assessment = assess_sample_size(normalized_times_all)
            
            # Fit distributions
            fit_results = {}
            for dist_name, dist in distributions:
                try:
                    # Fit distribution
                    params = dist.fit(normalized_times_all)
                    
                    # Calculate goodness of fit
                    logL = np.sum(dist.logpdf(normalized_times_all, *params))
                    k = len(params)
                    n = len(normalized_times_all)
                    aic = 2 * k - 2 * logL
                    bic = np.log(n) * k - 2 * logL
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = stats.kstest(normalized_times_all, dist.name, params)
                    
                    fit_results[dist_name] = {
                        'params': params,
                        'logL': logL,
                        'aic': aic,
                        'bic': bic,
                        'ks_stat': ks_stat,
                        'ks_p': ks_p
                    }
                except Exception as e:
                    print(f"Failed to fit {dist_name} for Patch {patch}, Env {env}: {str(e)}")
            
            # Plot fits
            plt.figure(figsize=(10, 6))
            plt.hist(normalized_times_all, bins='auto', density=True, alpha=0.6, 
                    label='Data', color='gray')
            
            x = np.linspace(min(normalized_times_all), max(normalized_times_all), 200)
            for dist_name, res in fit_results.items():
                try:
                    dist = dict(distributions)[dist_name]
                    y = dist.pdf(x, *res['params'])
                    plt.plot(x, y, label=f'{dist_name}\nAIC: {res["aic"]:.2f}')
                except:
                    continue
            
            plt.title(f'Patch {patch} - Env {env}\nSkewness: {skewness:.2f} (p={skew_p:.3f})\n'
                     f'N={len(normalized_times_all)} trials')
            plt.xlabel('Normalized Leaving Time (Z-scored within subjects)')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(f'figures/population_fits/patch{patch}_env{env}_fits.png', 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            # Store results
            for dist_name, res in fit_results.items():
                results.append({
                    'patch_type': f'Patch {patch}',
                    'environment': f'Env {env}',
                    'distribution': dist_name,
                    'skewness': skewness,
                    'skewness_p': skew_p,
                    'n_samples': size_assessment['n_samples'],
                    'n_subjects': len(df['sub'].unique()),
                    'sample_size_assessment': size_assessment['assessment'],
                    'normality_p': size_assessment['normality_p_value'],
                    'aic': res['aic'],
                    'bic': res['bic'],
                    'ks_stat': res['ks_stat'],
                    'ks_p': res['ks_p']
                })
    
    return pd.DataFrame(results)

def plot_score_summaries(results_df: pd.DataFrame):
    """
    Create summary plots of mean AIC and BIC scores by patch type and distribution.
    
    Args:
        results_df: DataFrame containing the distribution fitting results
    """
    # Create figure directory if it doesn't exist
    os.makedirs('figures/population_fits', exist_ok=True)
    
    # Calculate mean scores for each patch type and distribution
    mean_scores = (results_df
        .groupby(['patch_type', 'distribution'])
        .agg({
            'aic': 'mean',
            'bic': 'mean'
        })
        .reset_index()
    )
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Set bar width and positions
    distributions = mean_scores['distribution'].unique()
    n_distributions = len(distributions)
    bar_width = 0.8 / n_distributions
    
    # Plot AIC means with log scale
    for i, dist in enumerate(distributions):
        dist_data = mean_scores[mean_scores['distribution'] == dist]
        x = np.arange(len(dist_data['patch_type'].unique()))
        offset = (i - n_distributions/2 + 0.5) * bar_width
        ax1.bar(x + offset, dist_data['aic'], bar_width, 
               label=dist, alpha=0.8)
    
    ax1.set_yscale('log')
    ax1.set_title('Mean AIC Scores by Patch Type and Distribution')
    ax1.set_xlabel('Patch Type')
    ax1.set_ylabel('Mean AIC Score (log scale)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dist_data['patch_type'].unique(), rotation=45)
    ax1.legend(title='Distribution')
    
    # Plot BIC means with log scale
    for i, dist in enumerate(distributions):
        dist_data = mean_scores[mean_scores['distribution'] == dist]
        x = np.arange(len(dist_data['patch_type'].unique()))
        offset = (i - n_distributions/2 + 0.5) * bar_width
        ax2.bar(x + offset, dist_data['bic'], bar_width, 
               label=dist, alpha=0.8)
    
    ax2.set_yscale('log')
    ax2.set_title('Mean BIC Scores by Patch Type and Distribution')
    ax2.set_xlabel('Patch Type')
    ax2.set_ylabel('Mean BIC Score (log scale)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(dist_data['patch_type'].unique(), rotation=45)
    ax2.legend(title='Distribution')
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, which='both')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('figures/population_fits/mean_score_distributions.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create summary statistics table
    summary_stats = (results_df
        .groupby(['patch_type', 'distribution'])
        .agg({
            'aic': ['mean', 'std', 'min'],
            'bic': ['mean', 'std', 'min']
        })
        .round(2)
    )
    
    # Save summary statistics
    summary_stats.to_csv('results/score_summary_stats.csv')
    
    return summary_stats

def main():
    """Main function to run the analysis."""
    # Load data
    df = pd.read_csv("leheron_trialbytrial/leheron_trialbytrial.csv")
    
    # Perform population-level analysis
    results_df = fit_population_distributions(df)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/population_distribution_fits.csv', index=False)
    
    # Create and print score summaries
    summary_stats = plot_score_summaries(results_df)
    
    # Print existing summaries
    print("\nSample Size Assessment:")
    size_summary = results_df.groupby(['patch_type', 'environment'])['sample_size_assessment'].first()
    print(size_summary)
    
    print("\nSkewness Analysis:")
    skew_summary = results_df.groupby(['patch_type', 'environment'])[['skewness', 'skewness_p']].first()
    print(skew_summary)
    
    print("\nBest fitting distributions by AIC:")
    best_fits = results_df.loc[results_df.groupby(['patch_type', 'environment'])['aic'].idxmin()]
    print(best_fits[['patch_type', 'environment', 'distribution', 'aic', 'ks_p']])
    
    print("\nScore Summary Statistics:")
    print(summary_stats)

if __name__ == "__main__":
    main()