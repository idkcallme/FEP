#!/usr/bin/env python3
"""
Empirical Validation Framework
=============================

Implements rigorous empirical validation protocols as required by peer review.
Provides statistical significance testing, baseline comparisons, and
reproducible experimental procedures to replace unsubstantiated claims.

This framework addresses the peer review requirement for proper experimental
validation with confidence intervals and statistical analysis.
"""

import numpy as np
import torch
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from scipy import stats
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for empirical validation experiments."""
    # Statistical parameters
    significance_level: float = 0.05  # Alpha for statistical tests
    confidence_level: float = 0.95   # Confidence interval level
    power_threshold: float = 0.8     # Minimum statistical power
    effect_size_threshold: float = 0.2  # Minimum meaningful effect size
    
    # Experimental design
    n_bootstrap_samples: int = 1000   # Bootstrap iterations
    n_cross_validation_folds: int = 5  # K-fold CV
    n_random_seeds: int = 10          # Multiple random seeds
    test_size: float = 0.2            # Hold-out test set size
    
    # Sample size parameters
    min_samples_per_group: int = 30   # Minimum for statistical validity
    max_samples_per_group: int = 1000  # Computational limit
    
    # Reproducibility
    random_seed: int = 42
    deterministic_mode: bool = True


@dataclass 
class StatisticalResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    power: Optional[float] = None
    interpretation: str = ""


class PowerAnalysis:
    """Statistical power analysis for sample size determination."""
    
    @staticmethod
    def compute_sample_size(effect_size: float, power: float = 0.8, alpha: float = 0.05) -> int:
        """
        Compute required sample size for given effect size and power.
        
        Uses Cohen's conventions for effect sizes:
        - Small: 0.2
        - Medium: 0.5  
        - Large: 0.8
        """
        from scipy.stats import norm
        
        # Two-tailed test
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        # Sample size calculation for two-sample t-test
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))
    
    @staticmethod
    def compute_achieved_power(n: int, effect_size: float, alpha: float = 0.05) -> float:
        """Compute achieved statistical power for given sample size."""
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = effect_size * np.sqrt(n/2) - z_alpha
        
        power = norm.cdf(z_beta)
        return power


class BaselineComparison:
    """Implements proper baseline comparison protocols."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.baselines = {}
        self.results_history = []
    
    def register_baseline(self, name: str, implementation: Callable, description: str = ""):
        """Register a baseline implementation for comparison."""
        self.baselines[name] = {
            'implementation': implementation,
            'description': description,
            'results': []
        }
        logger.info(f"Registered baseline: {name}")
    
    def run_comparison(self, test_implementation: Callable, test_data: List[Any], 
                      test_name: str = "test_system") -> Dict[str, StatisticalResult]:
        """
        Run comprehensive comparison between test implementation and all baselines.
        
        Args:
            test_implementation: Function to test
            test_data: List of test cases
            test_name: Name for the test implementation
            
        Returns:
            Dictionary of statistical comparison results
        """
        if not self.baselines:
            raise ValueError("No baselines registered. Use register_baseline() first.")
        
        logger.info(f"Running baseline comparison for {test_name}")
        
        # Run test implementation
        test_results = self._run_implementation(test_implementation, test_data, test_name)
        
        # Run all baselines
        baseline_results = {}
        for baseline_name, baseline_info in self.baselines.items():
            baseline_results[baseline_name] = self._run_implementation(
                baseline_info['implementation'], test_data, baseline_name
            )
        
        # Statistical comparisons
        comparison_results = {}
        for baseline_name, baseline_data in baseline_results.items():
            comparison_results[baseline_name] = self._statistical_comparison(
                test_results, baseline_data, test_name, baseline_name
            )
        
        return comparison_results
    
    def _run_implementation(self, implementation: Callable, test_data: List[Any], 
                          name: str) -> Dict[str, List[float]]:
        """Run implementation on test data and collect metrics."""
        results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'processing_time': [],
            'confidence_scores': []
        }
        
        for data_point in test_data:
            start_time = time.time()
            
            try:
                # Run implementation
                output = implementation(data_point)
                processing_time = time.time() - start_time
                
                # Extract metrics (implementation-specific)
                metrics = self._extract_metrics(output, data_point)
                
                # Store results
                for metric_name, value in metrics.items():
                    if metric_name in results:
                        results[metric_name].append(value)
                
                results['processing_time'].append(processing_time)
                
            except Exception as e:
                logger.warning(f"Implementation {name} failed on data point: {e}")
                # Record failure
                for metric_name in results:
                    results[metric_name].append(0.0)
        
        return results
    
    def _extract_metrics(self, output: Any, ground_truth: Any) -> Dict[str, float]:
        """Extract performance metrics from implementation output."""
        # This is a placeholder - should be customized for specific implementations
        if isinstance(output, dict):
            return {
                'accuracy': output.get('accuracy', 0.0),
                'precision': output.get('precision', 0.0),
                'recall': output.get('recall', 0.0),
                'f1_score': output.get('f1_score', 0.0),
                'confidence_scores': output.get('confidence', 0.5)
            }
        else:
            # Default binary classification metrics
            return {
                'accuracy': float(output == ground_truth.get('label', 0)),
                'precision': 1.0 if output == ground_truth.get('label', 0) else 0.0,
                'recall': 1.0 if output == ground_truth.get('label', 0) else 0.0,
                'f1_score': 1.0 if output == ground_truth.get('label', 0) else 0.0,
                'confidence_scores': 0.5
            }
    
    def _statistical_comparison(self, test_results: Dict[str, List[float]], 
                              baseline_results: Dict[str, List[float]],
                              test_name: str, baseline_name: str) -> Dict[str, StatisticalResult]:
        """Perform statistical comparison between test and baseline."""
        comparisons = {}
        
        for metric_name in test_results:
            if metric_name in baseline_results:
                test_values = np.array(test_results[metric_name])
                baseline_values = np.array(baseline_results[metric_name])
                
                # Perform t-test
                statistic, p_value = stats.ttest_ind(test_values, baseline_values)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(test_values) - 1) * np.var(test_values, ddof=1) + 
                                    (len(baseline_values) - 1) * np.var(baseline_values, ddof=1)) / 
                                   (len(test_values) + len(baseline_values) - 2))
                effect_size = (np.mean(test_values) - np.mean(baseline_values)) / pooled_std
                
                # Confidence interval for difference in means
                se_diff = pooled_std * np.sqrt(1/len(test_values) + 1/len(baseline_values))
                df = len(test_values) + len(baseline_values) - 2
                t_critical = stats.t.ppf(1 - self.config.significance_level/2, df)
                mean_diff = np.mean(test_values) - np.mean(baseline_values)
                ci_lower = mean_diff - t_critical * se_diff
                ci_upper = mean_diff + t_critical * se_diff
                
                # Statistical power
                power = PowerAnalysis.compute_achieved_power(
                    len(test_values), abs(effect_size), self.config.significance_level
                )
                
                # Interpretation
                is_significant = p_value < self.config.significance_level
                interpretation = self._interpret_comparison(
                    mean_diff, effect_size, is_significant, power
                )
                
                comparisons[metric_name] = StatisticalResult(
                    test_name=f"{test_name}_vs_{baseline_name}_{metric_name}",
                    statistic=statistic,
                    p_value=p_value,
                    effect_size=effect_size,
                    confidence_interval=(ci_lower, ci_upper),
                    is_significant=is_significant,
                    power=power,
                    interpretation=interpretation
                )
        
        return comparisons
    
    def _interpret_comparison(self, mean_diff: float, effect_size: float, 
                            is_significant: bool, power: float) -> str:
        """Generate interpretation of statistical comparison."""
        interpretation_parts = []
        
        # Significance
        if is_significant:
            interpretation_parts.append("Statistically significant difference")
        else:
            interpretation_parts.append("No statistically significant difference")
        
        # Effect size interpretation
        abs_effect_size = abs(effect_size)
        if abs_effect_size < 0.2:
            interpretation_parts.append("negligible effect size")
        elif abs_effect_size < 0.5:
            interpretation_parts.append("small effect size")
        elif abs_effect_size < 0.8:
            interpretation_parts.append("medium effect size")
        else:
            interpretation_parts.append("large effect size")
        
        # Direction
        if mean_diff > 0:
            interpretation_parts.append("(test system performs better)")
        elif mean_diff < 0:
            interpretation_parts.append("(baseline performs better)")
        
        # Power assessment
        if power < 0.8:
            interpretation_parts.append(f"LOW POWER ({power:.2f}) - results may be unreliable")
        
        return "; ".join(interpretation_parts)


class CrossValidationFramework:
    """Implements rigorous cross-validation protocols."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.cv_results = []
    
    def run_cross_validation(self, implementation: Callable, dataset: List[Any], 
                           labels: List[Any] = None) -> Dict[str, Any]:
        """
        Run k-fold cross-validation with statistical analysis.
        
        Args:
            implementation: Function to validate
            dataset: Input data
            labels: Ground truth labels (if applicable)
            
        Returns:
            Cross-validation results with statistics
        """
        logger.info(f"Running {self.config.n_cross_validation_folds}-fold cross-validation")
        
        # Initialize k-fold
        kf = KFold(n_splits=self.config.n_cross_validation_folds, 
                  shuffle=True, random_state=self.config.random_seed)
        
        fold_results = []
        
        # Run cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            logger.info(f"Running fold {fold_idx + 1}/{self.config.n_cross_validation_folds}")
            
            # Split data
            train_data = [dataset[i] for i in train_idx]
            val_data = [dataset[i] for i in val_idx]
            
            if labels:
                train_labels = [labels[i] for i in train_idx]
                val_labels = [labels[i] for i in val_idx]
            else:
                train_labels = val_labels = None
            
            # Train and validate
            fold_result = self._run_fold(implementation, train_data, val_data, 
                                       train_labels, val_labels, fold_idx)
            fold_results.append(fold_result)
        
        # Aggregate results
        aggregated_results = self._aggregate_cv_results(fold_results)
        
        return aggregated_results
    
    def _run_fold(self, implementation: Callable, train_data: List[Any], 
                 val_data: List[Any], train_labels: Optional[List[Any]], 
                 val_labels: Optional[List[Any]], fold_idx: int) -> Dict[str, Any]:
        """Run single fold of cross-validation."""
        
        fold_metrics = {
            'fold_index': fold_idx,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'metrics': {}
        }
        
        try:
            # Training phase (if implementation supports it)
            if hasattr(implementation, 'fit') or hasattr(implementation, 'train'):
                if hasattr(implementation, 'fit'):
                    implementation.fit(train_data, train_labels)
                else:
                    implementation.train(train_data, train_labels)
            
            # Validation phase
            val_predictions = []
            val_processing_times = []
            
            for val_sample in val_data:
                start_time = time.time()
                prediction = implementation(val_sample)
                processing_time = time.time() - start_time
                
                val_predictions.append(prediction)
                val_processing_times.append(processing_time)
            
            # Compute metrics
            if val_labels:
                # Classification metrics
                accuracy = accuracy_score(val_labels, val_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    val_labels, val_predictions, average='weighted'
                )
                
                fold_metrics['metrics'] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'avg_processing_time': np.mean(val_processing_times),
                    'std_processing_time': np.std(val_processing_times)
                }
            else:
                # Generic metrics
                fold_metrics['metrics'] = {
                    'avg_processing_time': np.mean(val_processing_times),
                    'std_processing_time': np.std(val_processing_times),
                    'predictions_made': len(val_predictions)
                }
                
        except Exception as e:
            logger.error(f"Fold {fold_idx} failed: {e}")
            fold_metrics['error'] = str(e)
            fold_metrics['metrics'] = {}
        
        return fold_metrics
    
    def _aggregate_cv_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate cross-validation results across folds."""
        
        # Collect metrics across folds
        all_metrics = {}
        successful_folds = []
        
        for fold_result in fold_results:
            if 'error' not in fold_result and fold_result['metrics']:
                successful_folds.append(fold_result)
                for metric_name, value in fold_result['metrics'].items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
        
        # Compute statistics
        aggregated = {
            'n_folds': len(fold_results),
            'successful_folds': len(successful_folds),
            'failed_folds': len(fold_results) - len(successful_folds),
            'fold_results': fold_results,
            'aggregated_metrics': {}
        }
        
        for metric_name, values in all_metrics.items():
            values_array = np.array(values)
            
            # Basic statistics
            mean_val = np.mean(values_array)
            std_val = np.std(values_array, ddof=1)
            
            # Confidence interval
            n = len(values_array)
            se = std_val / np.sqrt(n)
            t_critical = stats.t.ppf(1 - self.config.significance_level/2, n-1)
            ci_lower = mean_val - t_critical * se
            ci_upper = mean_val + t_critical * se
            
            aggregated['aggregated_metrics'][metric_name] = {
                'mean': mean_val,
                'std': std_val,
                'min': np.min(values_array),
                'max': np.max(values_array),
                'median': np.median(values_array),
                'confidence_interval': (ci_lower, ci_upper),
                'values': values.copy()
            }
        
        return aggregated


class ReproducibilityFramework:
    """Ensures experimental reproducibility."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_logs = []
    
    def run_reproducible_experiment(self, experiment_fn: Callable, 
                                  experiment_args: Dict[str, Any],
                                  experiment_name: str) -> Dict[str, Any]:
        """
        Run experiment with multiple random seeds for reproducibility assessment.
        
        Args:
            experiment_fn: Experiment function to run
            experiment_args: Arguments for experiment function
            experiment_name: Name of experiment
            
        Returns:
            Results across multiple seeds with reproducibility metrics
        """
        logger.info(f"Running reproducible experiment: {experiment_name}")
        
        seed_results = []
        
        for seed_idx, seed in enumerate(range(self.config.random_seed, 
                                            self.config.random_seed + self.config.n_random_seeds)):
            logger.info(f"Running with seed {seed} ({seed_idx + 1}/{self.config.n_random_seeds})")
            
            # Set reproducibility
            self._set_random_seeds(seed)
            
            # Run experiment
            try:
                experiment_args_with_seed = experiment_args.copy()
                experiment_args_with_seed['random_seed'] = seed
                
                result = experiment_fn(**experiment_args_with_seed)
                result['seed'] = seed
                result['seed_index'] = seed_idx
                
                seed_results.append(result)
                
            except Exception as e:
                logger.error(f"Experiment failed with seed {seed}: {e}")
                seed_results.append({
                    'seed': seed,
                    'seed_index': seed_idx,
                    'error': str(e),
                    'failed': True
                })
        
        # Analyze reproducibility
        reproducibility_analysis = self._analyze_reproducibility(seed_results, experiment_name)
        
        return {
            'experiment_name': experiment_name,
            'config': self.config,
            'seed_results': seed_results,
            'reproducibility_analysis': reproducibility_analysis
        }
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if self.config.deterministic_mode:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _analyze_reproducibility(self, seed_results: List[Dict[str, Any]], 
                               experiment_name: str) -> Dict[str, Any]:
        """Analyze reproducibility across random seeds."""
        
        # Filter successful runs
        successful_results = [r for r in seed_results if not r.get('failed', False)]
        
        if len(successful_results) < 2:
            return {
                'status': 'insufficient_data',
                'successful_runs': len(successful_results),
                'total_runs': len(seed_results)
            }
        
        # Extract numeric metrics
        numeric_metrics = {}
        for result in successful_results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and key not in ['seed', 'seed_index']:
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        # Compute reproducibility statistics
        reproducibility_stats = {}
        for metric_name, values in numeric_metrics.items():
            if len(values) > 1:
                values_array = np.array(values)
                coefficient_of_variation = np.std(values_array) / np.mean(values_array) if np.mean(values_array) != 0 else float('inf')
                
                reproducibility_stats[metric_name] = {
                    'mean': np.mean(values_array),
                    'std': np.std(values_array),
                    'coefficient_of_variation': coefficient_of_variation,
                    'min': np.min(values_array),
                    'max': np.max(values_array),
                    'range': np.max(values_array) - np.min(values_array),
                    'is_reproducible': coefficient_of_variation < 0.1  # 10% CV threshold
                }
        
        # Overall reproducibility assessment
        reproducible_metrics = sum(1 for stats in reproducibility_stats.values() 
                                 if stats['is_reproducible'])
        total_metrics = len(reproducibility_stats)
        reproducibility_rate = reproducible_metrics / total_metrics if total_metrics > 0 else 0
        
        return {
            'status': 'analyzed',
            'successful_runs': len(successful_results),
            'total_runs': len(seed_results),
            'reproducibility_rate': reproducibility_rate,
            'reproducible_metrics': reproducible_metrics,
            'total_metrics': total_metrics,
            'metric_statistics': reproducibility_stats,
            'overall_assessment': 'reproducible' if reproducibility_rate >= 0.8 else 'variable'
        }


def main():
    """Test the empirical validation framework."""
    print("Testing Empirical Validation Framework")
    print("=" * 45)
    
    # Configuration
    config = ExperimentConfig(
        n_cross_validation_folds=3,
        n_random_seeds=5,
        n_bootstrap_samples=100
    )
    
    # Mock implementations for testing
    def test_implementation(x):
        """Mock test implementation."""
        return np.random.random() > 0.4  # 60% accuracy
    
    def baseline_random(x):
        """Random baseline."""
        return np.random.random() > 0.5  # 50% accuracy
    
    def baseline_simple(x):
        """Simple rule-based baseline."""
        return len(str(x)) > 5  # Simple heuristic
    
    # Generate test data
    test_data = [{'input': f'test_input_{i}', 'label': i % 2} for i in range(100)]
    
    print("1. Testing Baseline Comparison Framework")
    print("-" * 30)
    
    # Baseline comparison
    baseline_comp = BaselineComparison(config)
    baseline_comp.register_baseline('random', baseline_random, "Random baseline")
    baseline_comp.register_baseline('simple_rule', baseline_simple, "Simple rule baseline")
    
    # Run comparison
    comparison_results = baseline_comp.run_comparison(test_implementation, test_data)
    
    for baseline_name, results in comparison_results.items():
        print(f"\nComparison with {baseline_name}:")
        for metric, stat_result in results.items():
            print(f"  {metric}: p={stat_result.p_value:.3f}, effect_size={stat_result.effect_size:.3f}")
            print(f"    {stat_result.interpretation}")
    
    print("\n2. Testing Cross-Validation Framework")
    print("-" * 30)
    
    # Cross-validation
    cv_framework = CrossValidationFramework(config)
    cv_results = cv_framework.run_cross_validation(test_implementation, 
                                                  [d['input'] for d in test_data],
                                                  [d['label'] for d in test_data])
    
    print("Cross-validation results:")
    for metric_name, stats in cv_results['aggregated_metrics'].items():
        print(f"  {metric_name}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"    95% CI: [{stats['confidence_interval'][0]:.3f}, {stats['confidence_interval'][1]:.3f}]")
    
    print("\n3. Testing Reproducibility Framework")  
    print("-" * 30)
    
    # Reproducibility test
    def mock_experiment(data, random_seed=42):
        np.random.seed(random_seed)
        return {
            'accuracy': np.random.uniform(0.5, 0.8),
            'processing_time': np.random.uniform(0.1, 0.5)
        }
    
    repro_framework = ReproducibilityFramework(config)
    repro_results = repro_framework.run_reproducible_experiment(
        mock_experiment,
        {'data': test_data},
        'mock_experiment_test'
    )
    
    repro_analysis = repro_results['reproducibility_analysis']
    print(f"Reproducibility rate: {repro_analysis['reproducibility_rate']:.2%}")
    print(f"Overall assessment: {repro_analysis['overall_assessment']}")
    
    for metric, stats in repro_analysis['metric_statistics'].items():
        print(f"  {metric}: CV = {stats['coefficient_of_variation']:.3f} "
              f"(reproducible: {stats['is_reproducible']})")
    
    print("\nEmpirical validation framework testing completed!")


if __name__ == "__main__":
    main()
