# Rigorous Experimental Framework for FEP Cognitive Architecture

## Purpose

This document outlines a proper experimental framework to address the lack of scientific rigor identified in the current research. It defines falsifiable hypotheses, experimental designs, and statistical validation procedures.

## Current Problems

1. **Vague claims without statistical validation**
2. **No proper baselines or control groups**
3. **Arbitrary performance metrics**
4. **No reproducible datasets**
5. **Missing statistical significance testing**

## Proposed Experimental Framework

### Hypothesis 1: Environmental Adaptation

**Falsifiable Hypothesis**: "A hierarchical FEP agent with meta-cognitive monitoring recovers to baseline variational free energy within 50 timesteps after an environmental rule change, compared to >100 timesteps for a baseline agent (p < 0.05, n = 50 trials)."

**Experimental Design**:
- **Environment**: Grid world with reward structure that changes at timestep 500
- **Agents**: FEP+MCM vs. Vanilla Q-learning baseline
- **Metrics**: Timesteps to recovery (free energy within 5% of pre-change baseline)
- **Sample Size**: 50 independent runs per agent
- **Statistical Test**: Mann-Whitney U test

**Data Collection**:
- Record free energy at each timestep
- Measure adaptation time (timesteps to recovery)
- Track cumulative reward during adaptation period
- Log all hyperparameters and random seeds

### Hypothesis 2: Anomaly Detection Capability

**Falsifiable Hypothesis**: "The MCM component detects distributional shifts with precision >0.7 and recall >0.6 on standard anomaly detection benchmarks, significantly better than threshold-based baselines (p < 0.05)."

**Experimental Design**:
- **Dataset**: MNIST with synthetic anomalies (rotated, noisy, out-of-distribution)
- **Baselines**: Statistical threshold, Isolation Forest, One-Class SVM
- **Metrics**: Precision, Recall, F1-score, AUC-ROC
- **Validation**: 5-fold cross-validation
- **Statistical Test**: Paired t-test across folds

### Hypothesis 3: Scalability Analysis

**Falsifiable Hypothesis**: "Computational complexity of FEP agent scales as O(n²) where n is state dimension, with empirical validation up to n=1000."

**Experimental Design**:
- **State Dimensions**: [10, 25, 50, 100, 250, 500, 1000]
- **Measurements**: Wall-clock time, memory usage, numerical stability
- **Repetitions**: 20 runs per dimension
- **Analysis**: Regression analysis of time vs. dimension
- **Validation**: R² > 0.9 for complexity model fit

### Hypothesis 4: Security Feature Evaluation

**Falsifiable Hypothesis**: "Unicode anomaly detection achieves >0.8 precision and >0.5 recall on adversarial prompt datasets, with false positive rate <0.1 on benign prompts."

**Experimental Design**:
- **Dataset**: AdvBench + custom adversarial prompts + benign baseline
- **Metrics**: Precision, Recall, False Positive Rate
- **Baselines**: Simple keyword filtering, commercial prompt filters
- **Sample Size**: 1000 adversarial + 1000 benign prompts
- **Validation**: Bootstrap confidence intervals

## Required Datasets

### 1. Environmental Adaptation Dataset
- **Grid World Environments**: 10x10 grids with varying reward structures
- **Rule Changes**: Reward location shifts, penalty introductions
- **Baseline Trajectories**: Pre-computed optimal policies
- **Size**: 1000 environment configurations

### 2. Anomaly Detection Dataset  
- **Base Distribution**: MNIST training set
- **Anomalies**: Rotated digits (15°, 30°, 45°), Gaussian noise, Fashion-MNIST
- **Labels**: Binary anomaly/normal classification
- **Size**: 10,000 normal + 2,000 anomalous samples

### 3. Security Evaluation Dataset
- **Adversarial Prompts**: Unicode obfuscation, injection attacks, jailbreaks
- **Benign Prompts**: Normal conversation, technical questions, creative writing
- **Sources**: AdvBench, custom generation, community datasets
- **Size**: 1000 adversarial + 1000 benign prompts

### 4. Scalability Benchmarks
- **Synthetic Environments**: Parameterized by state/action dimensions
- **Complexity Variations**: Linear, quadratic, exponential scaling
- **Resource Constraints**: Memory limits, time budgets
- **Size**: 7 dimension levels × 20 repetitions

## Statistical Analysis Plan

### Power Analysis
- **Effect Size**: Cohen's d = 0.8 (large effect)
- **Power**: 0.8
- **Alpha**: 0.05
- **Required Sample Size**: n ≥ 26 per group (calculated)

### Multiple Comparisons Correction
- **Method**: Bonferroni correction for family-wise error rate
- **Adjusted Alpha**: 0.05/4 = 0.0125 for 4 primary hypotheses

### Confidence Intervals
- **All Effect Sizes**: 95% confidence intervals
- **Bootstrap Methods**: For non-parametric distributions
- **Reporting**: Effect size + CI + p-value for all tests

### Reproducibility Requirements
- **Random Seeds**: Fixed and documented for all experiments
- **Hyperparameters**: Grid search with cross-validation
- **Environment Versions**: Pinned dependencies in requirements.txt
- **Code Availability**: All experimental code in public repository

## Baseline Comparisons

### Cognitive Architecture Baselines
1. **Vanilla Active Inference**: Standard implementation without MCM
2. **Q-Learning**: Standard reinforcement learning baseline
3. **Random Policy**: Uniform random action selection
4. **Optimal Policy**: Oracle baseline where available

### Anomaly Detection Baselines
1. **Statistical Thresholds**: Mean ± 2σ outlier detection
2. **Isolation Forest**: Ensemble-based anomaly detection
3. **One-Class SVM**: Support vector-based outlier detection
4. **Autoencoder**: Reconstruction error-based detection

### Security Baselines
1. **Keyword Filtering**: Simple blacklist-based filtering
2. **Commercial Filters**: OpenAI Moderation API
3. **Rule-Based Systems**: Hand-crafted security rules
4. **No Filtering**: Baseline false positive rate

## Implementation Plan

### Phase 1: Dataset Creation (4 weeks)
- Generate synthetic environments
- Curate adversarial prompt datasets
- Create anomaly detection benchmarks
- Validate dataset quality

### Phase 2: Baseline Implementation (3 weeks)
- Implement all baseline methods
- Validate baseline performance
- Create evaluation pipelines
- Test statistical analysis code

### Phase 3: Experimental Execution (6 weeks)
- Run all experiments with proper controls
- Collect performance metrics
- Perform statistical analysis
- Generate confidence intervals

### Phase 4: Analysis and Reporting (2 weeks)
- Compile results with statistical tests
- Create visualizations and tables
- Write methods and results sections
- Prepare supplementary materials

## Expected Outcomes

### Success Criteria
- **Hypothesis 1**: Significant improvement in adaptation time
- **Hypothesis 2**: Competitive anomaly detection performance
- **Hypothesis 3**: Confirmed complexity scaling behavior
- **Hypothesis 4**: Reasonable security filtering performance

### Failure Criteria
- **No Significant Differences**: Null hypothesis not rejected
- **Worse Than Baselines**: FEP performance below simple baselines
- **Poor Scaling**: Complexity worse than predicted
- **High False Positives**: Security features unusable in practice

### Reporting Standards
- **Negative Results**: Report all outcomes, including failures
- **Effect Sizes**: Always report with confidence intervals
- **Limitations**: Discuss threats to validity
- **Reproducibility**: Provide all code and data

## Quality Assurance

### Experimental Validation
- **Pilot Studies**: Small-scale validation of experimental design
- **Code Review**: Independent verification of analysis code
- **Data Validation**: Sanity checks on all datasets
- **Reproducibility Tests**: Independent replication of key results

### Statistical Rigor
- **Assumptions Testing**: Validate statistical test assumptions
- **Sensitivity Analysis**: Test robustness to parameter changes
- **Cross-Validation**: Prevent overfitting in model selection
- **Preregistration**: Document hypotheses before data collection

This framework addresses the fundamental scientific credibility issues by providing:
1. **Falsifiable hypotheses** with specific predictions
2. **Proper experimental controls** and baselines
3. **Statistical validation** with appropriate tests
4. **Reproducible methodology** with public datasets
5. **Honest reporting** of both positive and negative results
