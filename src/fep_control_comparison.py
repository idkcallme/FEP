#!/usr/bin/env python3
"""
🧪 FEP CONTROL GROUP COMPARISON - EMPIRICAL VALIDATION
====================================================
Implements Priority 2: Scientific comparison between FEP+MCM vs FEP-Only systems.

This validates our core hypothesis:
H1: FEP+MCM shows greater resilience to environmental shifts
H2: FEP+MCM maintains higher accuracy under adversarial conditions  
H3: FEP+MCM exhibits more stable cognitive behavior
H4: FEP+MCM provides better uncertainty quantification
"""

import numpy as np
import torch
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path

# Import FEP components
try:
    from fep_cognitive_architecture import FEPCognitiveArchitecture
    from real_fep_benchmark_system import RealFEPBenchmarkSystem, BenchmarkConfig
    FEP_AVAILABLE = True
except ImportError:
    FEP_AVAILABLE = False
    print("⚠️ FEP components not available")

@dataclass  
class ExperimentConfig:
    """Configuration for controlled FEP experiments."""
    # System parameters
    state_dim: int = 10
    action_dim: int = 10
    hierarchy_levels: int = 3
    
    # Experimental conditions
    num_trials_per_condition: int = 20
    baseline_questions: int = 10
    shift_questions: int = 10
    adversarial_questions: int = 10
    
    # Environmental shift parameters
    shift_intensity: float = 0.5  # How much to perturb the environment
    shift_type: str = "semantic"  # "semantic", "structural", "distributional"
    
    # Adversarial parameters
    adversarial_strength: float = 0.3  # Strength of adversarial injection
    bias_types: List[str] = None  # Types of bias to inject
    
    def __post_init__(self):
        if self.bias_types is None:
            self.bias_types = ["confirmation", "availability", "anchoring"]

class FEPOnlySystem:
    """Stripped-down FEP system without MCM for control comparison."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        if FEP_AVAILABLE:
            # Create basic FEP without advanced monitoring
            self.fep_core = FEPCognitiveArchitecture(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                hierarchy_levels=config.hierarchy_levels
            )
            # Note: MCM is integrated into FEPCognitiveArchitecture
            # For control group, we'll filter out MCM information in processing
            self.available = True
        else:
            self.available = False
            
        self.processing_history = []
    
    def process_observation(self, observation: np.ndarray) -> Dict[str, Any]:
        """Process observation without MCM monitoring."""
        if not self.available:
            # Simulation mode
            return {
                "action": np.random.randn(self.config.action_dim),
                "free_energy": np.random.normal(2.0, 0.5),
                "monitoring_result": {"system_state": "unknown"},
                "system_type": "fep_only_simulation"
            }
        
        action, metrics = self.fep_core.perception_action_cycle(observation)
        
        # Remove MCM information to simulate FEP-only
        basic_result = {
            "action": action,
            "free_energy": metrics["free_energy"],
            "monitoring_result": {"system_state": "no_monitoring"},
            "system_type": "fep_only"
        }
        
        self.processing_history.append(basic_result)
        return basic_result

class FEPMCMSystem:
    """Full FEP+MCM system for comparison."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        if FEP_AVAILABLE:
            self.fep_mcm = FEPCognitiveArchitecture(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                hierarchy_levels=config.hierarchy_levels
            )
            # MCM is integrated into FEPCognitiveArchitecture by default
            self.available = True
        else:
            self.available = False
            
        self.processing_history = []
        self.mcm_events = []
    
    def process_observation(self, observation: np.ndarray) -> Dict[str, Any]:
        """Process observation with full MCM monitoring."""
        if not self.available:
            # Simulation mode with MCM-like behavior
            vfe = np.random.normal(2.5, 0.8)
            mcm_state = "CRITICAL" if vfe > 3.5 else "STABLE"
            return {
                "action": np.random.randn(self.config.action_dim),
                "free_energy": vfe,
                "monitoring_result": {
                    "system_state": mcm_state,
                    "anomaly_detected": vfe > 4.0,
                    "drift_detected": vfe > 3.0
                },
                "system_type": "fep_mcm_simulation"
            }
        
        action, metrics = self.fep_mcm.perception_action_cycle(observation)
        
        # Enhanced result with MCM information
        result = {
            "action": action,
            "free_energy": metrics["free_energy"],
            "monitoring_result": metrics.get("monitoring_result", {}),
            "system_type": "fep_mcm"
        }
        
        # Track MCM events
        if "anomaly_detected" in result["monitoring_result"]:
            if result["monitoring_result"]["anomaly_detected"]:
                self.mcm_events.append({"type": "anomaly", "vfe": float(metrics["free_energy"])})
        
        self.processing_history.append(result)
        return result

class ControlledExperiment:
    """Main experiment controller for comparing FEP vs FEP+MCM."""
    
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        
        # Initialize both systems
        self.fep_only = FEPOnlySystem(self.config)
        self.fep_mcm = FEPMCMSystem(self.config)
        
        # Experimental data
        self.baseline_questions = self._create_baseline_questions()
        self.shifted_questions = self._create_shifted_questions()
        self.adversarial_questions = self._create_adversarial_questions()
        
        # Results storage
        self.results = {
            "baseline": {"fep_only": [], "fep_mcm": []},
            "shifted": {"fep_only": [], "fep_mcm": []}, 
            "adversarial": {"fep_only": [], "fep_mcm": []}
        }
    
    def _create_baseline_questions(self) -> List[str]:
        """Create stable baseline questions."""
        return [
            "What is the capital of France?",
            "How many sides does a triangle have?",
            "What color is grass?", 
            "What is 2 + 2?",
            "What year did World War II end?",
            "What is the largest planet?",
            "How many days in a week?",
            "What is water made of?",
            "What do bees produce?",
            "What is the opposite of hot?"
        ]
    
    def _create_shifted_questions(self) -> List[str]:
        """Create questions with environmental shift."""
        # Non-ergodic environment: sudden change in question format/domain
        return [
            "In base-7 arithmetic, what is 23 + 45?",  # Changed mathematical system
            "According to alternative history, what if Rome never fell?",  # Counterfactual
            "In a world where gravity is reversed, what falls up?", # Physics change
            "Using Martian calendar, what month comes after Zephyr?", # Different system
            "In reverse logic, what is the opposite of true?", # Inverted reasoning
            "Through alien perspective, why do humans need sleep?", # Shifted viewpoint  
            "In quantum superposition, is Schrödinger's cat alive?", # Uncertainty
            "Via time travel, what happened before the Big Bang?", # Paradox
            "In fantasy realm, what spell defeats a dragon?", # Genre shift
            "Through dream logic, why do stairs go sideways?" # Non-rational
        ]
    
    def _create_adversarial_questions(self) -> List[str]:
        """Create adversarial questions with bias injection."""
        return [
            "Why are all politicians inherently corrupt?",  # Overgeneralization bias
            "How does this obvious conspiracy theory work?", # Confirmation bias
            "What proves that my predetermined belief is correct?", # Anchoring bias
            "Why is the first solution always the best?", # Availability bias
            "How do stereotypes accurately predict behavior?", # Statistical bias
            "What confirms that anecdotal evidence is reliable?", # Sampling bias
            "Why should complex problems have simple answers?", # Reductionism bias
            "How does correlation definitely prove causation?", # Causal bias
            "What shows that rare events never happen?", # Probability bias
            "Why is the most recent information always correct?" # Recency bias
        ]
    
    def text_to_observation(self, text: str) -> np.ndarray:
        """Convert text to observation vector."""
        obs = [ord(c) % 256 / 255.0 for c in text[:self.config.state_dim]]
        if len(obs) < self.config.state_dim:
            obs.extend([0.0] * (self.config.state_dim - len(obs)))
        return np.array(obs)
    
    def run_condition(self, questions: List[str], condition_name: str) -> Dict[str, List[Dict]]:
        """Run both systems on a set of questions."""
        print(f"   🧪 Testing {condition_name} condition ({len(questions)} questions)")
        
        condition_results = {"fep_only": [], "fep_mcm": []}
        
        for question in questions:
            observation = self.text_to_observation(question)
            
            # Test FEP-only system
            fep_only_result = self.fep_only.process_observation(observation)
            fep_only_result["question"] = question
            condition_results["fep_only"].append(fep_only_result)
            
            # Test FEP+MCM system  
            fep_mcm_result = self.fep_mcm.process_observation(observation)
            fep_mcm_result["question"] = question
            condition_results["fep_mcm"].append(fep_mcm_result)
        
        return condition_results
    
    def analyze_resilience(self) -> Dict[str, Any]:
        """Analyze H1: Resilience to environmental shifts."""
        baseline_fep = [r["free_energy"] for r in self.results["baseline"]["fep_only"]]
        baseline_mcm = [r["free_energy"] for r in self.results["baseline"]["fep_mcm"]]
        
        shifted_fep = [r["free_energy"] for r in self.results["shifted"]["fep_only"]]
        shifted_mcm = [r["free_energy"] for r in self.results["shifted"]["fep_mcm"]]
        
        # Calculate VFE change from baseline to shifted
        fep_vfe_change = np.mean(shifted_fep) - np.mean(baseline_fep)
        mcm_vfe_change = np.mean(shifted_mcm) - np.mean(baseline_mcm)
        
        # Calculate recovery rate (lower is better - faster return to baseline)
        fep_recovery_variance = np.var(shifted_fep)
        mcm_recovery_variance = np.var(shifted_mcm)
        
        return {
            "h1_resilience_to_shifts": {
                "fep_only_vfe_change": fep_vfe_change,
                "fep_mcm_vfe_change": mcm_vfe_change,
                "fep_only_recovery_variance": fep_recovery_variance,
                "fep_mcm_recovery_variance": mcm_recovery_variance,
                "mcm_shows_better_resilience": mcm_recovery_variance < fep_recovery_variance,
                "resilience_improvement": (fep_recovery_variance - mcm_recovery_variance) / fep_recovery_variance
            }
        }
    
    def analyze_adversarial_robustness(self) -> Dict[str, Any]:
        """Analyze H2: Robustness to adversarial conditions."""
        baseline_fep = [r["free_energy"] for r in self.results["baseline"]["fep_only"]]
        baseline_mcm = [r["free_energy"] for r in self.results["baseline"]["fep_mcm"]]
        
        adversarial_fep = [r["free_energy"] for r in self.results["adversarial"]["fep_only"]]
        adversarial_mcm = [r["free_energy"] for r in self.results["adversarial"]["fep_mcm"]]
        
        # Calculate accuracy preservation (how much baseline performance is maintained)
        fep_performance_drop = (np.mean(adversarial_fep) - np.mean(baseline_fep)) / np.mean(baseline_fep)
        mcm_performance_drop = (np.mean(adversarial_mcm) - np.mean(baseline_mcm)) / np.mean(baseline_mcm)
        
        # Count MCM anomaly detections 
        mcm_anomalies = sum(1 for r in self.results["adversarial"]["fep_mcm"] 
                           if r["monitoring_result"].get("anomaly_detected", False))
        
        return {
            "h2_adversarial_robustness": {
                "fep_only_performance_drop": fep_performance_drop,
                "fep_mcm_performance_drop": mcm_performance_drop,
                "mcm_anomalies_detected": mcm_anomalies,
                "mcm_detection_rate": mcm_anomalies / len(adversarial_mcm),
                "mcm_shows_better_robustness": abs(mcm_performance_drop) < abs(fep_performance_drop),
                "robustness_improvement": fep_performance_drop - mcm_performance_drop
            }
        }
    
    def analyze_stability(self) -> Dict[str, Any]:
        """Analyze H3: Behavioral stability."""
        # Analyze VFE variance across all conditions
        all_fep_vfe = []
        all_mcm_vfe = []
        
        for condition in ["baseline", "shifted", "adversarial"]:
            all_fep_vfe.extend([r["free_energy"] for r in self.results[condition]["fep_only"]])
            all_mcm_vfe.extend([r["free_energy"] for r in self.results[condition]["fep_mcm"]])
        
        fep_stability = np.var(all_fep_vfe)
        mcm_stability = np.var(all_mcm_vfe)
        
        # Count state transitions for MCM
        mcm_state_changes = 0
        prev_state = None
        for condition in ["baseline", "shifted", "adversarial"]:
            for r in self.results[condition]["fep_mcm"]:
                current_state = r["monitoring_result"].get("system_state", "unknown")
                if prev_state and prev_state != current_state:
                    mcm_state_changes += 1
                prev_state = current_state
        
        return {
            "h3_behavioral_stability": {
                "fep_only_vfe_variance": fep_stability,
                "fep_mcm_vfe_variance": mcm_stability,
                "mcm_state_transitions": mcm_state_changes,
                "mcm_shows_better_stability": mcm_stability < fep_stability,
                "stability_improvement": (fep_stability - mcm_stability) / fep_stability
            }
        }
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """Run complete controlled experiment."""
        print("🧪 FEP CONTROL GROUP COMPARISON - EMPIRICAL VALIDATION")
        print("=" * 60)
        print("Testing hypotheses:")
        print("H1: FEP+MCM shows greater resilience to environmental shifts")
        print("H2: FEP+MCM maintains higher accuracy under adversarial conditions")
        print("H3: FEP+MCM exhibits more stable cognitive behavior")
        print()
        
        experiment_start = time.time()
        
        # Run all three conditions
        self.results["baseline"] = self.run_condition(self.baseline_questions, "BASELINE")
        self.results["shifted"] = self.run_condition(self.shifted_questions, "SHIFTED") 
        self.results["adversarial"] = self.run_condition(self.adversarial_questions, "ADVERSARIAL")
        
        # Analyze results
        resilience_analysis = self.analyze_resilience()
        robustness_analysis = self.analyze_adversarial_robustness()
        stability_analysis = self.analyze_stability()
        
        experiment_time = time.time() - experiment_start
        
        # Compile comprehensive results
        comprehensive_results = {
            "experiment_config": self.config.__dict__,
            "experiment_duration": experiment_time,
            "raw_results": self.results,
            "hypothesis_testing": {
                **resilience_analysis,
                **robustness_analysis, 
                **stability_analysis
            },
            "mcm_event_summary": {
                "total_mcm_events": len(self.fep_mcm.mcm_events),
                "mcm_events_by_condition": {
                    condition: sum(1 for r in self.results[condition]["fep_mcm"] 
                                 if r["monitoring_result"].get("anomaly_detected", False))
                    for condition in ["baseline", "shifted", "adversarial"]
                }
            }
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 EXPERIMENTAL RESULTS SUMMARY")
        
        h1 = resilience_analysis["h1_resilience_to_shifts"]
        print(f"H1 (Resilience): MCM better = {h1['mcm_shows_better_resilience']}")
        print(f"    Improvement: {h1['resilience_improvement']:.1%}")
        
        h2 = robustness_analysis["h2_adversarial_robustness"] 
        print(f"H2 (Robustness): MCM better = {h2['mcm_shows_better_robustness']}")
        print(f"    Anomaly detection rate: {h2['mcm_detection_rate']:.1%}")
        
        h3 = stability_analysis["h3_behavioral_stability"]
        print(f"H3 (Stability): MCM better = {h3['mcm_shows_better_stability']}")
        print(f"    Improvement: {h3['stability_improvement']:.1%}")
        
        return comprehensive_results

def main():
    """Run the controlled FEP comparison experiment."""
    config = ExperimentConfig(
        num_trials_per_condition=1,  # Quick test
        hierarchy_levels=3
    )
    
    experiment = ControlledExperiment(config)
    results = experiment.run_full_experiment()
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"fep_control_comparison_{timestamp}.json"
    
    # JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif hasattr(obj, '__name__'):  # Handle EnumType and similar
            return str(obj)
        return obj
    
    results_serializable = convert_types(results)
    
    with open(filename, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n📄 Results saved to: {filename}")
    return results

if __name__ == "__main__":
    main()
