#!/usr/bin/env python3
"""
Koopman Operator-Based Meta-Cognitive Monitor
============================================

Implements true Koopman operator analysis for drift detection and meta-cognitive monitoring,
replacing the variance-based heuristic detector with principled dynamical systems analysis.

This implementation uses Dynamic Mode Decomposition (DMD) to approximate Koopman operators
and track long-term eigenfunctions for system state monitoring.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.linalg import svd
from collections import deque

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System operational states based on Koopman eigenfunction analysis."""
    STABLE = "stable"
    ADAPTING = "adapting"
    CRITICAL = "critical"
    RECOVERING = "recovering"


@dataclass
class KoopmanConfig:
    """Configuration for Koopman operator analysis."""
    # DMD parameters
    window_size: int = 50  # Number of snapshots for DMD
    rank_truncation: int = 10  # SVD rank for DMD
    eigenvalue_threshold: float = 0.95  # Stability threshold for eigenvalues
    
    # State detection parameters
    drift_threshold: float = 0.1  # Threshold for drift detection
    critical_eigenvalue_threshold: float = 1.1  # Threshold for critical state
    recovery_window: int = 10  # Window for recovery detection
    
    # System parameters
    observation_dim: int = 10  # Dimension of system observations
    update_frequency: int = 5  # How often to recompute Koopman operator


class DynamicModeDecomposition:
    """
    Dynamic Mode Decomposition for Koopman operator approximation.
    
    Implements the standard DMD algorithm to extract dynamic modes and eigenvalues
    from time-series data, providing a finite-dimensional approximation of the
    infinite-dimensional Koopman operator.
    """
    
    def __init__(self, rank: int = 10):
        self.rank = rank
        self.modes = None
        self.eigenvalues = None
        self.amplitudes = None
        
    def fit(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
        """
        Fit DMD to snapshot pairs (X, Y) where Y = F(X) for some dynamics F.
        
        Args:
            X: Snapshot matrix at time t (n_features, n_snapshots)
            Y: Snapshot matrix at time t+1 (n_features, n_snapshots)
            
        Returns:
            Dictionary containing DMD results and diagnostics
        """
        # SVD of snapshot matrix X
        U, s, Vt = svd(X, full_matrices=False)
        
        # Rank truncation
        r = min(self.rank, len(s))
        U_r = U[:, :r]
        s_r = s[:r]
        V_r = Vt[:r, :].T
        
        # Build reduced-order linear operator
        S_inv = np.diag(1.0 / s_r)
        A_tilde = U_r.T @ Y @ V_r @ S_inv
        
        # Eigendecomposition of reduced operator
        eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
        
        # DMD modes (full-order eigenvectors)
        modes = Y @ V_r @ S_inv @ eigenvectors
        
        # Compute amplitudes
        amplitudes = np.linalg.pinv(modes) @ X[:, 0]
        
        # Store results
        self.modes = modes
        self.eigenvalues = eigenvalues
        self.amplitudes = amplitudes
        
        # Compute diagnostics
        residual_norm = np.linalg.norm(Y - self.predict(X))
        spectral_radius = np.max(np.abs(eigenvalues))
        
        return {
            'eigenvalues': eigenvalues,
            'modes': modes,
            'amplitudes': amplitudes,
            'residual_norm': residual_norm,
            'spectral_radius': spectral_radius,
            'rank_used': r
        }
    
    def predict(self, X0: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Predict future states using DMD model.
        
        Args:
            X0: Initial state (n_features, n_initial_conditions)
            steps: Number of time steps to predict
            
        Returns:
            Predicted states (n_features, n_initial_conditions)
        """
        if self.modes is None:
            raise ValueError("DMD must be fitted before prediction")
        
        # Compute time evolution
        time_evolution = np.power(self.eigenvalues, steps)
        
        # Predict using DMD reconstruction
        prediction = self.modes @ np.diag(time_evolution * self.amplitudes)
        
        return prediction


class KoopmanEigenfunctionTracker:
    """
    Tracks evolution of Koopman eigenfunctions for drift detection.
    
    Monitors how eigenfunctions evolve over time to detect changes in
    system dynamics that indicate drift or anomalous behavior.
    """
    
    def __init__(self, config: KoopmanConfig):
        self.config = config
        self.eigenfunction_history = deque(maxlen=config.window_size)
        self.baseline_eigenfunctions = None
        self.drift_scores = deque(maxlen=config.recovery_window)
        
    def update(self, eigenfunctions: np.ndarray) -> Dict[str, float]:
        """
        Update eigenfunction tracker with new eigenfunctions.
        
        Args:
            eigenfunctions: Current eigenfunctions from DMD analysis
            
        Returns:
            Dictionary containing drift metrics
        """
        self.eigenfunction_history.append(eigenfunctions.copy())
        
        # Initialize baseline if not set
        if self.baseline_eigenfunctions is None:
            if len(self.eigenfunction_history) >= 10:
                self.baseline_eigenfunctions = np.mean(
                    list(self.eigenfunction_history)[:10], axis=0
                )
        
        # Compute drift metrics
        drift_metrics = self._compute_drift_metrics(eigenfunctions)
        self.drift_scores.append(drift_metrics['drift_score'])
        
        return drift_metrics
    
    def _compute_drift_metrics(self, current_eigenfunctions: np.ndarray) -> Dict[str, float]:
        """Compute various drift detection metrics."""
        if self.baseline_eigenfunctions is None:
            return {'drift_score': 0.0, 'eigenfunction_deviation': 0.0}
        
        # Eigenfunction deviation from baseline
        deviation = np.linalg.norm(
            current_eigenfunctions - self.baseline_eigenfunctions
        )
        
        # Drift score based on recent history
        if len(self.eigenfunction_history) >= 5:
            recent_mean = np.mean(list(self.eigenfunction_history)[-5:], axis=0)
            drift_score = np.linalg.norm(recent_mean - self.baseline_eigenfunctions)
        else:
            drift_score = deviation
        
        return {
            'drift_score': drift_score,
            'eigenfunction_deviation': deviation,
            'baseline_distance': np.linalg.norm(self.baseline_eigenfunctions)
        }
    
    def detect_drift(self) -> bool:
        """Detect if system has drifted based on eigenfunction evolution."""
        if len(self.drift_scores) < 3:
            return False
        
        recent_drift = np.mean(list(self.drift_scores)[-3:])
        return recent_drift > self.config.drift_threshold


class KoopmanMetaCognitiveMonitor:
    """
    Meta-Cognitive Monitor based on Koopman operator theory.
    
    Replaces the variance-based heuristic detector with principled
    dynamical systems analysis using Koopman operators approximated
    via Dynamic Mode Decomposition.
    """
    
    def __init__(self, config: KoopmanConfig = None):
        self.config = config or KoopmanConfig()
        self.dmd = DynamicModeDecomposition(rank=self.config.rank_truncation)
        self.eigenfunction_tracker = KoopmanEigenfunctionTracker(self.config)
        
        # System state management
        self.current_state = SystemState.STABLE
        self.state_history = deque(maxlen=100)
        
        # Data collection for DMD
        self.snapshot_buffer = deque(maxlen=self.config.window_size + 1)
        self.update_counter = 0
        
        # Analysis results
        self.current_eigenvalues = None
        self.current_modes = None
        self.spectral_radius = 1.0
        self.drift_detected = False
        
        logger.info(f"Initialized Koopman MCM with window_size={self.config.window_size}")
    
    def process_observation(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Process new observation through Koopman analysis.
        
        Args:
            observation: Current system observation vector
            
        Returns:
            Dictionary containing monitoring results and system state
        """
        # Add observation to buffer
        self.snapshot_buffer.append(observation.flatten())
        self.update_counter += 1
        
        # Perform DMD analysis if we have enough data
        if len(self.snapshot_buffer) >= self.config.window_size:
            if self.update_counter % self.config.update_frequency == 0:
                self._update_koopman_analysis()
        
        # Determine system state
        previous_state = self.current_state
        self.current_state = self._determine_system_state()
        
        # Log state transitions
        if self.current_state != previous_state:
            logger.info(f"MCM state transition: {previous_state.value} → {self.current_state.value}")
        
        self.state_history.append(self.current_state)
        
        return {
            'system_state': self.current_state.value,
            'spectral_radius': self.spectral_radius,
            'drift_detected': self.drift_detected,
            'eigenvalues': self.current_eigenvalues.tolist() if self.current_eigenvalues is not None else None,
            'state_transition': previous_state.value != self.current_state.value,
            'monitoring_active': True,
            'koopman_analysis_available': self.current_eigenvalues is not None
        }
    
    def _update_koopman_analysis(self):
        """Update Koopman operator approximation using DMD."""
        if len(self.snapshot_buffer) < 2:
            return
        
        # Prepare snapshot matrices
        snapshots = np.array(list(self.snapshot_buffer))
        X = snapshots[:-1].T  # States at time t
        Y = snapshots[1:].T   # States at time t+1
        
        try:
            # Fit DMD
            dmd_results = self.dmd.fit(X, Y)
            
            # Update analysis results
            self.current_eigenvalues = dmd_results['eigenvalues']
            self.current_modes = dmd_results['modes']
            self.spectral_radius = dmd_results['spectral_radius']
            
            # Update eigenfunction tracking
            if self.current_modes is not None:
                # Use first few modes as eigenfunctions
                eigenfunctions = np.abs(self.current_modes[:, :5]).flatten()
                drift_metrics = self.eigenfunction_tracker.update(eigenfunctions)
                self.drift_detected = self.eigenfunction_tracker.detect_drift()
            
            logger.debug(f"DMD update: spectral_radius={self.spectral_radius:.3f}, "
                        f"drift_detected={self.drift_detected}")
            
        except Exception as e:
            logger.warning(f"DMD analysis failed: {e}")
            # Fall back to stable state if analysis fails
            self.current_state = SystemState.STABLE
    
    def _determine_system_state(self) -> SystemState:
        """
        Determine system state based on Koopman analysis.
        
        Uses eigenvalue analysis and drift detection to classify
        the current system state according to dynamical systems theory.
        """
        # Default to stable if no analysis available
        if self.current_eigenvalues is None:
            return SystemState.STABLE
        
        # Check for critical state (unstable eigenvalues)
        max_eigenvalue_magnitude = np.max(np.abs(self.current_eigenvalues))
        if max_eigenvalue_magnitude > self.config.critical_eigenvalue_threshold:
            return SystemState.CRITICAL
        
        # Check for drift
        if self.drift_detected:
            # If we were critical and drift is decreasing, we're recovering
            if (len(self.state_history) > 0 and 
                self.state_history[-1] == SystemState.CRITICAL and
                len(self.eigenfunction_tracker.drift_scores) >= 3):
                
                recent_drift = list(self.eigenfunction_tracker.drift_scores)[-3:]
                if len(recent_drift) >= 2 and recent_drift[-1] < recent_drift[-2]:
                    return SystemState.RECOVERING
            
            return SystemState.ADAPTING
        
        # Check for recovery completion
        if (len(self.state_history) >= 2 and 
            self.state_history[-1] in [SystemState.CRITICAL, SystemState.RECOVERING] and
            max_eigenvalue_magnitude < self.config.eigenvalue_threshold):
            return SystemState.STABLE
        
        return SystemState.STABLE
    
    def trigger_controlled_exploration(self) -> Dict[str, Any]:
        """
        Trigger controlled exploration when system is in critical state.
        
        Returns:
            Dictionary containing exploration parameters and recommendations
        """
        if self.current_state != SystemState.CRITICAL:
            return {'exploration_triggered': False, 'reason': 'System not in critical state'}
        
        # Analyze which modes are unstable
        unstable_modes = []
        if self.current_eigenvalues is not None:
            for i, eigenval in enumerate(self.current_eigenvalues):
                if np.abs(eigenval) > self.config.critical_eigenvalue_threshold:
                    unstable_modes.append(i)
        
        exploration_params = {
            'exploration_triggered': True,
            'unstable_modes': unstable_modes,
            'spectral_radius': self.spectral_radius,
            'recommended_action': 'increase_exploration_variance',
            'focus_dimensions': unstable_modes[:3] if unstable_modes else [0, 1, 2]
        }
        
        logger.info(f"Controlled exploration triggered: {len(unstable_modes)} unstable modes detected")
        
        return exploration_params
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics."""
        return {
            'current_state': self.current_state.value,
            'spectral_radius': self.spectral_radius,
            'drift_detected': self.drift_detected,
            'eigenvalues': self.current_eigenvalues.tolist() if self.current_eigenvalues is not None else None,
            'buffer_size': len(self.snapshot_buffer),
            'analysis_updates': self.update_counter // self.config.update_frequency,
            'state_history': [s.value for s in list(self.state_history)[-10:]],
            'drift_scores': list(self.eigenfunction_tracker.drift_scores) if self.eigenfunction_tracker.drift_scores else [],
            'configuration': {
                'window_size': self.config.window_size,
                'rank_truncation': self.config.rank_truncation,
                'eigenvalue_threshold': self.config.eigenvalue_threshold,
                'drift_threshold': self.config.drift_threshold
            }
        }


def main():
    """Test the Koopman MCM implementation."""
    print("Testing Koopman Meta-Cognitive Monitor")
    print("=" * 50)
    
    # Initialize MCM
    config = KoopmanConfig(
        window_size=20,
        rank_truncation=5,
        observation_dim=5
    )
    mcm = KoopmanMetaCognitiveMonitor(config)
    
    # Simulate system observations
    print("\nSimulating system observations...")
    
    # Stable phase
    for i in range(15):
        obs = np.random.normal(0, 0.1, 5)  # Stable observations
        result = mcm.process_observation(obs)
        if i % 5 == 0:
            print(f"Step {i:2d}: State={result['system_state']}, "
                  f"Spectral Radius={result['spectral_radius']:.3f}")
    
    # Introduce instability
    print("\nIntroducing system instability...")
    for i in range(15, 30):
        obs = np.random.normal(0, 1.0, 5) + np.sin(i * 0.5) * 2  # Unstable observations
        result = mcm.process_observation(obs)
        if i % 5 == 0:
            print(f"Step {i:2d}: State={result['system_state']}, "
                  f"Spectral Radius={result['spectral_radius']:.3f}, "
                  f"Drift={result['drift_detected']}")
        
        # Trigger exploration if critical
        if result['system_state'] == 'critical':
            exploration = mcm.trigger_controlled_exploration()
            if exploration['exploration_triggered']:
                print(f"  → Controlled exploration triggered")
    
    # Return to stability
    print("\nReturning to stability...")
    for i in range(30, 45):
        obs = np.random.normal(0, 0.1, 5)  # Return to stable
        result = mcm.process_observation(obs)
        if i % 5 == 0:
            print(f"Step {i:2d}: State={result['system_state']}, "
                  f"Spectral Radius={result['spectral_radius']:.3f}")
    
    # Final diagnostics
    print("\nFinal System Diagnostics:")
    diagnostics = mcm.get_system_diagnostics()
    print(f"Final State: {diagnostics['current_state']}")
    print(f"Spectral Radius: {diagnostics['spectral_radius']:.3f}")
    print(f"Drift Detected: {diagnostics['drift_detected']}")
    print(f"Analysis Updates: {diagnostics['analysis_updates']}")
    print(f"Recent States: {diagnostics['state_history']}")


if __name__ == "__main__":
    main()
