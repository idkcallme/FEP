#!/usr/bin/env python3
"""
Timescale-Separated Free Energy Principle Implementation
======================================================

Implements proper timescale separation with epsilon parameter control,
stability controller, and asymptotic stability analysis as required
by the theoretical framework.

This addresses the peer review requirement to implement the claimed
three-layer architecture with fast inference, slow learning, and
stability control operating on different timescales.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class TimescaleConfig:
    """Configuration for timescale-separated FEP system."""
    # Timescale parameters
    epsilon: float = 0.01  # Timescale separation parameter (slow/fast ratio)
    fast_inference_rate: float = 1.0  # Fast inference timescale
    slow_learning_rate: float = None  # Computed as epsilon * fast_rate
    
    # Stability control
    stability_check_frequency: int = 10  # How often to check stability
    hessian_regularization: float = 1e-6  # Regularization for Hessian computation
    max_eigenvalue_threshold: float = 1.0  # Stability threshold
    
    # System parameters
    state_dim: int = 10
    action_dim: int = 5
    hierarchy_levels: int = 3
    
    def __post_init__(self):
        if self.slow_learning_rate is None:
            self.slow_learning_rate = self.epsilon * self.fast_inference_rate


class StabilityController:
    """
    Stability controller implementing singular perturbation theory principles.
    
    Monitors system stability through Hessian analysis and adjusts learning
    rates to maintain asymptotic stability according to the theoretical
    framework described in the report.
    """
    
    def __init__(self, config: TimescaleConfig):
        self.config = config
        self.stability_history = deque(maxlen=50)
        self.eigenvalue_history = deque(maxlen=20)
        self.adjustment_history = deque(maxlen=10)
        
        # Stability state
        self.is_stable = True
        self.last_hessian = None
        self.last_eigenvalues = None
        
        logger.info(f"Initialized stability controller with ε={config.epsilon}")
    
    def check_stability(self, free_energy_fn, parameters: torch.Tensor) -> Dict[str, Any]:
        """
        Check system stability using Hessian analysis.
        
        Computes the Hessian of the free energy functional and analyzes
        eigenvalues to determine stability according to singular perturbation theory.
        
        Args:
            free_energy_fn: Function that computes free energy given parameters
            parameters: Current system parameters
            
        Returns:
            Dictionary containing stability analysis results
        """
        try:
            # Compute Hessian of free energy functional
            hessian = self._compute_hessian(free_energy_fn, parameters)
            
            # Eigenvalue analysis
            eigenvalues = torch.linalg.eigvals(hessian).real
            max_eigenvalue = torch.max(eigenvalues).item()
            min_eigenvalue = torch.min(eigenvalues).item()
            
            # Positive definiteness check (required for stability)
            is_positive_definite = torch.all(eigenvalues > self.config.hessian_regularization)
            
            # Stability assessment
            is_stable = (is_positive_definite and 
                        max_eigenvalue < self.config.max_eigenvalue_threshold)
            
            # Store results
            self.last_hessian = hessian
            self.last_eigenvalues = eigenvalues
            self.is_stable = is_stable
            
            stability_result = {
                'is_stable': is_stable,
                'is_positive_definite': is_positive_definite.item(),
                'max_eigenvalue': max_eigenvalue,
                'min_eigenvalue': min_eigenvalue,
                'eigenvalues': eigenvalues.tolist(),
                'condition_number': max_eigenvalue / max(min_eigenvalue, 1e-8),
                'hessian_trace': torch.trace(hessian).item(),
                'hessian_determinant': torch.det(hessian).item()
            }
            
            self.stability_history.append(stability_result)
            self.eigenvalue_history.append(eigenvalues.clone())
            
            return stability_result
            
        except Exception as e:
            logger.warning(f"Stability check failed: {e}")
            return {
                'is_stable': False,
                'error': str(e),
                'max_eigenvalue': float('inf'),
                'min_eigenvalue': float('-inf')
            }
    
    def _compute_hessian(self, free_energy_fn, parameters: torch.Tensor) -> torch.Tensor:
        """
        Compute Hessian matrix of free energy functional.
        
        Uses automatic differentiation to compute the second-order derivatives
        of the free energy with respect to system parameters.
        """
        # Ensure parameters require gradients
        if not parameters.requires_grad:
            parameters = parameters.detach().requires_grad_(True)
        
        # Compute free energy
        fe = free_energy_fn(parameters)
        
        # Compute gradient
        grad = torch.autograd.grad(fe, parameters, create_graph=True)[0]
        
        # Compute Hessian
        hessian = torch.zeros(len(grad), len(grad))
        for i in range(len(grad)):
            grad2 = torch.autograd.grad(grad[i], parameters, retain_graph=True)[0]
            hessian[i] = grad2
        
        # Add regularization for numerical stability
        hessian += self.config.hessian_regularization * torch.eye(len(grad))
        
        return hessian
    
    def adjust_learning_rates(self, current_rates: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust learning rates based on stability analysis.
        
        Implements adaptive learning rate control to maintain stability
        according to singular perturbation theory principles.
        
        Args:
            current_rates: Dictionary of current learning rates
            
        Returns:
            Dictionary of adjusted learning rates
        """
        if not self.stability_history:
            return current_rates
        
        latest_stability = self.stability_history[-1]
        adjusted_rates = current_rates.copy()
        
        # Adjustment strategy based on stability
        if not latest_stability['is_stable']:
            # Reduce learning rates if unstable
            adjustment_factor = 0.5
            logger.warning("System unstable - reducing learning rates")
            
        elif latest_stability['max_eigenvalue'] > 0.8 * self.config.max_eigenvalue_threshold:
            # Cautious reduction if approaching instability
            adjustment_factor = 0.8
            logger.info("Approaching instability - cautious learning rate reduction")
            
        else:
            # Gradual increase if stable
            adjustment_factor = 1.05
            
        # Apply adjustments while maintaining timescale separation
        for key in adjusted_rates:
            adjusted_rates[key] *= adjustment_factor
            
        # Ensure timescale separation is maintained
        if 'fast_inference' in adjusted_rates and 'slow_learning' in adjusted_rates:
            adjusted_rates['slow_learning'] = self.config.epsilon * adjusted_rates['fast_inference']
        
        self.adjustment_history.append({
            'adjustment_factor': adjustment_factor,
            'stability': latest_stability['is_stable'],
            'max_eigenvalue': latest_stability['max_eigenvalue']
        })
        
        return adjusted_rates
    
    def get_stability_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive stability diagnostics."""
        if not self.stability_history:
            return {'status': 'no_data'}
        
        recent_stability = list(self.stability_history)[-5:]
        stability_rate = sum(s['is_stable'] for s in recent_stability) / len(recent_stability)
        
        return {
            'current_stability': self.is_stable,
            'stability_rate_recent': stability_rate,
            'last_max_eigenvalue': recent_stability[-1]['max_eigenvalue'] if recent_stability else None,
            'eigenvalue_trend': self._compute_eigenvalue_trend(),
            'adjustment_history': list(self.adjustment_history),
            'total_stability_checks': len(self.stability_history)
        }
    
    def _compute_eigenvalue_trend(self) -> str:
        """Compute trend in maximum eigenvalues."""
        if len(self.eigenvalue_history) < 3:
            return 'insufficient_data'
        
        recent_max_eigenvals = [torch.max(eigs).item() for eigs in list(self.eigenvalue_history)[-3:]]
        
        if recent_max_eigenvals[-1] > recent_max_eigenvals[0] * 1.1:
            return 'increasing'
        elif recent_max_eigenvals[-1] < recent_max_eigenvals[0] * 0.9:
            return 'decreasing'
        else:
            return 'stable'


class TimescaleSeparatedFEP:
    """
    Free Energy Principle implementation with proper timescale separation.
    
    Implements the three-layer architecture described in the theoretical framework:
    1. Fast inference (perception and belief updating)
    2. Slow learning (parameter updates)
    3. Stability control (learning rate adaptation)
    
    Uses epsilon parameter to control the ratio between fast and slow processes.
    """
    
    def __init__(self, config: TimescaleConfig = None):
        self.config = config or TimescaleConfig()
        
        # Initialize stability controller
        self.stability_controller = StabilityController(self.config)
        
        # System components
        self.generative_model = self._build_generative_model()
        self.recognition_model = self._build_recognition_model()
        
        # Timescale-specific optimizers
        self.fast_optimizer = torch.optim.Adam(
            self.recognition_model.parameters(), 
            lr=self.config.fast_inference_rate
        )
        self.slow_optimizer = torch.optim.Adam(
            self.generative_model.parameters(),
            lr=self.config.slow_learning_rate
        )
        
        # System state
        self.inference_step = 0
        self.learning_step = 0
        self.stability_check_counter = 0
        
        # Learning rates tracking
        self.learning_rates = {
            'fast_inference': self.config.fast_inference_rate,
            'slow_learning': self.config.slow_learning_rate
        }
        
        logger.info(f"Initialized timescale-separated FEP with ε={self.config.epsilon}")
    
    def _build_generative_model(self) -> nn.Module:
        """Build hierarchical generative model."""
        layers = []
        current_dim = self.config.state_dim
        
        for level in range(self.config.hierarchy_levels):
            next_dim = max(current_dim // 2, self.config.action_dim)
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = next_dim
        
        return nn.Sequential(*layers)
    
    def _build_recognition_model(self) -> nn.Module:
        """Build recognition (encoder) model."""
        return nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.state_dim * 2),
            nn.ReLU(),
            nn.Linear(self.config.state_dim * 2, self.config.state_dim),
            nn.ReLU(),
            nn.Linear(self.config.state_dim, self.config.action_dim * 2)  # Mean and log-var
        )
    
    def compute_free_energy(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute variational free energy with proper mathematical formulation.
        
        F = E_q[log q(z|x) - log p(x,z)] = KL[q(z|x) || p(z)] + E_q[-log p(x|z)]
        """
        batch_size = observations.shape[0]
        
        # Recognition model: q(z|x)
        recognition_output = self.recognition_model(observations)
        z_mean = recognition_output[:, :self.config.action_dim]
        z_logvar = recognition_output[:, self.config.action_dim:]
        
        # Sample from recognition distribution
        z_std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(z_std)
        z_samples = z_mean + eps * z_std
        
        # Generative model: p(x|z)
        reconstructions = self.generative_model(z_samples)
        
        # Compute free energy components
        reconstruction_error = nn.MSELoss(reduction='sum')(reconstructions, observations)
        
        # KL divergence: KL[q(z|x) || p(z)] where p(z) = N(0,I)
        kl_divergence = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        
        # Total free energy
        free_energy = reconstruction_error + kl_divergence
        
        return {
            'total_free_energy': free_energy,
            'reconstruction_error': reconstruction_error,
            'kl_divergence': kl_divergence,
            'z_samples': z_samples,
            'z_mean': z_mean,
            'z_logvar': z_logvar,
            'reconstructions': reconstructions
        }
    
    def fast_inference_step(self, observations: torch.Tensor) -> Dict[str, Any]:
        """
        Perform fast inference step (belief updating).
        
        Updates recognition model parameters on the fast timescale
        to minimize free energy through belief updating.
        """
        self.inference_step += 1
        
        # Compute free energy
        fe_components = self.compute_free_energy(observations)
        free_energy = fe_components['total_free_energy']
        
        # Fast inference: update recognition model only
        self.fast_optimizer.zero_grad()
        free_energy.backward(retain_graph=True)
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.recognition_model.parameters(), 1.0)
        
        self.fast_optimizer.step()
        
        return {
            'inference_step': self.inference_step,
            'free_energy': free_energy.item(),
            'reconstruction_error': fe_components['reconstruction_error'].item(),
            'kl_divergence': fe_components['kl_divergence'].item(),
            'timescale': 'fast',
            'parameters_updated': 'recognition_model'
        }
    
    def slow_learning_step(self, observations: torch.Tensor) -> Dict[str, Any]:
        """
        Perform slow learning step (generative model updates).
        
        Updates generative model parameters on the slow timescale
        according to the epsilon timescale separation.
        """
        self.learning_step += 1
        
        # Only update every 1/epsilon steps to maintain timescale separation
        if self.inference_step % max(1, int(1/self.config.epsilon)) != 0:
            return {'skipped': True, 'reason': 'timescale_separation'}
        
        # Compute free energy
        fe_components = self.compute_free_energy(observations)
        free_energy = fe_components['total_free_energy']
        
        # Slow learning: update generative model only
        self.slow_optimizer.zero_grad()
        free_energy.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.generative_model.parameters(), 1.0)
        
        self.slow_optimizer.step()
        
        return {
            'learning_step': self.learning_step,
            'free_energy': free_energy.item(),
            'reconstruction_error': fe_components['reconstruction_error'].item(),
            'kl_divergence': fe_components['kl_divergence'].item(),
            'timescale': 'slow',
            'parameters_updated': 'generative_model'
        }
    
    def stability_control_step(self, observations: torch.Tensor) -> Dict[str, Any]:
        """
        Perform stability control step.
        
        Checks system stability and adjusts learning rates according
        to singular perturbation theory principles.
        """
        self.stability_check_counter += 1
        
        # Only check stability periodically
        if self.stability_check_counter % self.config.stability_check_frequency != 0:
            return {'skipped': True, 'reason': 'frequency_control'}
        
        # Create free energy function for stability analysis
        def free_energy_fn(params):
            # Temporarily set parameters
            param_idx = 0
            for param in self.generative_model.parameters():
                param_size = param.numel()
                param.data = params[param_idx:param_idx + param_size].view(param.shape)
                param_idx += param_size
            
            # Compute free energy
            fe_components = self.compute_free_energy(observations)
            return fe_components['total_free_energy']
        
        # Get current parameters
        current_params = torch.cat([p.flatten() for p in self.generative_model.parameters()])
        
        # Stability analysis
        stability_result = self.stability_controller.check_stability(free_energy_fn, current_params)
        
        # Adjust learning rates if needed
        adjusted_rates = self.stability_controller.adjust_learning_rates(self.learning_rates)
        
        # Update optimizers with new learning rates
        if adjusted_rates != self.learning_rates:
            for param_group in self.fast_optimizer.param_groups:
                param_group['lr'] = adjusted_rates['fast_inference']
            for param_group in self.slow_optimizer.param_groups:
                param_group['lr'] = adjusted_rates['slow_learning']
            
            self.learning_rates = adjusted_rates
        
        return {
            'stability_check': self.stability_check_counter,
            'stability_result': stability_result,
            'learning_rates_adjusted': adjusted_rates != self.learning_rates,
            'current_learning_rates': self.learning_rates,
            'timescale': 'stability_control'
        }
    
    def perception_action_cycle(self, observations: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Complete perception-action cycle with proper timescale separation.
        
        Integrates fast inference, slow learning, and stability control
        according to the theoretical framework.
        """
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        
        # Fast inference (always performed)
        fast_result = self.fast_inference_step(observations)
        
        # Slow learning (performed according to timescale separation)
        slow_result = self.slow_learning_step(observations)
        
        # Stability control (performed periodically)
        stability_result = self.stability_control_step(observations)
        
        # Compute action from current belief state
        fe_components = self.compute_free_energy(observations)
        action = fe_components['z_mean'][0]  # Use posterior mean as action
        
        # Aggregate results
        cycle_result = {
            'fast_inference': fast_result,
            'slow_learning': slow_result,
            'stability_control': stability_result,
            'free_energy': fast_result['free_energy'],
            'learning_rates': self.learning_rates,
            'timescale_separation_epsilon': self.config.epsilon,
            'system_stable': stability_result.get('stability_result', {}).get('is_stable', True)
        }
        
        return action, cycle_result
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics."""
        stability_diagnostics = self.stability_controller.get_stability_diagnostics()
        
        return {
            'timescale_configuration': {
                'epsilon': self.config.epsilon,
                'fast_inference_rate': self.config.fast_inference_rate,
                'slow_learning_rate': self.config.slow_learning_rate,
                'timescale_ratio': self.config.fast_inference_rate / self.config.slow_learning_rate
            },
            'step_counters': {
                'inference_steps': self.inference_step,
                'learning_steps': self.learning_step,
                'stability_checks': self.stability_check_counter
            },
            'current_learning_rates': self.learning_rates,
            'stability_diagnostics': stability_diagnostics,
            'model_parameters': {
                'generative_model_params': sum(p.numel() for p in self.generative_model.parameters()),
                'recognition_model_params': sum(p.numel() for p in self.recognition_model.parameters())
            }
        }


def main():
    """Test the timescale-separated FEP implementation."""
    print("Testing Timescale-Separated FEP Implementation")
    print("=" * 50)
    
    # Initialize system
    config = TimescaleConfig(
        epsilon=0.05,  # 20:1 fast:slow ratio
        state_dim=8,
        action_dim=4,
        hierarchy_levels=2
    )
    
    fep_system = TimescaleSeparatedFEP(config)
    
    print(f"Initialized system with ε={config.epsilon}")
    print(f"Fast inference rate: {config.fast_inference_rate}")
    print(f"Slow learning rate: {config.slow_learning_rate}")
    print(f"Timescale ratio: {config.fast_inference_rate/config.slow_learning_rate:.1f}:1")
    
    # Test perception-action cycles
    print("\nRunning perception-action cycles...")
    
    for step in range(30):
        # Generate test observation
        obs = torch.randn(config.state_dim) * (0.5 + step * 0.02)  # Gradually increasing variance
        
        # Run perception-action cycle
        action, results = fep_system.perception_action_cycle(obs)
        
        # Print results every 5 steps
        if step % 5 == 0:
            fast_fe = results['fast_inference']['free_energy']
            stable = results.get('system_stable', True)
            lr_fast = results['learning_rates']['fast_inference']
            lr_slow = results['learning_rates']['slow_learning']
            
            print(f"Step {step:2d}: FE={fast_fe:7.3f}, Stable={stable}, "
                  f"LR_fast={lr_fast:.4f}, LR_slow={lr_slow:.4f}")
            
            # Show stability control actions
            stability_result = results['stability_control']
            if not stability_result.get('skipped', False):
                stability_info = stability_result.get('stability_result', {})
                if 'max_eigenvalue' in stability_info:
                    max_eig = stability_info['max_eigenvalue']
                    print(f"        Stability check: max_eigenvalue={max_eig:.3f}")
    
    # Final diagnostics
    print("\nFinal System Diagnostics:")
    diagnostics = fep_system.get_system_diagnostics()
    
    print("Timescale Configuration:")
    tc = diagnostics['timescale_configuration']
    print(f"  ε = {tc['epsilon']}")
    print(f"  Fast rate = {tc['fast_inference_rate']}")
    print(f"  Slow rate = {tc['slow_learning_rate']}")
    print(f"  Ratio = {tc['timescale_ratio']:.1f}:1")
    
    print("Step Counters:")
    sc = diagnostics['step_counters']
    print(f"  Inference steps: {sc['inference_steps']}")
    print(f"  Learning steps: {sc['learning_steps']}")
    print(f"  Stability checks: {sc['stability_checks']}")
    
    print("Stability Diagnostics:")
    sd = diagnostics['stability_diagnostics']
    if sd.get('status') != 'no_data':
        print(f"  Current stability: {sd.get('current_stability', 'Unknown')}")
        print(f"  Recent stability rate: {sd.get('stability_rate_recent', 0):.2f}")
        print(f"  Eigenvalue trend: {sd.get('eigenvalue_trend', 'Unknown')}")


if __name__ == "__main__":
    main()
