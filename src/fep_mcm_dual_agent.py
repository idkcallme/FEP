#!/usr/bin/env python3
"""
🧠 FEP-MCM DUAL AGENT SYSTEM - CORE ARCHITECTURE
===============================================
The missing central component that integrates FEP agent with MCM monitoring.

This implements the actual dual-agent architecture that was referenced but missing.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import time
from collections import deque

# Import our existing components
try:
    from .fep_mathematics import HierarchicalFEPSystem, create_fep_system
    from .active_inference import ActiveInferenceAgent, create_active_inference_agent
    from .predictive_coding import PredictiveCodingHierarchy, create_predictive_coding_system
except ImportError:
    # Fallback for direct execution
    from fep_mathematics import HierarchicalFEPSystem, create_fep_system
    from active_inference import ActiveInferenceAgent, create_active_inference_agent
    from predictive_coding import PredictiveCodingHierarchy, create_predictive_coding_system

logger = logging.getLogger(__name__)

@dataclass
class DualAgentConfig:
    """Configuration for the dual-agent system."""
    # FEP Agent parameters
    observation_dim: int = 100
    action_dim: int = 20
    latent_dim: int = 64
    hierarchy_levels: int = 3
    
    # MCM parameters
    vfe_buffer_size: int = 100
    chaos_threshold: float = 2.0
    surprise_threshold: float = 1.5
    monitoring_frequency: int = 1
    
    # Integration parameters
    learning_rate: float = 0.001
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

class MetaCognitiveMonitor:
    """
    Meta-Cognitive Monitor (MCM) that watches the FEP agent's internal states.
    This is the "monitor" part of the dual-agent system.
    """
    
    def __init__(self, config: DualAgentConfig):
        self.config = config
        self.vfe_history = deque(maxlen=config.vfe_buffer_size)
        self.surprise_history = deque(maxlen=config.vfe_buffer_size)
        self.chaos_detected = False
        self.monitoring_active = True
        
        # Chaos detection parameters
        self.chaos_threshold = config.chaos_threshold
        self.surprise_threshold = config.surprise_threshold
        
    def update(self, fep_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update monitor with current FEP agent state."""
        if not self.monitoring_active:
            return {"chaos_detected": False, "surprise_level": 0.0}
            
        # Extract key metrics from FEP state
        current_vfe = fep_state.get('free_energy', 0.0)
        current_surprise = fep_state.get('surprise', 0.0)
        
        # Convert tensors to scalars if needed
        if torch.is_tensor(current_vfe):
            current_vfe = current_vfe.item()
        if torch.is_tensor(current_surprise):
            current_surprise = current_surprise.item()
        
        # Update history
        self.vfe_history.append(current_vfe)
        self.surprise_history.append(current_surprise)
        
        # Detect chaos/anomalies
        chaos_detected = self._detect_chaos()
        surprise_level = self._compute_surprise_level()
        
        return {
            "chaos_detected": chaos_detected,
            "surprise_level": surprise_level,
            "vfe_mean": np.mean(self.vfe_history) if self.vfe_history else 0.0,
            "vfe_std": np.std(self.vfe_history) if len(self.vfe_history) > 1 else 0.0,
            "monitoring_active": self.monitoring_active
        }
    
    def _detect_chaos(self) -> bool:
        """Detect chaotic behavior in VFE dynamics."""
        if len(self.vfe_history) < 10:
            return False
            
        # Simple chaos detection: high variance in recent VFE
        recent_vfe = list(self.vfe_history)[-10:]
        vfe_variance = np.var(recent_vfe)
        
        # Also check for sudden spikes
        if len(recent_vfe) >= 2:
            last_change = abs(recent_vfe[-1] - recent_vfe[-2])
            spike_detected = last_change > self.chaos_threshold
        else:
            spike_detected = False
            
        return vfe_variance > self.chaos_threshold or spike_detected
    
    def _compute_surprise_level(self) -> float:
        """Compute current surprise level."""
        if not self.surprise_history:
            return 0.0
        return float(self.surprise_history[-1])

class FEPAgent:
    """
    Free Energy Principle Agent - the main cognitive component.
    This integrates our mathematical components into a coherent agent.
    """
    
    def __init__(self, config: DualAgentConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create integrated FEP system
        try:
            self.fep_system = create_fep_system(
                observation_dim=config.observation_dim,
                latent_dim=config.latent_dim,
                hierarchical=True
            )
            
            self.active_inference = create_active_inference_agent(
                observation_dim=config.observation_dim,
                action_dim=config.action_dim,
                latent_dim=config.latent_dim
            )
            
            self.predictive_coding, _ = create_predictive_coding_system(
                input_dim=config.observation_dim,
                hierarchy_levels=config.hierarchy_levels
            )
            
            self.components_available = True
            
        except Exception as e:
            logger.warning(f"Failed to initialize FEP components: {e}")
            self.components_available = False
            self._create_minimal_fallback()
    
    def _create_minimal_fallback(self):
        """Create minimal fallback implementation if full components fail."""
        self.observation_dim = self.config.observation_dim
        self.action_dim = self.config.action_dim
        self.latent_dim = self.config.latent_dim
        
        # Simple linear approximations for testing
        self.encoder = nn.Linear(self.observation_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, self.observation_dim)
        self.policy = nn.Linear(self.latent_dim, self.action_dim)
        
    def perceive_and_act(self, observation: torch.Tensor) -> Dict[str, Any]:
        """Main perception-action cycle."""
        if self.components_available:
            return self._full_perceive_and_act(observation)
        else:
            return self._fallback_perceive_and_act(observation)
    
    def _full_perceive_and_act(self, observation: torch.Tensor) -> Dict[str, Any]:
        """Full FEP-based perception and action."""
        try:
            # Hierarchical FEP processing
            fep_result = self.fep_system.process_observation(observation)
            
            # Active inference for action selection
            perception_result = self.active_inference.perceive(observation)
            action_result = self.active_inference.act()
            
            # Predictive coding for future states
            prediction_result = self.predictive_coding.process_sequence(
                observation.unsqueeze(0)  # Add sequence dimension
            )
            
            return {
                'action': action_result.get('action', torch.zeros(self.config.action_dim)),
                'free_energy': fep_result['free_energy'].mean(),
                'surprise': fep_result['surprise'].mean(),
                'beliefs': fep_result['posterior_mean'],
                'predictions': prediction_result['predictions'],
                'prediction_errors': prediction_result['errors'],
                'components_used': 'full'
            }
            
        except Exception as e:
            logger.warning(f"Full FEP processing failed: {e}")
            return self._fallback_perceive_and_act(observation)
    
    def _fallback_perceive_and_act(self, observation: torch.Tensor) -> Dict[str, Any]:
        """Simplified fallback processing."""
        # Simple encoding and decoding
        encoded = self.encoder(observation)
        decoded = self.decoder(encoded)
        action = self.policy(encoded)
        
        # Compute simple free energy approximation
        reconstruction_error = torch.nn.functional.mse_loss(decoded, observation)
        complexity = torch.norm(encoded)
        free_energy = reconstruction_error + 0.01 * complexity
        
        return {
            'action': action,
            'free_energy': free_energy,
            'surprise': reconstruction_error,
            'beliefs': encoded,
            'predictions': decoded.unsqueeze(0),
            'prediction_errors': (observation - decoded).unsqueeze(0),
            'components_used': 'fallback'
        }

class DualAgentSystem:
    """
    Complete Dual-Agent System: FEP Agent + Meta-Cognitive Monitor
    This is the main class that integrates everything.
    """
    
    def __init__(self, config: Optional[DualAgentConfig] = None):
        self.config = config or DualAgentConfig()
        
        # Initialize both agents
        self.fep_agent = FEPAgent(self.config)
        self.mcm_monitor = MetaCognitiveMonitor(self.config)
        
        # System state
        self.step_count = 0
        self.total_free_energy = 0.0
        self.performance_metrics = {
            'chaos_events': 0,
            'high_surprise_events': 0,
            'average_free_energy': 0.0,
            'system_uptime': 0.0
        }
        
        # Timing
        self.start_time = time.time()
        
        logger.info("DualAgentSystem initialized successfully")
    
    def process_observation(self, observation: torch.Tensor) -> Dict[str, Any]:
        """
        Main processing method: FEP agent processes observation,
        MCM monitor evaluates the agent's state.
        """
        # FEP agent processes observation
        fep_result = self.fep_agent.perceive_and_act(observation)
        
        # MCM monitor evaluates FEP agent state
        mcm_result = self.mcm_monitor.update(fep_result)
        
        # Update system metrics
        self._update_metrics(fep_result, mcm_result)
        
        # Combined result
        return {
            'fep_output': fep_result,
            'mcm_output': mcm_result,
            'system_metrics': self.performance_metrics.copy(),
            'step_count': self.step_count,
            'components_available': self.fep_agent.components_available
        }
    
    def _update_metrics(self, fep_result: Dict[str, Any], mcm_result: Dict[str, Any]):
        """Update system performance metrics."""
        self.step_count += 1
        
        # Track free energy
        current_fe = float(fep_result['free_energy'])
        self.total_free_energy += current_fe
        self.performance_metrics['average_free_energy'] = self.total_free_energy / self.step_count
        
        # Track events
        if mcm_result['chaos_detected']:
            self.performance_metrics['chaos_events'] += 1
            
        if mcm_result['surprise_level'] > self.config.surprise_threshold:
            self.performance_metrics['high_surprise_events'] += 1
        
        # System uptime
        self.performance_metrics['system_uptime'] = time.time() - self.start_time
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'config': self.config,
            'step_count': self.step_count,
            'performance_metrics': self.performance_metrics,
            'fep_components_available': self.fep_agent.components_available,
            'mcm_monitoring_active': self.mcm_monitor.monitoring_active,
            'vfe_history_length': len(self.mcm_monitor.vfe_history),
            'uptime_seconds': time.time() - self.start_time
        }
    
    def reset(self):
        """Reset the system to initial state."""
        self.step_count = 0
        self.total_free_energy = 0.0
        self.performance_metrics = {
            'chaos_events': 0,
            'high_surprise_events': 0,
            'average_free_energy': 0.0,
            'system_uptime': 0.0
        }
        self.mcm_monitor.vfe_history.clear()
        self.mcm_monitor.surprise_history.clear()
        self.start_time = time.time()

def create_dual_agent_system(config: Optional[DualAgentConfig] = None) -> DualAgentSystem:
    """Factory function to create a dual-agent system."""
    return DualAgentSystem(config)

def main():
    """Test the dual-agent system."""
    print("🧠 Testing DualAgentSystem")
    
    try:
        # Create system
        config = DualAgentConfig(observation_dim=50, action_dim=10)
        system = create_dual_agent_system(config)
        
        print(f"✅ System created: {system.fep_agent.components_available}")
        
        # Test with random observations
        for i in range(10):
            observation = torch.randn(config.observation_dim)
            result = system.process_observation(observation)
            
            print(f"Step {i+1}:")
            print(f"  Free Energy: {result['fep_output']['free_energy']:.3f}")
            print(f"  Chaos Detected: {result['mcm_output']['chaos_detected']}")
            print(f"  Components: {result['fep_output']['components_used']}")
        
        print(f"\n📊 Final Status:")
        status = system.get_status()
        for key, value in status['performance_metrics'].items():
            print(f"  {key}: {value}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
