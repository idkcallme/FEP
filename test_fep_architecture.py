#!/usr/bin/env python3
"""
FEP Architecture Integration Tests
=================================

Comprehensive tests for the FEP Cognitive Architecture system.
Tests both the conceptual framework and real implementations.
"""

import sys
import os
import numpy as np
import torch
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from fep_cognitive_architecture import FEPCognitiveArchitecture, SystemState
    from fep_mathematics import HierarchicalFEPSystem
    from active_inference import ActiveInferenceAgent
    from predictive_coding import PredictiveCodingHierarchy
    REAL_FEP_AVAILABLE = True
    print("✅ Real FEP components available for testing")
except ImportError as e:
    REAL_FEP_AVAILABLE = False
    print(f"⚠️ Real FEP components not available: {e}")

class TestFEPArchitecture:
    """Test suite for FEP Cognitive Architecture"""
    
    def test_architecture_initialization(self):
        """Test that architecture initializes correctly"""
        if not REAL_FEP_AVAILABLE:
            pytest.skip("Real FEP components not available")
            
        arch = FEPCognitiveArchitecture(
            state_dim=4,
            action_dim=3,
            hierarchy_levels=2
        )
        
        # Architecture should be initialized successfully
        assert hasattr(arch, 'perception_action_cycle')
        assert arch.system_state == SystemState.STABLE
    
    def test_perception_action_cycle(self):
        """Test the core perception-action cycle"""
        if not REAL_FEP_AVAILABLE:
            pytest.skip("Real FEP components not available")
            
        arch = FEPCognitiveArchitecture(
            state_dim=3,
            action_dim=2,
            hierarchy_levels=2
        )
        
        observations = np.array([0.1, 0.2, 0.3])
        action, info = arch.perception_action_cycle(observations)
        
        # Verify outputs
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        assert isinstance(info, dict)
        assert 'free_energy' in info
        assert isinstance(info['free_energy'], (int, float))
    
    def test_meta_cognitive_monitoring(self):
        """Test the Meta-Cognitive Monitor functionality"""
        if not REAL_FEP_AVAILABLE:
            pytest.skip("Real FEP components not available")
            
        arch = FEPCognitiveArchitecture(
            state_dim=5,
            action_dim=3,
            hierarchy_levels=3
        )
        
        # Run multiple cycles to trigger MCM
        high_noise_obs = np.random.randn(5) * 10  # High noise to trigger MCM
        
        for _ in range(5):
            action, info = arch.perception_action_cycle(high_noise_obs)
            
        # MCM should have detected the high free energy
        assert 'mcm_state' in info
        assert 'system_health' in info
    
    def test_hierarchical_processing(self):
        """Test hierarchical belief processing"""
        if not REAL_FEP_AVAILABLE:
            pytest.skip("Real FEP components not available")
            
        # Test with different hierarchy levels
        for levels in [2, 3, 4]:
            arch = FEPCognitiveArchitecture(
                state_dim=4,
                action_dim=2,
                hierarchy_levels=levels
            )
            
            observations = np.random.randn(4)
            action, info = arch.perception_action_cycle(observations)
            
            assert isinstance(action, np.ndarray)
            assert 'free_energy' in info

class TestFEPMathematics:
    """Test suite for FEP mathematical components"""
    
    def test_hierarchical_fep_system(self):
        """Test hierarchical FEP system operations"""
        if not REAL_FEP_AVAILABLE:
            pytest.skip("Real FEP components not available")
            
        system = HierarchicalFEPSystem(
            observation_dim=3,
            latent_dims=[8, 4]
        )
        
        observations = torch.randn(1, 3)
        results = system.hierarchical_inference(observations)
        free_energy = results[0]['free_energy'].item()
        
        assert isinstance(results, list)
        assert isinstance(free_energy, (int, float))
    
    def test_belief_updating(self):
        """Test belief updating mechanism"""
        if not REAL_FEP_AVAILABLE:
            pytest.skip("Real FEP components not available")
            
        system = HierarchicalFEPSystem(
            observation_dim=3,
            latent_dims=[6, 4]
        )
        
        # Initial beliefs
        observations1 = torch.tensor([[1.0, 0.0, 0.0]])
        results1 = system.hierarchical_inference(observations1)
        
        # Updated beliefs with new observations
        observations2 = torch.tensor([[0.0, 1.0, 0.0]])
        results2 = system.hierarchical_inference(observations2)
        
        # Results should be different for different inputs
        assert results1[0]['free_energy'] != results2[0]['free_energy']

class TestActiveInference:
    """Test suite for Active Inference components"""
    
    def test_action_selection(self):
        """Test action selection mechanism"""
        if not REAL_FEP_AVAILABLE:
            pytest.skip("Real FEP components not available")
            
        from active_inference import ActiveInferenceConfig
        config = ActiveInferenceConfig(
            observation_dim=3,
            action_dim=2,
            policy_horizon=2
        )
        agent = ActiveInferenceAgent(config)
        
        current_observation = torch.tensor([0.1, 0.2, 0.3])
        perception_result = agent.perceive(current_observation)
        action_result = agent.act()
        
        assert isinstance(perception_result, dict)
        assert isinstance(action_result, dict)
    
    def test_policy_evaluation(self):
        """Test policy evaluation and selection"""
        if not REAL_FEP_AVAILABLE:
            pytest.skip("Real FEP components not available")
            
        from active_inference import ActiveInferenceConfig
        config = ActiveInferenceConfig(
            observation_dim=4,
            action_dim=3,
            policy_horizon=3
        )
        agent = ActiveInferenceAgent(config)
        
        # Test with different states
        states = [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 0.0])
        ]
        
        results = []
        for state in states:
            perception_result = agent.perceive(state)
            results.append(str(perception_result))
        
        # Results should be different for different states
        assert len(set(results)) > 1

def run_integration_tests():
    """Run all integration tests manually"""
    print("🧪 RUNNING FEP ARCHITECTURE INTEGRATION TESTS")
    print("=" * 60)
    
    if not REAL_FEP_AVAILABLE:
        print("⚠️ Real FEP components not available - skipping tests")
        return False
    
    # Manual test execution
    test_arch = TestFEPArchitecture()
    test_math = TestFEPMathematics()
    test_ai = TestActiveInference()
    
    tests = [
        (test_arch.test_architecture_initialization, "Architecture Initialization"),
        (test_arch.test_perception_action_cycle, "Perception-Action Cycle"),
        (test_arch.test_meta_cognitive_monitoring, "Meta-Cognitive Monitoring"),
        (test_arch.test_hierarchical_processing, "Hierarchical Processing"),
        (test_math.test_hierarchical_fep_system, "Hierarchical FEP System"),
        (test_math.test_belief_updating, "Belief Updating"),
        (test_ai.test_action_selection, "Action Selection"),
        (test_ai.test_policy_evaluation, "Policy Evaluation")
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func, test_name in tests:
        try:
            print(f"\n🧪 Testing: {test_name}")
            test_func()
            print(f"✅ {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
    
    print(f"\n📊 RESULTS: {passed}/{total} tests passed")
    return passed == total

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        success = run_integration_tests()
        sys.exit(0 if success else 1)
    else:
        # Run with pytest
        pytest.main([__file__, "-v"])
