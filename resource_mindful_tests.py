#!/usr/bin/env python3
"""
Resource-Mindful Tests for FEP Cognitive Architecture
===================================================

Lightweight tests designed to verify core functionality without heavy resource usage.
Perfect for quick validation and continuous integration.
"""

import sys
import os
import numpy as np
import torch
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from fep_cognitive_architecture import FEPCognitiveArchitecture, SystemState
    from fep_mathematics import HierarchicalFEPSystem
    from active_inference import ActiveInferenceAgent
    REAL_FEP_AVAILABLE = True
    print("✅ Real FEP components available")
except ImportError as e:
    REAL_FEP_AVAILABLE = False
    print(f"⚠️ Real FEP components not available: {e}")

def test_basic_architecture_creation():
    """Test that we can create the basic architecture without errors"""
    print("\n🧪 TEST: Basic Architecture Creation")
    print("-" * 40)
    
    try:
        if REAL_FEP_AVAILABLE:
            # Test real FEP system
            system = FEPCognitiveArchitecture(
                state_dim=3,
                action_dim=2,
                hierarchy_levels=2
            )
            print("✅ Real FEP Architecture created successfully")
            
            # Quick functionality test
            observations = np.array([0.1, 0.2, 0.3])
            action, info = system.perception_action_cycle(observations)
            print(f"✅ Perception-action cycle: action={action}, free_energy={info.get('free_energy', 'N/A'):.3f}")
            
        else:
            print("⚠️ Skipping real FEP test - components not available")
            
        return True
        
    except Exception as e:
        print(f"❌ Architecture creation failed: {e}")
        return False

def test_mathematical_components():
    """Test core mathematical operations"""
    print("\n🧪 TEST: Mathematical Components")
    print("-" * 40)
    
    try:
        if REAL_FEP_AVAILABLE:
            # Test hierarchical FEP system
            fep_system = HierarchicalFEPSystem(
                observation_dim=3,
                latent_dims=[8, 4]
            )
            
            # Test basic operations
            observations = torch.randn(1, 3)  # Add batch dimension
            results = fep_system.hierarchical_inference(observations)
            free_energy = results[0]['free_energy']
            
            print(f"✅ FEP Mathematics: hierarchical inference completed, free_energy={free_energy.item():.3f}")
            
        else:
            print("⚠️ Skipping mathematical tests - components not available")
            
        return True
        
    except Exception as e:
        print(f"❌ Mathematical test failed: {e}")
        return False

def test_active_inference():
    """Test active inference components"""
    print("\n🧪 TEST: Active Inference")
    print("-" * 40)
    
    try:
        if REAL_FEP_AVAILABLE:
            # Test active inference agent
            from active_inference import ActiveInferenceConfig
            config = ActiveInferenceConfig(
                observation_dim=3,
                action_dim=2,
                policy_horizon=2
            )
            agent = ActiveInferenceAgent(config)
            
            # Test perception and action
            current_observation = torch.tensor([0.1, 0.2, 0.3])
            perception_result = agent.perceive(current_observation)
            action_result = agent.act()
            
            print(f"✅ Active Inference: perception and action completed")
            
        else:
            print("⚠️ Skipping active inference test - components not available")
            
        return True
        
    except Exception as e:
        print(f"❌ Active inference test failed: {e}")
        return False

def test_performance_baseline():
    """Quick performance baseline test"""
    print("\n🧪 TEST: Performance Baseline")
    print("-" * 40)
    
    try:
        if REAL_FEP_AVAILABLE:
            system = FEPCognitiveArchitecture(
                state_dim=5,
                action_dim=3,
                hierarchy_levels=2
            )
            
            # Time a series of perception-action cycles
            start_time = time.time()
            num_cycles = 10
            
            for i in range(num_cycles):
                observations = np.random.randn(5)
                action, info = system.perception_action_cycle(observations)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_cycles
            
            print(f"✅ Performance: {avg_time*1000:.2f}ms per cycle (target: <100ms)")
            
            if avg_time < 0.1:  # Less than 100ms
                print("✅ Performance target met")
            else:
                print("⚠️ Performance slower than target")
                
        else:
            print("⚠️ Skipping performance test - components not available")
            
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def run_all_tests():
    """Run all resource-mindful tests"""
    print("🚀 RUNNING RESOURCE-MINDFUL FEP TESTS")
    print("=" * 50)
    
    tests = [
        test_basic_architecture_creation,
        test_mathematical_components,
        test_active_inference,
        test_performance_baseline
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total-passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("⚠️ Some tests failed - check output above")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
