#!/usr/bin/env python3

import sys
import os
import time
import timeit
import psutil
import numpy as np
import torch
from datetime import datetime

# Add src to path
sys.path.append('src')
from fep_mathematics import HierarchicalFEPSystem
from active_inference import ActiveInferenceAgent, ActiveInferenceConfig
from fep_cognitive_architecture import FEPCognitiveArchitecture

def main():
    print('🔥 COMPREHENSIVE PERFORMANCE BENCHMARK')
    print('='*60)
    
    # System info
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / 1024 / 1024
    
    print(f'🖥️  System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total/1024/1024/1024:.1f}GB RAM')
    print(f'🐍 Python: {sys.version.split()[0]}')
    print(f'🔥 PyTorch: {torch.__version__}')
    print(f'📊 NumPy: {np.__version__}')
    print(f'💾 Baseline Memory: {baseline_memory:.1f}MB')
    
    # 1. EXECUTION TIME BENCHMARKS
    print('\n📊 1. EXECUTION TIME BENCHMARKS')
    print('-'*40)
    
    # Architecture creation time
    def create_arch():
        return FEPCognitiveArchitecture(state_dim=5, action_dim=3, hierarchy_levels=3)
    
    creation_time = timeit.timeit(create_arch, number=10) / 10
    print(f'⚡ Architecture Creation: {creation_time*1000:.2f}ms')
    
    # Perception-action cycle time
    arch = create_arch()
    def cycle():
        obs = np.random.randn(5)
        return arch.perception_action_cycle(obs)
    
    cycle_time = timeit.timeit(cycle, number=100) / 100
    print(f'⚡ Perception-Action Cycle: {cycle_time*1000:.2f}ms')
    
    # Hierarchical inference time
    fep_system = HierarchicalFEPSystem(observation_dim=5, latent_dims=[16, 8, 4])
    def inference():
        obs = torch.randn(1, 5)
        return fep_system.hierarchical_inference(obs)
    
    inference_time = timeit.timeit(inference, number=50) / 50
    print(f'⚡ Hierarchical Inference: {inference_time*1000:.2f}ms')
    
    # Active inference time
    config = ActiveInferenceConfig(observation_dim=5, action_dim=3, policy_horizon=4)
    agent = ActiveInferenceAgent(config)
    def ai_cycle():
        obs = torch.randn(5)
        perception = agent.perceive(obs)
        action = agent.act()
        return perception, action
    
    ai_time = timeit.timeit(ai_cycle, number=30) / 30
    print(f'⚡ Active Inference Cycle: {ai_time*1000:.2f}ms')
    
    # 2. MEMORY BENCHMARKS
    print('\n💾 2. MEMORY USAGE BENCHMARKS')
    print('-'*40)
    
    # Memory after architecture creation
    current_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = current_memory - baseline_memory
    print(f'💾 Post-Architecture Memory: {current_memory:.1f}MB (+{memory_increase:.1f}MB)')
    
    # Peak memory during intensive processing
    peak_memory = current_memory
    for i in range(50):
        obs = np.random.randn(5)
        action, info = arch.perception_action_cycle(obs)
        current_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, current_memory)
    
    peak_increase = peak_memory - baseline_memory
    print(f'💾 Peak Processing Memory: {peak_memory:.1f}MB (+{peak_increase:.1f}MB)')
    
    # 3. SCALABILITY BENCHMARKS
    print('\n📈 3. SCALABILITY BENCHMARKS')
    print('-'*40)
    
    configs = [
        {"state_dim": 3, "action_dim": 2, "hierarchy_levels": 2},
        {"state_dim": 5, "action_dim": 3, "hierarchy_levels": 3},
        {"state_dim": 8, "action_dim": 4, "hierarchy_levels": 3},
        {"state_dim": 10, "action_dim": 5, "hierarchy_levels": 4},
    ]
    
    for config in configs:
        # Time architecture creation
        start_time = time.time()
        arch = FEPCognitiveArchitecture(**config)
        creation_time = time.time() - start_time
        
        # Time perception-action cycles
        cycle_times = []
        for _ in range(10):
            obs = np.random.randn(config['state_dim'])
            start_cycle = time.time()
            action, info = arch.perception_action_cycle(obs)
            cycle_time = time.time() - start_cycle
            cycle_times.append(cycle_time)
        
        avg_cycle_time = np.mean(cycle_times)
        complexity = config['state_dim'] * config['action_dim'] * config['hierarchy_levels']
        
        print(f'📈 Config {config["state_dim"]}x{config["action_dim"]}x{config["hierarchy_levels"]} (complexity={complexity}):')
        print(f'   Creation: {creation_time*1000:.2f}ms, Cycle: {avg_cycle_time*1000:.2f}ms')
    
    # 4. STRESS TEST
    print('\n🔥 4. STRESS TEST BENCHMARKS')
    print('-'*40)
    
    arch = FEPCognitiveArchitecture(state_dim=8, action_dim=4, hierarchy_levels=3)
    
    # High-frequency processing test
    start_time = time.time()
    cycle_count = 0
    duration = 3.0  # 3 seconds
    
    while time.time() - start_time < duration:
        obs = np.random.randn(8)
        action, info = arch.perception_action_cycle(obs)
        cycle_count += 1
    
    actual_duration = time.time() - start_time
    cycles_per_second = cycle_count / actual_duration
    
    print(f'🔥 High-Frequency Test: {cycle_count} cycles in {actual_duration:.2f}s')
    print(f'🔥 Processing Rate: {cycles_per_second:.1f} cycles/second')
    
    # Extreme input test
    extreme_inputs = [
        np.zeros(8),              # Zero input
        np.ones(8) * 1000,        # Large values
        np.ones(8) * -1000,       # Large negative values
        np.random.randn(8) * 100, # High variance
    ]
    
    print(f'🔥 Extreme Input Handling:')
    for i, extreme_input in enumerate(extreme_inputs):
        try:
            start_time = time.time()
            action, info = arch.perception_action_cycle(extreme_input)
            processing_time = time.time() - start_time
            fe = info.get('free_energy', 0)
            print(f'   Input {i}: {processing_time*1000:.2f}ms, FE={fe:.3f} ✅')
        except Exception as e:
            print(f'   Input {i}: FAILED - {e} ❌')
    
    # 5. PERFORMANCE SUMMARY
    print('\n🏆 5. PERFORMANCE SUMMARY')
    print('='*60)
    
    # Speed grade
    if cycle_time * 1000 < 1:
        speed_grade = 'A+ (Excellent)'
    elif cycle_time * 1000 < 5:
        speed_grade = 'A (Very Good)'
    elif cycle_time * 1000 < 10:
        speed_grade = 'B (Good)'
    elif cycle_time * 1000 < 50:
        speed_grade = 'C (Acceptable)'
    else:
        speed_grade = 'D (Needs Improvement)'
    
    # Memory grade
    if peak_increase < 50:
        memory_grade = 'A+ (Excellent)'
    elif peak_increase < 100:
        memory_grade = 'A (Very Good)'
    elif peak_increase < 200:
        memory_grade = 'B (Good)'
    elif peak_increase < 500:
        memory_grade = 'C (Acceptable)'
    else:
        memory_grade = 'D (Needs Improvement)'
    
    # Stress grade
    if cycles_per_second > 1000:
        stress_grade = 'A+ (Excellent)'
    elif cycles_per_second > 500:
        stress_grade = 'A (Very Good)'
    elif cycles_per_second > 200:
        stress_grade = 'B (Good)'
    elif cycles_per_second > 50:
        stress_grade = 'C (Acceptable)'
    else:
        stress_grade = 'D (Needs Improvement)'
    
    print(f'🏃 Speed Performance: {speed_grade}')
    print(f'   ⚡ {cycle_time*1000:.2f}ms per perception-action cycle')
    print(f'💾 Memory Efficiency: {memory_grade}')
    print(f'   📊 Peak memory increase: +{peak_increase:.1f}MB')
    print(f'🔥 Stress Resilience: {stress_grade}')
    print(f'   ⚡ {cycles_per_second:.1f} cycles/second under stress')
    
    # Overall grade calculation
    grades = {'A+': 4.0, 'A': 3.7, 'B': 3.0, 'C': 2.0, 'D': 1.0}
    speed_val = grades[speed_grade.split()[0]]
    memory_val = grades[memory_grade.split()[0]]
    stress_val = grades[stress_grade.split()[0]]
    avg_grade = (speed_val + memory_val + stress_val) / 3
    
    if avg_grade >= 3.8:
        overall = "A+ (Outstanding)"
    elif avg_grade >= 3.5:
        overall = "A (Excellent)"
    elif avg_grade >= 2.5:
        overall = "B (Good)"
    elif avg_grade >= 1.5:
        overall = "C (Acceptable)"
    else:
        overall = "D (Needs Improvement)"
    
    print(f'\n🏆 OVERALL PERFORMANCE: {overall}')
    print('='*60)
    
    # Component-specific metrics
    print(f'\n📋 DETAILED METRICS:')
    print(f'   Architecture Creation: {creation_time*1000:.2f}ms')
    print(f'   Perception-Action Cycle: {cycle_time*1000:.2f}ms')
    print(f'   Hierarchical Inference: {inference_time*1000:.2f}ms')
    print(f'   Active Inference: {ai_time*1000:.2f}ms')
    print(f'   Memory Baseline: {baseline_memory:.1f}MB')
    print(f'   Memory Peak: {peak_memory:.1f}MB')
    print(f'   Stress Cycles/sec: {cycles_per_second:.1f}')
    print(f'   Test Duration: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == "__main__":
    main()
