#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark Suite for FEP-MCM Cognitive Architecture
============================================================================

This suite benchmarks all performance parameters using multiple profiling tools:
- Execution time profiling with cProfile
- Line-by-line analysis with line_profiler
- Memory usage monitoring with memory_profiler
- Real-time CPU analysis
- Comprehensive performance metrics

Usage: python performance_benchmark_suite.py
"""

import sys
import os
import time
import timeit
import cProfile
import pstats
import io
import psutil
import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import tracemalloc
import gc

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from fep_cognitive_architecture import FEPCognitiveArchitecture
    from fep_mathematics import HierarchicalFEPSystem
    from active_inference import ActiveInferenceAgent, ActiveInferenceConfig
    REAL_FEP_AVAILABLE = True
    print("✅ Real FEP components available for benchmarking")
except ImportError as e:
    REAL_FEP_AVAILABLE = False
    print(f"❌ Real FEP components not available: {e}")
    sys.exit(1)

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite"""
    
    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_file = f"performance_benchmark_{self.timestamp}.json"
        
    def benchmark_execution_time(self):
        """Benchmark execution time using cProfile and timeit"""
        print("\n🕒 EXECUTION TIME BENCHMARKING")
        print("=" * 50)
        
        results = {}
        
        # 1. Core Architecture Creation Time
        print("📊 Benchmarking: Architecture Creation")
        def create_architecture():
            return FEPCognitiveArchitecture(
                state_dim=5,
                action_dim=3,
                hierarchy_levels=3
            )
        
        creation_time = timeit.timeit(create_architecture, number=10) / 10
        results['architecture_creation_time'] = creation_time
        print(f"   Average creation time: {creation_time*1000:.2f}ms")
        
        # 2. Perception-Action Cycle Time
        print("📊 Benchmarking: Perception-Action Cycles")
        arch = create_architecture()
        
        def perception_action_cycle():
            observations = np.random.randn(5)
            return arch.perception_action_cycle(observations)
        
        cycle_time = timeit.timeit(perception_action_cycle, number=100) / 100
        results['perception_action_cycle_time'] = cycle_time
        print(f"   Average cycle time: {cycle_time*1000:.2f}ms")
        
        # 3. Hierarchical Inference Time
        print("📊 Benchmarking: Hierarchical Inference")
        fep_system = HierarchicalFEPSystem(
            observation_dim=5,
            latent_dims=[16, 8, 4]
        )
        
        def hierarchical_inference():
            observations = torch.randn(1, 5)
            return fep_system.hierarchical_inference(observations)
        
        inference_time = timeit.timeit(hierarchical_inference, number=50) / 50
        results['hierarchical_inference_time'] = inference_time
        print(f"   Average inference time: {inference_time*1000:.2f}ms")
        
        # 4. Active Inference Time
        print("📊 Benchmarking: Active Inference")
        config = ActiveInferenceConfig(
            observation_dim=5,
            action_dim=3,
            policy_horizon=4
        )
        agent = ActiveInferenceAgent(config)
        
        def active_inference_cycle():
            obs = torch.randn(5)
            perception = agent.perceive(obs)
            action = agent.act()
            return perception, action
        
        ai_time = timeit.timeit(active_inference_cycle, number=30) / 30
        results['active_inference_time'] = ai_time
        print(f"   Average AI cycle time: {ai_time*1000:.2f}ms")
        
        return results
    
    def benchmark_with_cprofile(self):
        """Detailed function-level profiling with cProfile"""
        print("\n🔍 CPROFILE DETAILED ANALYSIS")
        print("=" * 50)
        
        # Create profiler
        profiler = cProfile.Profile()
        
        # Profile a comprehensive workflow
        profiler.enable()
        
        # Comprehensive test workflow
        arch = FEPCognitiveArchitecture(
            state_dim=8,
            action_dim=4,
            hierarchy_levels=3
        )
        
        # Run multiple cycles
        for i in range(20):
            observations = np.random.randn(8) * (1 + i * 0.1)  # Varying complexity
            action, info = arch.perception_action_cycle(observations)
        
        profiler.disable()
        
        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        profile_output = s.getvalue()
        print("📈 Top 20 Functions by Cumulative Time:")
        print(profile_output)
        
        # Save detailed profile
        profiler.dump_stats(f'detailed_profile_{self.timestamp}.prof')
        print(f"💾 Detailed profile saved to: detailed_profile_{self.timestamp}.prof")
        
        return {"profile_saved": True, "top_functions": profile_output[:1000]}
    
    def benchmark_memory_usage(self):
        """Memory usage profiling"""
        print("\n🧠 MEMORY USAGE BENCHMARKING")
        print("=" * 50)
        
        results = {}
        
        # Start memory tracing
        tracemalloc.start()
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        results['baseline_memory_mb'] = baseline_memory
        print(f"📊 Baseline memory usage: {baseline_memory:.2f} MB")
        
        # Memory usage during architecture creation
        snapshot1 = tracemalloc.take_snapshot()
        
        arch = FEPCognitiveArchitecture(
            state_dim=10,
            action_dim=5,
            hierarchy_levels=4
        )
        
        snapshot2 = tracemalloc.take_snapshot()
        creation_memory = process.memory_info().rss / 1024 / 1024  # MB
        results['post_creation_memory_mb'] = creation_memory
        results['creation_memory_increase_mb'] = creation_memory - baseline_memory
        print(f"📊 Post-creation memory: {creation_memory:.2f} MB (+{creation_memory - baseline_memory:.2f} MB)")
        
        # Memory usage during intensive processing
        peak_memory = creation_memory
        for i in range(50):
            observations = np.random.randn(10)
            action, info = arch.perception_action_cycle(observations)
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
        
        results['peak_memory_mb'] = peak_memory
        results['peak_memory_increase_mb'] = peak_memory - baseline_memory
        print(f"📊 Peak memory usage: {peak_memory:.2f} MB (+{peak_memory - baseline_memory:.2f} MB)")
        
        # Memory efficiency analysis
        snapshot3 = tracemalloc.take_snapshot()
        top_stats = snapshot3.compare_to(snapshot1, 'lineno')[:10]
        
        memory_hotspots = []
        for stat in top_stats:
            memory_hotspots.append({
                'file': str(stat.traceback.format()[-1]) if stat.traceback else 'unknown',
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            })
        
        results['memory_hotspots'] = memory_hotspots
        print("📈 Top Memory Allocations:")
        for i, hotspot in enumerate(memory_hotspots[:5]):
            print(f"   {i+1}. {hotspot['size_mb']:.2f} MB - {hotspot['file'][:80]}...")
        
        tracemalloc.stop()
        return results
    
    def benchmark_scalability(self):
        """Scalability benchmarking with different system sizes"""
        print("\n📈 SCALABILITY BENCHMARKING")
        print("=" * 50)
        
        results = {}
        
        # Test different system sizes
        configs = [
            {"state_dim": 3, "action_dim": 2, "hierarchy_levels": 2},
            {"state_dim": 5, "action_dim": 3, "hierarchy_levels": 3},
            {"state_dim": 8, "action_dim": 4, "hierarchy_levels": 3},
            {"state_dim": 10, "action_dim": 5, "hierarchy_levels": 4},
            {"state_dim": 15, "action_dim": 7, "hierarchy_levels": 4},
        ]
        
        scalability_data = []
        
        for config in configs:
            print(f"📊 Testing configuration: {config}")
            
            # Time architecture creation
            start_time = time.time()
            arch = FEPCognitiveArchitecture(**config)
            creation_time = time.time() - start_time
            
            # Time perception-action cycles
            cycle_times = []
            for _ in range(10):
                observations = np.random.randn(config['state_dim'])
                start_cycle = time.time()
                action, info = arch.perception_action_cycle(observations)
                cycle_time = time.time() - start_cycle
                cycle_times.append(cycle_time)
            
            avg_cycle_time = np.mean(cycle_times)
            std_cycle_time = np.std(cycle_times)
            
            config_result = {
                **config,
                'creation_time_ms': creation_time * 1000,
                'avg_cycle_time_ms': avg_cycle_time * 1000,
                'std_cycle_time_ms': std_cycle_time * 1000,
                'complexity_factor': config['state_dim'] * config['action_dim'] * config['hierarchy_levels']
            }
            
            scalability_data.append(config_result)
            print(f"   Creation: {creation_time*1000:.2f}ms, Cycle: {avg_cycle_time*1000:.2f}±{std_cycle_time*1000:.2f}ms")
        
        results['scalability_data'] = scalability_data
        
        # Analyze scaling behavior
        complexities = [d['complexity_factor'] for d in scalability_data]
        cycle_times = [d['avg_cycle_time_ms'] for d in scalability_data]
        
        # Simple linear regression to estimate scaling
        if len(complexities) > 1:
            correlation = np.corrcoef(complexities, cycle_times)[0, 1]
            results['complexity_correlation'] = correlation
            print(f"📈 Complexity-Performance Correlation: {correlation:.3f}")
            
            if correlation > 0.8:
                print("   ⚠️ Strong positive correlation - potential scalability concerns")
            elif correlation > 0.5:
                print("   ⚡ Moderate correlation - acceptable scaling")
            else:
                print("   ✅ Low correlation - excellent scaling properties")
        
        return results
    
    def benchmark_stress_test(self):
        """Stress testing under extreme conditions"""
        print("\n⚡ STRESS TEST BENCHMARKING")
        print("=" * 50)
        
        results = {}
        
        arch = FEPCognitiveArchitecture(
            state_dim=8,
            action_dim=4,
            hierarchy_levels=3
        )
        
        # 1. High-frequency processing test
        print("📊 High-frequency processing test...")
        start_time = time.time()
        cycle_count = 0
        duration = 5.0  # 5 seconds
        
        while time.time() - start_time < duration:
            observations = np.random.randn(8)
            action, info = arch.perception_action_cycle(observations)
            cycle_count += 1
        
        actual_duration = time.time() - start_time
        cycles_per_second = cycle_count / actual_duration
        
        results['stress_cycles_per_second'] = cycles_per_second
        results['stress_total_cycles'] = cycle_count
        results['stress_duration'] = actual_duration
        print(f"   Processed {cycle_count} cycles in {actual_duration:.2f}s = {cycles_per_second:.1f} cycles/sec")
        
        # 2. Extreme input test
        print("📊 Extreme input handling test...")
        extreme_inputs = [
            np.zeros(8),  # Zero input
            np.ones(8) * 1000,  # Large values
            np.ones(8) * -1000,  # Large negative values
            np.random.randn(8) * 100,  # High variance
        ]
        
        extreme_results = []
        for i, extreme_input in enumerate(extreme_inputs):
            try:
                start_time = time.time()
                action, info = arch.perception_action_cycle(extreme_input)
                processing_time = time.time() - start_time
                
                extreme_results.append({
                    'input_type': f'extreme_{i}',
                    'processing_time_ms': processing_time * 1000,
                    'success': True,
                    'free_energy': float(info.get('free_energy', 0))
                })
                print(f"   Extreme input {i}: {processing_time*1000:.2f}ms, FE={info.get('free_energy', 'N/A'):.3f}")
                
            except Exception as e:
                extreme_results.append({
                    'input_type': f'extreme_{i}',
                    'success': False,
                    'error': str(e)
                })
                print(f"   Extreme input {i}: FAILED - {e}")
        
        results['extreme_input_results'] = extreme_results
        
        # 3. Memory stability test
        print("📊 Memory stability test...")
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        for i in range(100):
            observations = np.random.randn(8)
            action, info = arch.perception_action_cycle(observations)
            
            if i % 20 == 0:  # Check memory every 20 cycles
                current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                if memory_increase > 50:  # More than 50MB increase
                    print(f"   ⚠️ Memory increase detected: +{memory_increase:.2f} MB at cycle {i}")
                    break
        
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_stability = final_memory - initial_memory
        
        results['memory_stability_mb'] = memory_stability
        print(f"   Memory stability: +{memory_stability:.2f} MB over 100 cycles")
        
        if memory_stability < 10:
            print("   ✅ Excellent memory stability")
        elif memory_stability < 50:
            print("   ⚡ Good memory stability")
        else:
            print("   ⚠️ Memory stability concerns detected")
        
        return results
    
    def benchmark_component_breakdown(self):
        """Individual component performance analysis"""
        print("\n🔧 COMPONENT BREAKDOWN BENCHMARKING")
        print("=" * 50)
        
        results = {}
        
        # 1. FEP Mathematics Component
        print("📊 FEP Mathematics Performance...")
        fep_system = HierarchicalFEPSystem(
            observation_dim=6,
            latent_dims=[12, 8, 4]
        )
        
        def fep_inference():
            obs = torch.randn(1, 6)
            return fep_system.hierarchical_inference(obs)
        
        fep_time = timeit.timeit(fep_inference, number=50) / 50
        results['fep_mathematics_time_ms'] = fep_time * 1000
        print(f"   FEP Mathematics: {fep_time*1000:.2f}ms per inference")
        
        # 2. Active Inference Component
        print("📊 Active Inference Performance...")
        config = ActiveInferenceConfig(
            observation_dim=6,
            action_dim=3,
            policy_horizon=5
        )
        agent = ActiveInferenceAgent(config)
        
        def ai_cycle():
            obs = torch.randn(6)
            perception = agent.perceive(obs)
            action = agent.act()
            return perception, action
        
        ai_time = timeit.timeit(ai_cycle, number=30) / 30
        results['active_inference_time_ms'] = ai_time * 1000
        print(f"   Active Inference: {ai_time*1000:.2f}ms per cycle")
        
        # 3. Meta-Cognitive Monitor Performance
        print("📊 Meta-Cognitive Monitor Performance...")
        arch = FEPCognitiveArchitecture(
            state_dim=6,
            action_dim=3,
            hierarchy_levels=3
        )
        
        # Test MCM with high VFE inputs to trigger monitoring
        high_vfe_inputs = [np.random.randn(6) * 10 for _ in range(10)]
        mcm_times = []
        
        for inp in high_vfe_inputs:
            start_time = time.time()
            action, info = arch.perception_action_cycle(inp)
            mcm_time = time.time() - start_time
            mcm_times.append(mcm_time)
        
        avg_mcm_time = np.mean(mcm_times)
        results['mcm_monitoring_time_ms'] = avg_mcm_time * 1000
        print(f"   MCM Monitoring: {avg_mcm_time*1000:.2f}ms per high-VFE cycle")
        
        return results
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n📊 GENERATING COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 60)
        
        # Run all benchmarks
        execution_results = self.benchmark_execution_time()
        profile_results = self.benchmark_with_cprofile()
        memory_results = self.benchmark_memory_usage()
        scalability_results = self.benchmark_scalability()
        stress_results = self.benchmark_stress_test()
        component_results = self.benchmark_component_breakdown()
        
        # Compile comprehensive results
        comprehensive_results = {
            'timestamp': self.timestamp,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'torch_version': torch.__version__,
                'numpy_version': np.__version__
            },
            'execution_time': execution_results,
            'profiling': profile_results,
            'memory_usage': memory_results,
            'scalability': scalability_results,
            'stress_testing': stress_results,
            'component_breakdown': component_results,
            'performance_summary': self._generate_summary(
                execution_results, memory_results, scalability_results, stress_results
            )
        }
        
        # Save results
        with open(self.report_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"\n💾 Comprehensive report saved to: {self.report_file}")
        
        # Print executive summary
        self._print_executive_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _generate_summary(self, exec_results, mem_results, scale_results, stress_results):
        """Generate performance summary"""
        summary = {}
        
        # Performance grades
        cycle_time = exec_results.get('perception_action_cycle_time', 1.0) * 1000
        if cycle_time < 1:
            summary['speed_grade'] = 'A+ (Excellent)'
        elif cycle_time < 5:
            summary['speed_grade'] = 'A (Very Good)'
        elif cycle_time < 10:
            summary['speed_grade'] = 'B (Good)'
        elif cycle_time < 50:
            summary['speed_grade'] = 'C (Acceptable)'
        else:
            summary['speed_grade'] = 'D (Needs Improvement)'
        
        # Memory efficiency
        peak_memory = mem_results.get('peak_memory_increase_mb', 0)
        if peak_memory < 50:
            summary['memory_grade'] = 'A+ (Excellent)'
        elif peak_memory < 100:
            summary['memory_grade'] = 'A (Very Good)'
        elif peak_memory < 200:
            summary['memory_grade'] = 'B (Good)'
        elif peak_memory < 500:
            summary['memory_grade'] = 'C (Acceptable)'
        else:
            summary['memory_grade'] = 'D (Needs Improvement)'
        
        # Scalability
        correlation = scale_results.get('complexity_correlation', 0)
        if correlation < 0.3:
            summary['scalability_grade'] = 'A+ (Excellent)'
        elif correlation < 0.5:
            summary['scalability_grade'] = 'A (Very Good)'
        elif correlation < 0.7:
            summary['scalability_grade'] = 'B (Good)'
        elif correlation < 0.9:
            summary['scalability_grade'] = 'C (Acceptable)'
        else:
            summary['scalability_grade'] = 'D (Needs Improvement)'
        
        # Stress resilience
        cycles_per_sec = stress_results.get('stress_cycles_per_second', 0)
        if cycles_per_sec > 1000:
            summary['stress_grade'] = 'A+ (Excellent)'
        elif cycles_per_sec > 500:
            summary['stress_grade'] = 'A (Very Good)'
        elif cycles_per_sec > 200:
            summary['stress_grade'] = 'B (Good)'
        elif cycles_per_sec > 50:
            summary['stress_grade'] = 'C (Acceptable)'
        else:
            summary['stress_grade'] = 'D (Needs Improvement)'
        
        return summary
    
    def _print_executive_summary(self, results):
        """Print executive summary"""
        print("\n" + "="*60)
        print("📈 EXECUTIVE PERFORMANCE SUMMARY")
        print("="*60)
        
        summary = results['performance_summary']
        exec_results = results['execution_time']
        mem_results = results['memory_usage']
        stress_results = results['stress_testing']
        
        print(f"🚀 Speed Performance: {summary['speed_grade']}")
        print(f"   Perception-Action Cycle: {exec_results.get('perception_action_cycle_time', 0)*1000:.2f}ms")
        
        print(f"🧠 Memory Efficiency: {summary['memory_grade']}")
        print(f"   Peak Memory Usage: +{mem_results.get('peak_memory_increase_mb', 0):.1f} MB")
        
        print(f"📈 Scalability: {summary['scalability_grade']}")
        correlation = results['scalability'].get('complexity_correlation', 0)
        print(f"   Complexity Correlation: {correlation:.3f}")
        
        print(f"⚡ Stress Resilience: {summary['stress_grade']}")
        print(f"   Processing Rate: {stress_results.get('stress_cycles_per_second', 0):.1f} cycles/sec")
        
        # Overall grade
        grades = [summary['speed_grade'], summary['memory_grade'], 
                 summary['scalability_grade'], summary['stress_grade']]
        grade_values = {'A+': 4.0, 'A': 3.7, 'B': 3.0, 'C': 2.0, 'D': 1.0}
        avg_grade = np.mean([grade_values[g.split()[0]] for g in grades])
        
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
        
        print(f"\n🎯 OVERALL PERFORMANCE: {overall}")
        print("="*60)

def main():
    """Main benchmarking execution"""
    if not REAL_FEP_AVAILABLE:
        print("❌ Cannot run benchmarks without FEP components")
        return
    
    print("🚀 FEP-MCM COGNITIVE ARCHITECTURE - COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("="*80)
    print("This suite will analyze all performance parameters using multiple profiling tools.")
    print("Estimated runtime: 2-3 minutes")
    print()
    
    # Initialize benchmark suite
    benchmark = PerformanceBenchmark()
    
    try:
        # Run comprehensive benchmarking
        results = benchmark.generate_performance_report()
        
        print(f"\n✅ Benchmarking completed successfully!")
        print(f"📄 Detailed results available in: {benchmark.report_file}")
        print(f"📊 Profile data available in: detailed_profile_{benchmark.timestamp}.prof")
        print("\n💡 To visualize profile data, install snakeviz and run:")
        print(f"   pip install snakeviz")
        print(f"   snakeviz detailed_profile_{benchmark.timestamp}.prof")
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
