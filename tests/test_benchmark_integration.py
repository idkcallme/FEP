#!/usr/bin/env python3
"""
Simple test script for FEP-MCM benchmark integration
"""

import sys
import os

def main():
    print("🚀 Testing FEP-MCM Benchmark Integration")
    print("=" * 45)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        import numpy as np
        print("   ✅ NumPy imported successfully")
        
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        print("   ✅ Matplotlib imported successfully")
        
        # Test our benchmark integration
        print("🧠 Testing benchmark integration...")
        from fep_mcm_benchmark_integration import FEPMCMBenchmarkIntegration
        
        # Initialize with mock system
        benchmark_system = FEPMCMBenchmarkIntegration(use_real_fep_mcm=False)
        print("   ✅ Benchmark system initialized")
        
        # Run a small test
        print("🔍 Running mini TruthfulQA test...")
        results = benchmark_system.run_truthfulqa_benchmark(limit=5, model_name="mock")
        print(f"   ✅ Test completed: {results['total_questions']} questions processed")
        print(f"   📊 Average VFE: {results['average_vfe']:.3f}")
        print(f"   🚨 Chaos rate: {results['chaos_detection_rate']:.1%}")
        
        print("\n🎉 All tests passed! Benchmark integration is working.")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
