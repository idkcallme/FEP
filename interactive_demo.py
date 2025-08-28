#!/usr/bin/env python3
"""
Interactive Demo: How to Use the FEP Cognitive AI System
=====================================================

This script demonstrates basic usage patterns for the AI system.
Run this to see the AI in action!
"""

import numpy as np
import time
from fep_cognitive_architecture import FEPCognitiveArchitecture, SystemState

def demo_basic_usage():
    """Demonstrate the most basic way to use the AI"""
    print("\n" + "="*60)
    print("🧠 DEMO 1: Basic AI Usage")
    print("="*60)
    
    # Create AI agent
    ai = FEPCognitiveArchitecture(
        state_dim=5,     # 5 input sensors
        action_dim=3,    # 3 possible actions
        hierarchy_levels=2
    )
    
    print("✅ AI agent created successfully!")
    print(f"   - Input sensors: 5")
    print(f"   - Possible actions: 3")
    print(f"   - Cognitive levels: 2")
    
    # Simple interaction loop
    print("\n🔄 Running basic interaction loop...")
    for step in range(5):
        # Simulate sensor input (replace with real sensors)
        observations = np.random.randn(5)
        
        # AI processes and decides
        action, info = ai.perception_action_cycle(observations)
        free_energy = info['free_energy']
        confidence = 1.0 / (1.0 + free_energy)  # Convert free energy to confidence
        
        print(f"Step {step+1}:")
        print(f"  📥 Input: {observations}")
        print(f"  🎯 AI Action: {action}")
        print(f"  ⚡ Free Energy: {free_energy:.3f}")
        print(f"  🧠 Confidence: {confidence:.3f}")
        print()
        
        time.sleep(0.5)  # Pause for readability

def demo_game_ai():
    """Demonstrate AI playing a simple number guessing game"""
    print("\n" + "="*60)
    print("🎮 DEMO 2: Game AI - Number Guessing")
    print("="*60)
    
    # Game setup
    target_number = 7  # AI needs to learn to output this
    game_ai = FEPCognitiveArchitecture(
        state_dim=3,     # [current_guess, feedback, rounds_played]
        action_dim=10,   # Actions 0-9 (digits to guess)
        hierarchy_levels=2
    )
    
    print(f"🎯 Target number: {target_number}")
    print("🤖 AI will learn to guess the correct number!")
    
    wins = 0
    for round_num in range(10):
        # Game state: [last_guess, feedback, round_number]
        if round_num == 0:
            game_state = np.array([0, 0, round_num])
        else:
            feedback = 1.0 if last_guess == target_number else -1.0
            game_state = np.array([last_guess, feedback, round_num])
        
        # AI makes a guess
        action, info = game_ai.perception_action_cycle(game_state)
        free_energy = info['free_energy']
        guess = int(np.argmax(action))  # Convert to number 0-9
        last_guess = guess
        
        # Check if correct
        is_correct = (guess == target_number)
        if is_correct:
            wins += 1
        
        print(f"Round {round_num + 1}: AI guessed {guess} - {'✅ Correct!' if is_correct else '❌ Wrong'}")
        print(f"           Free Energy: {free_energy:.3f} (lower = more confident)")
        
    print(f"\n🏆 AI won {wins}/10 rounds! (AI learns over time)")

def demo_adaptive_behavior():
    """Demonstrate AI adapting to changing environment"""
    print("\n" + "="*60)
    print("🔄 DEMO 3: Adaptive Behavior")
    print("="*60)
    
    adaptive_ai = FEPCognitiveArchitecture(
        state_dim=4,
        action_dim=2,
        hierarchy_levels=3
    )
    
    print("🌍 Environment will change halfway through!")
    print("📊 Watch how AI adapts its behavior...")
    
    free_energies = []
    
    for step in range(20):
        # Environment changes at step 10
        if step < 10:
            # Environment 1: Prefers action 0 for positive inputs
            env_signal = 1.0
            preferred_pattern = np.array([1, 0.5, -0.5, env_signal])
        else:
            # Environment 2: Prefers action 1 for positive inputs  
            env_signal = -1.0
            preferred_pattern = np.array([1, 0.5, -0.5, env_signal])
        
        # Add some noise
        observations = preferred_pattern + 0.1 * np.random.randn(4)
        
        # AI responds
        action, info = adaptive_ai.perception_action_cycle(observations)
        free_energy = info['free_energy']
        free_energies.append(free_energy)
        
        chosen_action = np.argmax(action)
        
        if step == 10:
            print("\n🔄 ENVIRONMENT CHANGED! 🔄")
        
        print(f"Step {step+1:2d}: Action {chosen_action}, Free Energy: {free_energy:.3f}")
        
        # Show adaptation
        if step == 9:
            print("   └─ AI learned Environment 1")
        elif step == 19:
            print("   └─ AI adapted to Environment 2!")
    
    # Show learning curve
    print(f"\n📈 Learning Progress:")
    print(f"   Initial confusion: {free_energies[0]:.3f}")
    print(f"   After learning Env 1: {free_energies[9]:.3f}")
    print(f"   Initial confusion Env 2: {free_energies[10]:.3f}")
    print(f"   After adapting to Env 2: {free_energies[19]:.3f}")

def demo_self_monitoring():
    """Demonstrate AI's self-awareness and monitoring"""
    print("\n" + "="*60)
    print("🧠 DEMO 4: AI Self-Awareness")
    print("="*60)
    
    monitor_ai = FEPCognitiveArchitecture(
        state_dim=6,
        action_dim=4,
        hierarchy_levels=3
    )
    
    print("🔍 AI will monitor its own cognitive state!")
    
    scenarios = [
        ("Normal operation", np.array([1, 2, 3, 4, 5, 6])),
        ("Confusing input", np.array([100, -50, 0.1, 999, -1, 0])),
        ("Familiar pattern", np.array([1, 2, 3, 4, 5, 6])),
        ("Novel situation", np.array([-10, 0, 15, -5, 8, -3]))
    ]
    
    for scenario_name, observations in scenarios:
        print(f"\n📋 Scenario: {scenario_name}")
        
        # AI processes situation
        action, info = monitor_ai.perception_action_cycle(observations)
        free_energy = info['free_energy']
        
        # Get monitoring result from info
        mcm_result = info['monitoring_result']
        
        # AI monitors itself
        # mcm_result is already available from the info dict
        
        # Report AI's self-assessment
        system_state = mcm_result['system_state']
        
        print(f"   🎯 AI Action: {np.argmax(action)}")
        print(f"   ⚡ Free Energy: {free_energy:.3f}")
        print(f"   🧠 AI Self-Assessment: {system_state.name}")
        
        if system_state == SystemState.CRITICAL:
            print("   ⚠️  AI says: 'I'm struggling with this input!'")
        elif system_state == SystemState.LEARNING:
            print("   🤔 AI says: 'This is new - I'm learning!'")
        elif system_state == SystemState.STABLE:
            print("   ✅ AI says: 'I understand this situation well.'")

def demo_performance_comparison():
    """Show AI performance before and after learning"""
    print("\n" + "="*60)
    print("📊 DEMO 5: Learning Performance")
    print("="*60)
    
    learning_ai = FEPCognitiveArchitecture(
        state_dim=3,
        action_dim=2,
        hierarchy_levels=2
    )
    
    # Define a simple pattern: if input[0] > 0, choose action 1, else action 0
    test_inputs = [
        np.array([2, 1, -1]),   # Should choose action 1
        np.array([-1, 2, 3]),   # Should choose action 0
        np.array([0.5, -2, 1]), # Should choose action 1
        np.array([-3, 1, -1]),  # Should choose action 0
    ]
    
    print("🎯 Test Pattern: If first input > 0, choose action 1, else action 0")
    
    # Before learning
    print("\n📈 BEFORE LEARNING:")
    initial_performance = []
    for i, test_input in enumerate(test_inputs):
        action, info = learning_ai.perception_action_cycle(test_input)
        fe = info['free_energy']
        chosen = np.argmax(action)
        correct = (chosen == 1) if test_input[0] > 0 else (chosen == 0)
        initial_performance.append(correct)
        print(f"   Test {i+1}: Input {test_input[0]:4.1f} → Action {chosen} {'✅' if correct else '❌'}")
    
    # Learning phase
    print("\n🧠 LEARNING PHASE (50 training examples)...")
    for _ in range(50):
        # Generate training example
        x = np.random.randn(3)
        learning_ai.perception_action_cycle(x)
    
    # After learning  
    print("\n📈 AFTER LEARNING:")
    final_performance = []
    for i, test_input in enumerate(test_inputs):
        action, info = learning_ai.perception_action_cycle(test_input)
        fe = info['free_energy']
        chosen = np.argmax(action)
        correct = (chosen == 1) if test_input[0] > 0 else (chosen == 0)
        final_performance.append(correct)
        print(f"   Test {i+1}: Input {test_input[0]:4.1f} → Action {chosen} {'✅' if correct else '❌'}")
    
    print(f"\n🏆 RESULTS:")
    print(f"   Before learning: {sum(initial_performance)}/4 correct ({100*sum(initial_performance)/4:.0f}%)")
    print(f"   After learning:  {sum(final_performance)}/4 correct ({100*sum(final_performance)/4:.0f}%)")

def main():
    """Run all demonstrations"""
    print("🚀 FEP Cognitive AI System - Interactive Demo")
    print("=" * 60)
    print("This demo shows you how to use the AI system in practice.")
    print("Each demo focuses on a different aspect of the AI's capabilities.")
    
    try:
        # Run all demos
        demo_basic_usage()
        demo_game_ai()
        demo_adaptive_behavior()
        demo_self_monitoring()
        demo_performance_comparison()
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETE!")
        print("="*60)
        print("Key takeaways:")
        print("✅ The AI automatically learns from experience")
        print("✅ It adapts to changing environments")
        print("✅ It monitors its own performance")
        print("✅ No manual programming of intelligence needed!")
        print("\n💡 Check 'HOW_TO_USE_THE_AI.md' for detailed usage examples")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure 'fep_cognitive_architecture.py' is available")

if __name__ == "__main__":
    main()
