# How to Use the FEP Cognitive AI System

## 🚀 Quick Start - Basic Usage

### Step 1: Import and Initialize
```python
from fep_cognitive_architecture import FEPCognitiveArchitecture
import numpy as np

# Create your AI agent
ai_agent = FEPCognitiveArchitecture(
    state_dim=10,        # Number of input sensors/observations
    action_dim=5,        # Number of possible actions
    hierarchy_levels=3   # Depth of cognitive processing
)
```

### Step 2: Basic Interaction Loop
```python
# Simple environment interaction
for step in range(100):
    # Get observations from your environment
    observations = np.random.randn(10)  # Replace with real sensor data
    
    # AI processes and responds
    action, free_energy, beliefs = ai_agent.perception_action_cycle(observations)
    
    # Use the action in your environment
    print(f"Step {step}: AI chose action {action}, free energy: {free_energy:.3f}")
```

## 🎯 Practical Use Cases

### 1. Game AI Agent
```python
# Chess/Game AI Example
class GameEnvironment:
    def __init__(self):
        self.board_state = np.zeros(64)  # Chess board
        
    def get_observation(self):
        return self.board_state
        
    def apply_move(self, action):
        # Convert AI action to game move
        move = self.action_to_move(action)
        self.board_state = self.make_move(move)

# Create game AI
game_ai = FEPCognitiveArchitecture(
    state_dim=64,    # Chess board positions
    action_dim=20,   # Possible move types
    hierarchy_levels=4  # Deep strategic thinking
)

# Game loop
game = GameEnvironment()
for turn in range(50):
    board_state = game.get_observation()
    move_action, _, _ = game_ai.perception_action_cycle(board_state)
    game.apply_move(move_action)
    
    print(f"Turn {turn}: AI made move {move_action}")
```

### 2. Robot Control System
```python
# Robot Navigation Example
class RobotSensors:
    def __init__(self):
        self.camera = None
        self.lidar = None
        self.imu = None
    
    def get_sensor_data(self):
        # Combine all sensor inputs
        vision = self.process_camera()      # 10 values
        distance = self.process_lidar()     # 8 values  
        motion = self.process_imu()         # 6 values
        return np.concatenate([vision, distance, motion])

class RobotMotors:
    def execute_action(self, action):
        # Convert AI decision to motor commands
        forward_speed = action[0]
        turn_rate = action[1]
        self.move(forward_speed, turn_rate)

# Create robot AI
robot_brain = FEPCognitiveArchitecture(
    state_dim=24,    # All sensor inputs
    action_dim=2,    # [speed, steering]
    hierarchy_levels=3
)

# Robot operation loop
sensors = RobotSensors()
motors = RobotMotors()

while robot_is_running:
    # Perceive environment
    sensor_data = sensors.get_sensor_data()
    
    # AI decides what to do
    action, uncertainty, beliefs = robot_brain.perception_action_cycle(sensor_data)
    
    # Execute action
    motors.execute_action(action)
    
    # Monitor AI state
    if uncertainty > 0.8:
        print("Robot AI is uncertain - entering careful mode")
```

### 3. Industrial Process Control
```python
# Factory Monitoring Example
class FactoryMonitor:
    def __init__(self):
        self.temperature_sensors = [0] * 10
        self.pressure_sensors = [0] * 5
        self.flow_sensors = [0] * 8
        self.quality_metrics = [0] * 3
    
    def get_process_state(self):
        all_readings = (self.temperature_sensors + 
                       self.pressure_sensors + 
                       self.flow_sensors + 
                       self.quality_metrics)
        return np.array(all_readings)
    
    def apply_control_action(self, action):
        # Adjust process parameters based on AI decision
        valve_adjustments = action[:5]
        heater_adjustments = action[5:8]
        pump_adjustments = action[8:]
        
        self.adjust_valves(valve_adjustments)
        self.adjust_heaters(heater_adjustments)
        self.adjust_pumps(pump_adjustments)

# Create industrial AI
factory_ai = FEPCognitiveArchitecture(
    state_dim=26,    # All sensor readings
    action_dim=10,   # Control outputs
    hierarchy_levels=4  # Complex process understanding
)

# Factory control loop
monitor = FactoryMonitor()
for hour in range(24):  # 24-hour operation
    # Get current process state
    process_state = monitor.get_process_state()
    
    # AI analyzes and decides
    control_action, confidence, prediction = factory_ai.perception_action_cycle(process_state)
    
    # Apply AI's control decision
    monitor.apply_control_action(control_action)
    
    print(f"Hour {hour}: Process optimized, confidence: {confidence:.2f}")
```

## 🔧 Advanced Usage Patterns

### Custom Environment Integration
```python
class YourCustomEnvironment:
    def __init__(self):
        self.state = np.zeros(15)  # Your environment state
    
    def step(self, action):
        # How your environment responds to AI actions
        self.state = self.update_state(action)
        reward = self.calculate_reward()
        done = self.is_terminal()
        return self.state, reward, done
    
    def reset(self):
        self.state = np.random.randn(15)
        return self.state

# Connect AI to your environment
env = YourCustomEnvironment()
ai = FEPCognitiveArchitecture(state_dim=15, action_dim=4)

# Training/Learning loop
state = env.reset()
for episode in range(1000):
    action, free_energy, beliefs = ai.perception_action_cycle(state)
    next_state, reward, done = env.step(action)
    
    if done:
        state = env.reset()
    else:
        state = next_state
    
    # AI automatically learns and adapts!
```

### Real-Time Monitoring
```python
# Monitor AI's internal state
def monitor_ai_health(ai_agent, observations):
    action, free_energy, beliefs = ai_agent.perception_action_cycle(observations)
    
    # Check AI's confidence and state
    mcm_state = ai_agent.meta_monitor.monitor_system(
        ai_agent.generative_model.current_beliefs,
        free_energy,
        ai_agent.current_state
    )
    
    # Respond to AI's internal state
    if mcm_state['system_state'] == SystemState.CRITICAL:
        print("⚠️  AI is struggling - may need intervention")
        
    elif mcm_state['system_state'] == SystemState.LEARNING:
        print("🧠 AI is actively learning new patterns")
        
    elif mcm_state['system_state'] == SystemState.STABLE:
        print("✅ AI is operating normally")
    
    return action, mcm_state
```

## 🎛️ Configuration Options

### Basic Configuration
```python
# Simple AI for basic tasks
simple_ai = FEPCognitiveArchitecture(
    state_dim=5,
    action_dim=2,
    hierarchy_levels=2,
    learning_rate=0.01
)

# Complex AI for sophisticated tasks
complex_ai = FEPCognitiveArchitecture(
    state_dim=50,
    action_dim=20,
    hierarchy_levels=5,
    learning_rate=0.001,
    precision_decay=0.95,
    stability_threshold=0.1
)
```

### Performance Tuning
```python
# For real-time applications (fast response)
realtime_ai = FEPCognitiveArchitecture(
    state_dim=10,
    action_dim=5,
    hierarchy_levels=2,  # Fewer levels = faster
    learning_rate=0.05   # Faster learning
)

# For complex analysis (better decisions)
analytical_ai = FEPCognitiveArchitecture(
    state_dim=100,
    action_dim=50,
    hierarchy_levels=6,  # More levels = deeper analysis
    learning_rate=0.001  # More careful learning
)
```

## 📊 Monitoring and Debugging

### Track AI Performance
```python
def track_ai_performance(ai_agent, observations, true_outcome=None):
    action, free_energy, beliefs = ai_agent.perception_action_cycle(observations)
    
    # Log key metrics
    performance_log = {
        'timestamp': time.time(),
        'free_energy': float(free_energy),
        'action_confidence': float(np.max(beliefs)),
        'prediction_error': float(ai_agent.latest_prediction_error),
        'learning_rate': ai_agent.current_learning_rate
    }
    
    # Optional: Compare with known correct answer
    if true_outcome is not None:
        performance_log['accuracy'] = calculate_accuracy(action, true_outcome)
    
    return action, performance_log
```

### Visualization
```python
import matplotlib.pyplot as plt

def visualize_ai_learning(ai_agent, history):
    """Plot AI's learning progress over time"""
    free_energies = [h['free_energy'] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(free_energies)
    plt.title('AI Learning Progress (Lower = Better)')
    plt.ylabel('Free Energy')
    
    plt.subplot(2, 1, 2)
    plt.plot([h['action_confidence'] for h in history])
    plt.title('AI Confidence Over Time')
    plt.ylabel('Confidence')
    plt.xlabel('Time Steps')
    
    plt.tight_layout()
    plt.show()
```

## 🚀 Deployment Strategies

### 1. Development/Testing
```python
# Safe testing environment
test_ai = FEPCognitiveArchitecture(state_dim=10, action_dim=3)

# Generate test data
test_observations = [np.random.randn(10) for _ in range(100)]

# Test AI responses
for i, obs in enumerate(test_observations):
    action, fe, beliefs = test_ai.perception_action_cycle(obs)
    print(f"Test {i}: Action {action}, Free Energy: {fe:.3f}")
```

### 2. Production Deployment
```python
class ProductionAIWrapper:
    def __init__(self, state_dim, action_dim):
        self.ai = FEPCognitiveArchitecture(state_dim, action_dim)
        self.performance_history = []
        self.error_count = 0
        
    def safe_decision(self, observations):
        try:
            action, fe, beliefs = self.ai.perception_action_cycle(observations)
            
            # Safety checks
            if fe > 10.0:  # High uncertainty
                return self.fallback_action(observations)
            
            self.performance_history.append(fe)
            return action
            
        except Exception as e:
            self.error_count += 1
            print(f"AI Error: {e}")
            return self.emergency_action()
    
    def fallback_action(self, observations):
        # Conservative action when AI is uncertain
        return np.zeros(self.ai.action_dim)
    
    def emergency_action(self):
        # Safe action when AI fails
        return np.zeros(self.ai.action_dim)

# Deploy safely
production_ai = ProductionAIWrapper(state_dim=20, action_dim=8)
```

## 🎯 Best Practices

### 1. Start Simple
```python
# Begin with small dimensions
starter_ai = FEPCognitiveArchitecture(
    state_dim=5,     # Start small
    action_dim=2,    # Simple actions
    hierarchy_levels=2  # Basic thinking
)

# Gradually increase complexity as needed
```

### 2. Monitor Performance
```python
# Always track how well the AI is doing
def run_with_monitoring(ai, observations):
    action, free_energy, beliefs = ai.perception_action_cycle(observations)
    
    # Check for problems
    if free_energy > 5.0:
        print("⚠️  AI is confused - check your input data")
    
    return action
```

### 3. Validate Inputs
```python
def safe_ai_interaction(ai, raw_observations):
    # Clean and validate input
    observations = np.array(raw_observations)
    
    # Check for valid input
    if np.any(np.isnan(observations)):
        print("❌ Invalid input detected")
        return None
    
    # Normalize if needed
    observations = observations / np.linalg.norm(observations)
    
    return ai.perception_action_cycle(observations)
```

## 🔄 Next Steps

1. **Choose your application domain** (gaming, robotics, control, etc.)
2. **Define your state and action spaces** (what the AI perceives and does)
3. **Create your environment interface** (how AI connects to your system)
4. **Start with simple tests** (basic input/output validation)
5. **Gradually increase complexity** (more sensors, actions, hierarchy)
6. **Monitor and tune performance** (track free energy and confidence)

The AI will automatically learn and adapt to whatever environment you give it! 🧠✨
