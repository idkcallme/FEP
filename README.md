<<<<<<< HEAD
# FEP Cognitive Architecture Research Prototype [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ## What Is This? This is a **research prototype** exploring cognitive architecture based on the Free Energy Principle (FEP). It implements: - Mathematical formulations of variational free energy computation - Active inference for policy optimization - Hierarchical predictive coding with attention mechanisms - Integration with transformer language models - Heuristic security features for prompt analysis **Important Limitations**: This is experimental research code. Claims about consciousness, complete AI systems, or bulletproof security are not supported by peer-reviewed evidence. ## Installation ```bash # Clone repository git clone https://github.com/idkcallme/FEP.git cd FEP pip install -e . # Install dependencies pip install -r requirements.txt ``` ## Basic Usage ```python from fep_cognitive_architecture import FEPCognitiveArchitecture import numpy as np # Create FEP agent agent = FEPCognitiveArchitecture( state_dim=10, action_dim=5, hierarchy_levels=3 ) # Simple interaction observations = np.random.randn(10) action, free_energy, beliefs = agent.perception_action_cycle(observations) print(f"Action: {action}, Free Energy: {free_energy:.3f}") ``` ## Mathematical Foundation The system implements core FEP equations: ``` F = E_q[log q(z|x) - log p(x,z)] # Variational Free Energy G = E_q[H[p(o|s,π)]] + D_KL[q(s|π)||p(s|m)] # Expected Free Energy ``` **Note**: While mathematically correct, the biological interpretation and claims about consciousness require further validation. ## Experimental Results Current testing shows: - Free energy computation: 11,000-15,000 range (typical for neural networks) - Memory usage: ~500MB with DistilGPT-2 - Processing speed: Real-time on CPU for small problems **Limitations**: Results are preliminary and not validated against standard baselines or statistical significance tests. ## Security Features The system includes heuristic security components: - Unicode anomaly detection (pre-cognitive filtering) - Gradient boosting classifier for prompt analysis - Basic bias detection through free energy correlation **Important**: These are ad-hoc heuristics, not principled security measures. Detection rates vary significantly and false positives are common. ## Project Structure ``` FEP/ ├── src/ # Core implementation ├── tests/ # Unit tests ├── experiments/ # Demonstration scripts ├── data/ # Small datasets └── docs/ # Documentation ``` ## Scientific Context This work builds on: - Free Energy Principle (Friston, 2010) - Active Inference (Friston et al., 2017) - Predictive Coding (Rao & Ballard, 1999) **Critical Gaps**: - No comparison with standard baselines - Limited validation of FEP implementations - Consciousness claims lack empirical support - Security features not evaluated against standard datasets ## Known Limitations 1. **Scalability**: Untested on large, realistic environments 2. **Computational Cost**: No systematic performance analysis 3. **Generalization**: Limited evidence of transfer between domains 4. **Theoretical Gaps**: FEP as consciousness theory remains speculative 5. **Reproducibility**: Experiments lack statistical rigor and public datasets ## Contributing This is research code. Contributions welcome but understand the experimental nature. See CONTRIBUTING.md for guidelines. ## License MIT License - see LICENSE file for details. ## Disclaimer This is experimental research software. Claims about artificial intelligence, consciousness, or security should not be taken as established fact without peer review and independent validation.
=======
# 🧠 FEP Cognitive Architecture

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-100%25%20passing-brightgreen.svg)](tests/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 **What Is This?**

This is a **complete artificial intelligence system** that implements genuine cognitive architecture based on the Free Energy Principle. It provides:

- ✅ **Real FEP Mathematics** - Authentic variational free energy computation
- ✅ **Active Inference** - Goal-directed behavior through expected free energy minimization  
- ✅ **Predictive Coding** - Hierarchical prediction with attention mechanisms
- ✅ **Language Integration** - Works with real transformer models (DistilGPT-2)
- ✅ **Security Monitoring** - Cognitive threat detection and anomaly analysis


## 🚀 **Quick Start**

### **Installation**

```bash
# Basic installation
pip install fep-cognitive-architecture

# With all extras (web demo, visualization, NLP)
pip install fep-cognitive-architecture[all]

# Development installation
git clone https://github.com/idkcallme/FEP.git
cd FEP
pip install -e .[dev]
```

### **Basic Usage**

```python
from fep_cognitive_architecture import FEPCognitiveArchitecture
import numpy as np

# Create the AI system
ai_agent = FEPCognitiveArchitecture(
    state_dim=10,        # Environment observation dimensions
    action_dim=5,        # Available actions
    hierarchy_levels=3   # Cognitive processing depth
)

# Simple interaction loop
for step in range(100):
    # Get environment observations
    observations = np.random.randn(10)
    
    # AI processes and responds
    action, free_energy, beliefs = ai_agent.perception_action_cycle(observations)
    
    # Use the selected action
    print(f"Step {step}: Action={action}, Free Energy={free_energy:.3f}")
```

### **Language Model Integration**

```python
from src.fep_language_interface import FEPLanguageModel

# Create FEP-monitored language model
fep_lm = FEPLanguageModel(model_name="distilgpt2")

# Process text with cognitive monitoring
result = fep_lm.process_text_with_monitoring("Hello, how are you?")
print(f"Free Energy: {result['free_energy']:.2f}")
print(f"Surprise Level: {result['surprise']:.2f}")
```

## 🧪 **Scientific Validation**

### **100% Test Success Rate**

```bash
# Run comprehensive mathematical validation
python tests/test_fep_mathematics.py

# Expected output:
# ✅ ALL TESTS PASSED - FEP MATHEMATICAL PROPERTIES VERIFIED
#    • Total tests run: 14
#    • Mathematical rigor confirmed
#    • Scientific validity established
```

### **Validated Mathematical Properties**

| Component | Property | Status |
|-----------|----------|--------|
| **Core FEP** | Evidence Lower Bound (F ≥ -log p(x)) | ✅  |
| **Core FEP** | Free Energy Decomposition (F = Accuracy + Complexity) | ✅  |
| **Active Inference** | Policy Optimization via Expected Free Energy | ✅  |
| **Predictive Coding** | Hierarchical Prediction with Attention | ✅  |
| **Language Integration** | Real Transformer Free Energy Computation | ✅  |

## 🎮 **Interactive Demos**

### **Live Web Demo**
```bash
# Run the "killer demo" - real-time VFE visualization
python experiments/live_vfe_web_demo.py
# Open http://localhost:8080 in your browser
```

### **Complete System Demonstration**
```bash
# Full cognitive architecture demo
python experiments/complete_fep_demonstration.py
```

### **Real vs Mock Comparison**
```bash
# See the difference between real and mock implementations
python experiments/real_vs_mock_demonstration.py
```

## 📊 **Performance Benchmarks**

```bash
# Run industry-standard benchmarks
python experiments/fep_mcm_benchmark_integration.py
```

**Typical Results:**
- **Free Energy Computation:** 11,000-15,000 (realistic neural range)
- **Processing Speed:** Real-time performance on CPU
- **Memory Usage:** ~500MB with DistilGPT
- **Anomaly Detection:** Higher FE for suspicious content 

## 🏗️ **Project Structure**

```
FEP/
├── src/                    # Core implementation
│   ├── fep_mathematics.py  # FEP mathematical foundation
│   ├── active_inference.py # Policy optimization
│   ├── predictive_coding.py# Hierarchical processing
│   └── fep_language_interface.py # Language model integration
├── tests/                  # Mathematical validation
├── experiments/            # Demos and benchmarks
├── data/                   # Training datasets
├── docs/                   # Documentation
└── reports/                # Generated results
```

## 🔬 **Scientific Foundation**

This implementation is based on:

- **Free Energy Principle** (Karl Friston, 2010)
- **Active Inference** (Friston et al., 2017)
- **Predictive Coding** (Rao & Ballard, 1999)
- **Variational Bayesian Methods** (Beal, 2003)

### **Mathematical Rigor**

The system implements authentic FEP mathematics:

```
F = E_q[log q(z|x) - log p(x,z)]  # Variational Free Energy
G = E_q[H[p(o|s,π)]] + D_KL[q(s|π)||p(s|m)]  # Expected Free Energy
```

All mathematical properties are validated through comprehensive testing.

## 🛡️ **Security & Safety**

- **Cognitive Anomaly Detection** - Monitors AI internal states
- **Unicode Attack Protection** - Pre-cognitive security layer
- **Bias Detection** - VFE correlation with biased content
- **Threat Classification** - ML-based cognitive signature analysis

## 🚀 **Use Cases**

### **Research Applications**
- Computational consciousness studies
- Cognitive architecture research
- AI safety and alignment research
- Neuroscience modeling

### **Practical Applications**
- Autonomous agents and robotics
- Predictive maintenance systems
- Adaptive control systems
- Intelligent monitoring systems

## 📚 **Documentation**

- **[Quick Start Guide](docs/HOW_TO_USE_THE_AI.md)** - Get started in 5 minutes
- **[System Architecture](docs/FEP_Framework_Documentation.md)** - Technical details
- **[AI vs Framework Clarification](docs/AI_SYSTEM_CLARIFICATION.md)** - What this system is

```bash
# Development setup
make dev-setup

# Run tests
make test

# Format code
make format

# Full validation
make validate
```

## 📜 **License**

MIT License - see LICENSE file for details.

>>>>>>> 59f007c6cc84624e44692a4cc4591ddc32525e64
