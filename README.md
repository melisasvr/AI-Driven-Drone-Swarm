# AI-Driven Drone Swarm 
- A sophisticated drone swarm simulation that uses PyTorch neural networks to train individual drones for autonomous formation flying, obstacle avoidance, and collective behavior.
- Each drone is equipped with its own AI brain that learns through reinforcement learning to achieve complex swarm behaviors.

## 🚁 Features
### Core Capabilities
- **Individual AI Brains**: Each drone has its own neural network, making autonomous decisions
- **Formation Flying**: Support for sphere, line, triangle, and custom formations
- **Obstacle Avoidance**: Dynamic collision avoidance using AI-driven decision making
- **Real-time Learning**: Drones improve their performance through continuous training
- **3D Visualization**: Interactive web-based visualization of the trained swarm

### Technical Highlights
- PyTorch-based neural networks with proper weight initialization
- Epsilon-greedy exploration strategy with adaptive decay
- Multi-objective reward system (formation, collision avoidance, efficiency)
- GPU acceleration support for faster training
- Real-time performance metrics and training history

## 📋 Requirements

### Python Dependencies
```
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.5.0
```

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)
- Modern web browser (for visualization)

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ai-drone-swarm-pytorch
```

2. **Install dependencies**:
```bash
pip install torch numpy matplotlib
```

3. **Verify PyTorch installation**:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 🎯 Quick Start
### 1. Train the Drone Swarm
- Run the main training script:

```bash
python drone_swarm_pytorch.py
```

- The training process will:
- Initialize 12 drones with individual neural networks
- Train through 8 episodes with different formations
- Display real-time metrics during training
- Save trained models and simulation data

### 2. Visualize Results
- Open the visualization in your web browser:

```bash
# Simply open index.html in your browser
open index.html  # macOS
# or double-click index.html on Windows/Linux
```

Then:
1. Click "Load Training Data"
2. Select the generated `simulation_data_final.json` file
3. Click "Play" to watch your trained drones in action

## 📊 Understanding the Training
### Neural Network Architecture
Each drone uses a 3-layer neural network:
- **Input Layer**: 10 neurons (position, velocity, target direction, distance)
- **Hidden Layers**: 32 neurons each with ReLU activation
- **Output Layer**: 3 neurons (x, y, z acceleration commands)

### Reward System
Drones learn through a multi-component reward function:
- **Formation Reward**: +10 points for moving closer to the target position
- **Proximity Bonus**: +5 points for being within 2 units of the target
- **Collision Penalty**: -20 points for hitting obstacles
- **Separation Reward**: Maintains safe distances between drones
- **Efficiency Bonus**: Rewards smooth, purposeful movement

### Training Parameters
- **Learning Rate**: 0.01 with Adam optimizer
- **Exploration**: Starts at 30%, decays to 5%
- **Episodes**: 8 training episodes with varying formations
- **Steps per Episode**: 250-300 simulation steps

## 🎮 Visualization Controls
### Mouse Controls
- **Drag**: Rotate the camera around the swarm
- **Scroll**: Zoom in/out
- **Double-click**: Reset camera view

### Interface Controls
- **Play/Pause**: Control simulation playback
- **Speed Slider**: Adjust simulation speed (0.1x to 3.0x)
- **Trail Length**: Control drone path visualization
- **Formation Buttons**: Switch between formations in real-time

### Keyboard Shortcuts
- **Spacebar**: Play/Pause toggle
- **R**: Reset simulation
- **1-4**: Quick formation switching

## 📈 Performance Metrics
- The system tracks several key performance indicators:

### Fitness Score
- Measures overall drone performance
- Combines formation accuracy, collision avoidance, and efficiency
- Higher scores indicate better AI performance

### Formation Score
- Evaluates how well drones maintain target formation
- Scale of 0-10 (10 = perfect formation)
- Real-time calculation during visualization

### Collision Count
- Tracks obstacles, hits, and drone-to-drone collisions
- Goal is zero collisions while maintaining formation

### Target Distance
- Average distance from drones to their target positions
- Lower values indicate better formation accuracy

## 🔧 Configuration

### Modify Drone Count
```python
# In drone_swarm_pytorch.py
sim = DroneSwarmSimulation(num_drones=15, num_obstacles=8)  # Change from 12 to 15
```

### Adjust Training Parameters
```python
# In Drone class __init__
self.epsilon = 0.5        # Initial exploration rate
self.epsilon_decay = 0.995 # Exploration decay rate
self.max_speed = 2.0      # Maximum drone speed
```

### Add Custom Formations
```python
# In get_target_position method
elif formation['type'] == 'custom':
    # Your custom formation logic here
    return np.array([x, y, z])
```

## 📁 File Structure

```
ai-drone-swarm/
├── drone_swarm.py             # Main training script
├── index.html                 # 3D visualization interface
├── trained_drone_swarm_v2.pth # Saved neural network weights
├── simulation_data_final.json # Export data for visualization
├── training_progress.png      # Generated performance plots
└── README.md                  # This file
```

## 🧠 How It Works

### 1. Initialization
- Each drone spawns with a random neural network
- Obstacles are randomly placed in the environment
- Target formation is selected (sphere, line, triangle, etc.)

### 2. Simulation Loop
- Drones observe their environment (position, velocity, neighbors, obstacles)
- Neural networks process observations and output acceleration commands
- Physics engine updates drone positions and detects collisions
- Reward signals are calculated based on performance

### 3. Learning Process
- Drones receive rewards/penalties for their actions
- Neural networks are updated using supervised learning approaches
- Exploration rate gradually decreases as training progresses
- Performance metrics are logged for analysis

### 4. Formation Behavior
- Drones learn to navigate toward formation target positions
- Obstacle avoidance emerges from negative collision rewards
- Swarm cohesion develops through neighbor proximity rewards
- Complex behaviors emerge from simple reward structures

## 🎨 Customization Options

### Visual Themes
Modify the visualization colors and effects in `index.html`:
```javascript
// Change drone colors
const bodyMaterial = new THREE.MeshLambertMaterial({ 
    color: new THREE.Color().setHSL(your_hue, saturation, lightness)
});
```

### Training Scenarios
Create custom training scenarios:
```python
# Different obstacle layouts
def create_maze_obstacles(self):
    # Create maze-like obstacle course
    pass

# Custom reward functions
def calculate_custom_reward(self, ...):
    # Implement task-specific rewards
    pass
```

## 🐛 Troubleshooting

### Common Issues
**Training is slow**:
- Enable CUDA if you have a compatible GPU
- Reduce the number of drones or training episodes
- Decrease episode step count

**Drones not learning**:
- Check that the reward function is balanced
- Verify neural network architecture is appropriate
- Ensure exploration rate isn't too low or high

**Visualization not loading**:
- Make sure you've generated `simulation_data_final.json`
- Check browser console for JavaScript errors
- Try a different web browser

**Poor formation quality**:
- Increase training episodes
- Adjust reward function weights
- Verify target position calculations

## 📚 Advanced Usage

### Batch Training
Train multiple swarms with different parameters:
```python
for params in parameter_combinations:
    sim = DroneSwarmSimulation(**params)
    sim.run_episode(steps=500)
    sim.save_model(f'model_{params}.pth')
```

### Custom Environments
Create specialized training environments:
```python
class MazeEnvironment(DroneSwarmSimulation):
    def __init__(self):
        super().__init__()
        self.create_maze_obstacles()
        
    def create_maze_obstacles(self):
        # Custom obstacle layout
        pass
```

### Performance Analysis
Export detailed metrics for analysis:
```python
# Generate comprehensive training reports
sim.export_training_analytics('training_report.json')
sim.plot_detailed_metrics()
```

## 🤝 Contributing
- We welcome contributions! Areas for improvement:
- Additional formation types
- More sophisticated neural network architectures
- Enhanced visualization features
- Performance optimizations
- Multi-objective optimization algorithms

## 📄 License
- This project is open source. Please check the LICENSE file for details.

## 🔗 Related Projects
- **OpenAI Gym**: Reinforcement learning environments
- **PyTorch Lightning**: Simplified PyTorch training
- **Three.js**: 3D visualization library used in the web interface

## ⚡ Performance Tips
1. **Use GPU acceleration** when available for faster training
2. **Adjust batch sizes** based on your system's memory
3. **Monitor training progress** and stop early if performance plateaus
4. **Experiment with hyperparameters** for your specific use case
5. **Save checkpoints regularly** during long training sessions

## 📞 Support
- If you encounter issues or have questions:
1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Experiment with different parameters
4. Consider the specific requirements of your use case

---

**Happy Swarming!** 🚁✨
