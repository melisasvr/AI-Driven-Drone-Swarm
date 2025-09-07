import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import time
from collections import deque
import random

class DroneNeuralNetwork(nn.Module):
    """Neural network for individual drone decision making"""
    def __init__(self, input_size=10, hidden_size=32, output_size=3):
        super(DroneNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights properly
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output in range [-1, 1]
        return x

class Drone:
    """Individual drone with AI brain"""
    def __init__(self, drone_id, position=None, device='cpu'):
        self.id = drone_id
        self.device = device
        
        # Physics
        if position is None:
            # Start in a smaller area to reduce initial chaos
            self.position = np.random.uniform(-5, 5, 3)
            self.position[1] = np.random.uniform(2, 8)  # Keep above ground
        else:
            self.position = position.copy()
            
        self.velocity = np.random.uniform(-0.1, 0.1, 3)  # Small initial velocity
        self.acceleration = np.zeros(3)
        self.max_speed = 1.5
        self.max_acceleration = 0.3
        
        # AI Brain
        self.brain = DroneNeuralNetwork().to(device)
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.01, weight_decay=1e-5)
        
        # Performance tracking
        self.fitness = 0.0
        self.episode_fitness = 0.0  # Reset each episode
        self.collisions = 0
        self.episode_collisions = 0
        self.steps_alive = 0
        
        # Learning parameters
        self.epsilon = 0.3  # Lower initial exploration
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.05
        
    def reset_episode(self):
        """Reset drone for new episode"""
        # Reset position near center
        self.position = np.random.uniform(-3, 3, 3)
        self.position[1] = np.random.uniform(2, 6)
        self.velocity = np.random.uniform(-0.1, 0.1, 3)
        self.acceleration = np.zeros(3)
        
        # Reset episode stats
        self.episode_fitness = 0.0
        self.episode_collisions = 0
        self.steps_alive = 0
        
    def get_state(self, other_drones, obstacles, target_formation):
        """Get current state vector for neural network input"""
        state = []
        
        # Normalize position to [-1, 1] range
        normalized_pos = np.clip(self.position / 20.0, -1, 1)
        state.extend(normalized_pos)
        
        # Normalize velocity
        normalized_vel = np.clip(self.velocity / self.max_speed, -1, 1)
        state.extend(normalized_vel)
        
        # Target direction and distance (normalized)
        target_pos = self.get_target_position(target_formation)
        target_diff = target_pos - self.position
        target_distance = np.linalg.norm(target_diff)
        
        if target_distance > 0:
            target_direction = target_diff / target_distance
        else:
            target_direction = np.zeros(3)
            
        state.extend(target_direction)
        state.append(np.clip(target_distance / 15.0, 0, 1))  # Normalized distance
        
        return np.array(state, dtype=np.float32)
    
    def get_target_position(self, formation):
        """Calculate target position based on formation type"""
        if formation['type'] == 'sphere':
            angle = (self.id / formation['count']) * 2 * np.pi
            radius = formation.get('radius', 6.0)
            height_offset = np.cos(angle * 0.7) * 2
            return np.array([
                np.cos(angle) * radius,
                5 + height_offset,  # Keep at reasonable height
                np.sin(angle) * radius
            ])
        elif formation['type'] == 'line':
            spacing = formation.get('spacing', 2.5)
            return np.array([
                (self.id - formation['count'] / 2) * spacing,
                5,  # Fixed height
                0
            ])
        elif formation['type'] == 'triangle':
            # Better triangle formation
            layer_size = int(np.ceil((-1 + np.sqrt(1 + 8 * self.id)) / 2))
            pos_in_layer = self.id - (layer_size * (layer_size - 1)) // 2
            spacing = 2.0
            
            x = (pos_in_layer - layer_size / 2) * spacing
            z = layer_size * spacing * 0.8
            return np.array([x, 5, z])
        else:  # Default to sphere
            return self.get_target_position({'type': 'sphere', 'count': formation['count'], 'radius': 6.0})
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy with neural network"""
        if np.random.random() < self.epsilon:
            # Random exploration - but not completely random
            return np.random.uniform(-0.5, 0.5, 3)
        else:
            # Neural network decision
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.brain(state_tensor)
                return action.cpu().numpy()[0]
    
    def update(self, other_drones, obstacles, target_formation, dt=0.1):
        """Update drone state"""
        old_position = self.position.copy()
        
        # Get current state and choose action
        current_state = self.get_state(other_drones, obstacles, target_formation)
        action = self.choose_action(current_state)
        
        # Apply physics with limits
        self.acceleration = np.clip(action * self.max_acceleration, -self.max_acceleration, self.max_acceleration)
        self.velocity += self.acceleration * dt
        
        # Limit velocity
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
        
        # Update position
        self.position += self.velocity * dt
        
        # Boundary enforcement (soft walls)
        boundary = 20.0
        for i in range(3):
            if abs(self.position[i]) > boundary:
                self.position[i] = np.sign(self.position[i]) * boundary
                self.velocity[i] *= -0.5  # Bounce back with energy loss
        
        # Keep above ground
        if self.position[1] < 0.5:
            self.position[1] = 0.5
            self.velocity[1] = abs(self.velocity[1]) * 0.5
        
        # Calculate and apply reward
        reward = self.calculate_reward(other_drones, obstacles, target_formation, old_position)
        self.episode_fitness += reward
        self.fitness += reward
        self.steps_alive += 1
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return current_state, action, reward
    
    def calculate_reward(self, other_drones, obstacles, target_formation, old_position):
        """Calculate reward for current state/action"""
        reward = 0.0
        
        # Formation reward (main objective)
        target_pos = self.get_target_position(target_formation)
        target_distance = np.linalg.norm(self.position - target_pos)
        old_target_distance = np.linalg.norm(old_position - target_pos)
        
        # Reward getting closer to target
        distance_improvement = old_target_distance - target_distance
        reward += distance_improvement * 10
        
        # Bonus for being close to target
        if target_distance < 2.0:
            reward += 5.0
        elif target_distance < 5.0:
            reward += 2.0
        
        # Penalty for being far from target
        if target_distance > 15.0:
            reward -= 2.0
        
        # Obstacle avoidance
        collision_occurred = False
        for obstacle in obstacles:
            dist = np.linalg.norm(self.position - obstacle['position'])
            obstacle_radius = obstacle.get('radius', 1.5)
            
            if dist < obstacle_radius + 0.5:  # Collision
                reward -= 20.0
                self.episode_collisions += 1
                self.collisions += 1
                collision_occurred = True
            elif dist < obstacle_radius + 2.0:  # Close call
                reward -= (obstacle_radius + 2.0 - dist) * 2
        
        # Drone separation
        for other in other_drones:
            if other.id != self.id:
                dist = np.linalg.norm(self.position - other.position)
                if dist < 1.0:  # Too close
                    reward -= 5.0
                elif dist < 2.0:
                    reward -= (2.0 - dist)
        
        # Movement efficiency
        speed = np.linalg.norm(self.velocity)
        if speed > 0.1:  # Reward movement
            reward += 0.5
        else:  # Small penalty for staying still
            reward -= 0.1
        
        # Survival bonus
        reward += 0.1
        
        return reward
    
    def train_step(self, state, action, reward, next_state):
        """Single training step"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        
        # Predict current action
        predicted_action = self.brain(state_tensor)
        
        # Simple supervised learning approach
        # Target: move towards the action that gave positive reward
        if reward > 0:
            target_action = action_tensor
        else:
            # If negative reward, try opposite or modified action
            target_action = action_tensor * 0.5  # Reduce magnitude
        
        # Calculate loss
        loss = nn.MSELoss()(predicted_action, target_action)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 0.5)
        self.optimizer.step()
        
        return loss.item()

class DroneSwarmSimulation:
    """Main simulation class managing the drone swarm"""
    def __init__(self, num_drones=10, num_obstacles=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize drones
        self.drones = []
        for i in range(num_drones):
            drone = Drone(i, device=self.device)
            self.drones.append(drone)
        
        # Initialize obstacles (fewer and smaller)
        self.obstacles = []
        for i in range(num_obstacles):
            position = np.random.uniform(-15, 15, 3)
            position[1] = np.random.uniform(1, 8)  # Reasonable height
            self.obstacles.append({
                'position': position,
                'radius': np.random.uniform(0.8, 1.5),  # Smaller obstacles
                'id': i
            })
        
        # Formation settings
        self.target_formation = {
            'type': 'sphere',
            'count': num_drones,
            'radius': 6.0,
            'spacing': 2.5
        }
        
        # Simulation state
        self.step_count = 0
        self.episode_count = 0
        self.history = {
            'avg_fitness': [],
            'total_collisions': [],
            'formation_scores': [],
            'avg_target_distance': []
        }
    
    def reset_episode(self):
        """Reset all drones for new episode"""
        for drone in self.drones:
            drone.reset_episode()
    
    def step(self):
        """Run one simulation step"""
        states_actions_rewards = []
        
        for drone in self.drones:
            state, action, reward = drone.update(self.drones, self.obstacles, self.target_formation)
            states_actions_rewards.append((drone, state, action, reward))
        
        self.step_count += 1
        
        # Training every few steps
        if self.step_count % 5 == 0:
            for drone, state, action, reward in states_actions_rewards:
                next_state = drone.get_state(self.drones, self.obstacles, self.target_formation)
                drone.train_step(state, action, reward, next_state)
    
    def evaluate_swarm_performance(self):
        """Evaluate overall swarm performance"""
        total_fitness = sum(drone.episode_fitness for drone in self.drones)
        total_collisions = sum(drone.episode_collisions for drone in self.drones)
        
        # Formation quality - average distance to target positions
        total_target_distance = 0.0
        formation_score = 0.0
        
        for drone in self.drones:
            target_pos = drone.get_target_position(self.target_formation)
            dist = np.linalg.norm(drone.position - target_pos)
            total_target_distance += dist
            formation_score += max(0, 10 - dist)  # Score decreases with distance
        
        avg_target_distance = total_target_distance / len(self.drones)
        formation_score = formation_score / len(self.drones)
        
        return {
            'total_fitness': total_fitness,
            'avg_fitness': total_fitness / len(self.drones),
            'total_collisions': total_collisions,
            'formation_score': formation_score,
            'avg_target_distance': avg_target_distance,
            'step': self.step_count
        }
    
    def run_episode(self, steps=300):
        """Run a complete episode"""
        print(f"Starting Episode {self.episode_count + 1}")
        self.reset_episode()
        
        for step in range(steps):
            self.step()
            
            if step % 50 == 0 and step > 0:
                metrics = self.evaluate_swarm_performance()
                print(f"  Step {step}: Avg Fitness: {metrics['avg_fitness']:.2f}, "
                      f"Formation Score: {metrics['formation_score']:.2f}, "
                      f"Collisions: {metrics['total_collisions']}, "
                      f"Avg Target Dist: {metrics['avg_target_distance']:.2f}")
        
        # Episode summary
        final_metrics = self.evaluate_swarm_performance()
        self.history['avg_fitness'].append(final_metrics['avg_fitness'])
        self.history['total_collisions'].append(final_metrics['total_collisions'])
        self.history['formation_scores'].append(final_metrics['formation_score'])
        self.history['avg_target_distance'].append(final_metrics['avg_target_distance'])
        
        self.episode_count += 1
        print(f"Episode {self.episode_count} completed! "
              f"Final Fitness: {final_metrics['avg_fitness']:.2f}, "
              f"Formation Score: {final_metrics['formation_score']:.2f}")
        
        return final_metrics
    
    def change_formation(self, formation_type='sphere'):
        """Change target formation"""
        self.target_formation['type'] = formation_type
        print(f"Formation changed to: {formation_type}")
    
    def get_simulation_data(self):
        """Get current simulation state for visualization"""
        data = {
            'drones': [],
            'obstacles': [],
            'formation': self.target_formation,
            'metrics': self.evaluate_swarm_performance()
        }
        
        for drone in self.drones:
            target_pos = drone.get_target_position(self.target_formation)
            data['drones'].append({
                'id': drone.id,
                'position': drone.position.tolist(),
                'velocity': drone.velocity.tolist(),
                'fitness': drone.episode_fitness,
                'collisions': drone.episode_collisions,
                'target_position': target_pos.tolist()
            })
        
        for obstacle in self.obstacles:
            data['obstacles'].append({
                'position': obstacle['position'].tolist(),
                'radius': obstacle['radius']
            })
        
        return data
    
    def save_model(self, filepath):
        """Save all drone neural networks"""
        models = {}
        for i, drone in enumerate(self.drones):
            models[f'drone_{i}'] = drone.brain.state_dict()
        
        # Also save training history
        models['training_history'] = self.history
        models['episode_count'] = self.episode_count
        
        torch.save(models, filepath)
        print(f"Models and training data saved to {filepath}")
    
    def load_model(self, filepath):
        """Load drone neural networks"""
        models = torch.load(filepath, map_location=self.device)
        for i, drone in enumerate(self.drones):
            if f'drone_{i}' in models:
                drone.brain.load_state_dict(models[f'drone_{i}'])
                drone.epsilon = 0.1  # Reduce exploration for trained models
        
        if 'training_history' in models:
            self.history = models['training_history']
        if 'episode_count' in models:
            self.episode_count = models['episode_count']
            
        print(f"Models loaded from {filepath}")
    
    def plot_training_history(self):
        """Plot training progress"""
        if not self.history['avg_fitness']:
            print("No training history to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        episodes = range(1, len(self.history['avg_fitness']) + 1)
        
        axes[0,0].plot(episodes, self.history['avg_fitness'], 'b-', marker='o')
        axes[0,0].set_title('Average Fitness Over Episodes')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Avg Fitness')
        axes[0,0].grid(True)
        
        axes[0,1].plot(episodes, self.history['total_collisions'], 'r-', marker='s')
        axes[0,1].set_title('Total Collisions Over Episodes')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Collisions')
        axes[0,1].grid(True)
        
        axes[1,0].plot(episodes, self.history['formation_scores'], 'g-', marker='^')
        axes[1,0].set_title('Formation Score Over Episodes')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Formation Score')
        axes[1,0].grid(True)
        
        axes[1,1].plot(episodes, self.history['avg_target_distance'], 'm-', marker='d')
        axes[1,1].set_title('Average Distance to Target')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Distance')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training and simulation loop"""
    print("=== AI-Driven Drone Swarm Simulation with PyTorch ===")
    print("Initializing improved training system...")
    
    # Create simulation with better parameters
    sim = DroneSwarmSimulation(num_drones=12, num_obstacles=6)
    
    # Training loop
    num_episodes = 8
    formations = ['sphere', 'line', 'triangle', 'sphere', 'line', 'sphere', 'triangle', 'sphere']
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"TRAINING EPISODE {episode + 1}/{num_episodes}")
        print(f"Current Formation: {formations[episode]}")
        print(f"{'='*60}")
        
        # Set formation for this episode
        sim.change_formation(formations[episode])
        
        # Run episode with adjusted steps
        steps = 250 if episode < 3 else 300  # More steps for later episodes
        final_metrics = sim.run_episode(steps=steps)
        
        # Show improvement
        if episode > 0:
            prev_fitness = sim.history['avg_fitness'][-2] if len(sim.history['avg_fitness']) > 1 else 0
            improvement = final_metrics['avg_fitness'] - prev_fitness
            print(f"Fitness improvement from last episode: {improvement:+.2f}")
    
    # Training completed
    print(f"\n{'='*60}")
    print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
    print(f"{'='*60}")
    
    final_performance = sim.evaluate_swarm_performance()
    print(f"📊 FINAL RESULTS:")
    print(f"   • Average Fitness: {final_performance['avg_fitness']:.2f}")
    print(f"   • Formation Score: {final_performance['formation_score']:.2f}")
    print(f"   • Total Collisions: {final_performance['total_collisions']}")
    print(f"   • Average Target Distance: {final_performance['avg_target_distance']:.2f}")
    print(f"   • Total Training Steps: {sim.step_count}")
    print(f"   • Episodes Completed: {sim.episode_count}")
    
    # Show learning progress
    if len(sim.history['avg_fitness']) > 1:
        initial_fitness = sim.history['avg_fitness'][0]
        final_fitness = sim.history['avg_fitness'][-1]
        total_improvement = final_fitness - initial_fitness
        print(f"   • Total Learning Improvement: {total_improvement:+.2f}")
    
    # Save everything
    sim.save_model('trained_drone_swarm_v2.pth')
    
    # Export final simulation state
    sim_data = sim.get_simulation_data()
    with open('simulation_data_final.json', 'w') as f:
        json.dump(sim_data, f, indent=2)
    print(f"📁 Simulation data exported to simulation_data_final.json")
    
    # Show plots
    print("📈 Generating training progress plots...")
    sim.plot_training_history()
    
    print("\n✅ All done! Check the generated plots and saved models.")

if __name__ == "__main__":
    main()