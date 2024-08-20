Here's a README for your project in Obsidian format:

---

# PPO-based Drone Navigation and Wildfire Avoidance

This project implements a Proximal Policy Optimization (PPO) agent that controls a drone to avoid a spreading wildfire in a simulated environment. The drone learns to navigate towards or away from the wildfire based on rewards determined by the distance between the drone and the wildfire.

## Project Structure

### Classes

- **Drone**: Simulates the drone's physical dynamics, including position, velocity, and acceleration.
  
- **Wildfire**: Models the wildfire's behavior, including its position, intensity, spread rate, and radius.

- **DroneEnv**: A custom Gym environment that includes the drone and the wildfire. It defines the observation and action spaces, as well as the logic for resetting, stepping through the environment, and calculating rewards.

- **Actor**: The neural network that approximates the policy function. It outputs the mean and standard deviation of a normal distribution over actions.

- **Critic**: The neural network that estimates the value function, which is used to calculate advantages for the PPO algorithm.

- **PPOAgent**: The main agent class that handles the policy update, action selection, and training loop.

### Functions

- **train**: Trains the PPO agent over a specified number of episodes. It collects data, updates the policy and value networks, and logs rewards.

- **select_action**: Samples an action from the policy's distribution and returns the action along with its log probability.

- **update**: Updates the policy and value networks using the PPO algorithm.

### Simulation Parameters

- **Max Episodes**: 1000
- **Max Steps per Episode**: 200
- **Learning Rate**: 3e-4
- **Gamma**: 0.99 (discount factor)
- **Epsilon Clip**: 0.2 (PPO clip range)

### Reward Structure

- The reward is based on the distance between the drone and the wildfire:
  - **Inside the wildfire**: The reward is proportional to the negative difference between the wildfire's radius and the distance.
  - **Within a certain threshold (20 units)**: The drone receives a high reward of 1,000,000 for being close to the wildfire.
  - **Outside the threshold**: The reward is the negative of the distance.

## Training Process

During training, the agent iteratively interacts with the environment to collect experiences, updates the policy using these experiences, and logs rewards to track progress.

- **Episode Rewards**: Total reward collected during each episode.
- **Average Rewards**: Moving average of rewards over the last 100 episodes.
- **Average Acceleration**: Average action taken by the drone over the episode.

### Running the Training

To run the training process:

1. Initialize the environment (`DroneEnv`).
2. Create a PPO agent with the environment.
3. Train the agent by calling `train` on the PPO agent.

### Plotting Results

The training script will generate and save a plot showing the episode rewards and average rewards over time.

## How to Use

1. Install required dependencies:
   - `numpy`
   - `torch`
   - `gym`
   - `matplotlib`

2. Run the script:
   ```bash
   python main.py
   ```

3. The results, including the reward plot, will be saved in the project directory.

## Future Improvements

- Fine-tune hyperparameters for better training stability and performance.
- Introduce additional complexities to the environment, such as multiple wildfires or varying wind conditions.
- Implement multi-drone coordination strategies.

---

This README provides an overview of the project, how it is structured, and instructions for running the code. It also includes details on the reward system and potential future enhancements.
