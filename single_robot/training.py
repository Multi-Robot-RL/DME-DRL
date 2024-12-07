from os import makedirs
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import ActorCriticDQN, model, save_onnx, device
from hyperparam import (
    MAX_LINEAR_VELOCITY,
    MAX_ANGULAR_VELOCITY,
    MAX_STEPS,
    MAX_EPISODES,
    MAX_DETECTION_DIST,
    MAX_DETECTION_ANGLE,
    LR,
    EXPLORATION_STD,
    GAMMA,
    TRAIN_DATASET_SIZE,
    COLLISION_PENALTY,
    EXPLORATION_REWARD,
)
from feature_vector import generate_feature_vector
from simulate import simulate_robot_step, update_maps
from dataset import load_processed_dataset, data_path
from reward import evaluate_model_performance
from animation import animate_robot_progress
from datetime import datetime

# Initialize model and optimizer
model = model.to(device)  # Move model to device
optimizer = optim.Adam(model.parameters(), lr=LR)


# Generate random free location
def generate_random_free_location(ground_truth_obstacle_map):
    """
    Generate a random location within the map that is not an obstacle.

    Args:
        ground_truth_obstacle_map (torch.Tensor): Ground truth obstacle map (1 = obstacle, 0 = free space).

    Returns:
        tuple: (x, y) coordinates of the random free location.
    """
    # Find free space indices
    free_space_indices = torch.nonzero(ground_truth_obstacle_map == 0, as_tuple=False)

    # Check if free space exists
    if free_space_indices.size(0) == 0:
        raise ValueError("No free space available in the obstacle map.")

    # Randomly select one of the free cells
    random_index = torch.randint(0, free_space_indices.size(0), (1,)).item()
    random_free_location = tuple(free_space_indices[random_index].tolist())

    return random_free_location


# Environment setup
def create_environment(houseexpo_dataset, room_id):
    ground_truth_obstacle_map = torch.tensor(houseexpo_dataset[room_id]["binary_mask"], device=device)
    print(f"Map size of Room {room_id}: {ground_truth_obstacle_map.shape}")
    map_size = ground_truth_obstacle_map.shape
    frontier_map = torch.zeros(map_size, device=device)
    robot_obstacle_map = torch.zeros(map_size, device=device)
    return ground_truth_obstacle_map, frontier_map, robot_obstacle_map


# Training loop
dataset = load_processed_dataset(0, TRAIN_DATASET_SIZE)
train_id = "TRAIN" + str(datetime.now()).replace(" ", ".")
reward_history = []

for episode in range(MAX_EPISODES):
    ground_truth_obstacle_map, frontier_map, robot_obstacle_map = create_environment(
        dataset, episode % TRAIN_DATASET_SIZE
    )
    map_size = ground_truth_obstacle_map.shape
    current_location = generate_random_free_location(ground_truth_obstacle_map)
    current_direction = np.random.uniform(0, 2 * np.pi)
    total_reward = 0
    old_performance = 0
    locations, directions, frontier_maps, robot_obstacle_maps = [], [], [], []

    for step in range(MAX_STEPS):
        # Logging for visualization
        locations.append(current_location)
        directions.append(current_direction)
        frontier_maps.append(frontier_map)
        robot_obstacle_maps.append(robot_obstacle_map)
        print(f"Step {step} at {current_location}")

        # Generate features and move to device
        feature_tensor = generate_feature_vector(
            robot_obstacle_map,
            frontier_map,
            current_location,
            current_direction,
            0.5,
            0.5,
        ).unsqueeze(0).to(device)
        # Forward pass
        actor_output, critic_value = model(feature_tensor)
        action = actor_output[0].detach().cpu().numpy()
        linear_velocity, angular_velocity = action

        # Simulate next step
        next_location, next_direction, collision = simulate_robot_step(
            current_location,
            current_direction,
            (linear_velocity, angular_velocity),
            1.0,
            map_size,
            ground_truth_obstacle_map,
        )

        # Update maps
        frontier_map, robot_obstacle_map = update_maps(
            frontier_map,
            robot_obstacle_map,
            next_location,
            next_direction,
            ground_truth_obstacle_map,
            MAX_DETECTION_DIST,
            MAX_DETECTION_ANGLE,
        )

        # Evaluate performance
        new_performance = evaluate_model_performance(
            frontier_map, ground_truth_obstacle_map
        )

        # Reward computation
        if collision:
            reward = COLLISION_PENALTY
        else:
            reward = EXPLORATION_REWARD * np.tanh(new_performance - old_performance)
        old_performance = new_performance
        total_reward += reward

        # Prepare tensors for TD target and loss
        next_feature_tensor = generate_feature_vector(
            robot_obstacle_map, frontier_map, next_location, next_direction, 0.5, 0.5
        ).unsqueeze(0).to(device)

        _, next_critic_value = model(next_feature_tensor)
        td_target = reward + GAMMA * next_critic_value
        critic_loss = F.mse_loss(critic_value, td_target.detach())

        # Actor loss
        advantage = td_target - critic_value
        actor_loss = -torch.mean(advantage.detach() * torch.sum(actor_output))

        # Total loss
        loss = actor_loss + critic_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update state
        if collision:
            break
        current_location, current_direction = next_location, next_direction

    reward_history.append(total_reward)
    print(
        f"Episode {episode + 1}/{MAX_EPISODES}, Total Reward: {total_reward:.2f}, Free Space Discovered: {old_performance:.2f}%"
    )

    # Save animations periodically
    if episode % 20 == 0:
        if not (data_path / train_id).exists():
            makedirs(data_path / train_id)
        animate_robot_progress(
            frontier_maps,
            robot_obstacle_maps,
            ground_truth_obstacle_map,
            locations,
            directions,
            max_detection_dist=MAX_DETECTION_DIST,
            max_detection_angle=MAX_DETECTION_ANGLE,
            save_path=data_path / train_id / f"episode{episode}.gif",
        )

# Save model in ONNX format
save_onnx(model, data_path / train_id / "model.onnx")
print("Training Complete.")
