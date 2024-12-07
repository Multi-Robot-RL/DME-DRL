from datetime import datetime
import random
import torch
from model import device  # Use the globally available device
from animation import animate_robot_progress
from simulate import simulate_robot_step, update_maps
from hyperparam import MAX_DETECTION_DIST, MAX_DETECTION_ANGLE, MAX_EPISODES, TRAIN_DATASET_SIZE
from dataset import load_processed_dataset
from prep import create_environment, generate_random_free_location
import math


def calculate_potentially_explorable_distance(
    current_location, current_direction, frontier_map, robot_obstacle_map, max_detection_dist, num_directions=16
):
    """
    Calculate the potentially explorable distance for each angular section.

    Args:
        current_location (tuple): Current (x, y) coordinates of the robot.
        current_direction (float): Current direction of the robot in radians.
        frontier_map (torch.Tensor): Frontier map indicating explored areas.
        robot_obstacle_map (torch.Tensor): Robot's obstacle map.
        max_detection_dist (float): Maximum detection range of the robot.
        num_directions (int): Number of angular sections to consider.

    Returns:
        list: Potentially explorable distances for each direction.
    """
    angles = torch.linspace(0, 2 * math.pi, num_directions, device=device)



    # Calculate potentially explorable distances
    explorable_distances = []
    for angle in angles:
        ray_angle = current_direction + angle
        dx = torch.cos(ray_angle)
        dy = torch.sin(ray_angle)

        # Check along the ray
        for step in torch.linspace(0, max_detection_dist, steps=100, device=device):
            x = int(current_location[0] + step * dx)
            y = int(current_location[1] + step * dy)

            # Ensure within map bounds
            if x < 0 or x >= frontier_map.shape[0] or y < 0 or y >= frontier_map.shape[1]:
                explorable_distances.append(step.item())
                break

            # Check for obstacles or explored areas
            if robot_obstacle_map[x, y] == 1 or frontier_map[x, y] == 1:
                explorable_distances.append(step.item())
                break
        else:
            # If no obstacle or explored area is encountered, add max detection distance
            explorable_distances.append(max_detection_dist)

    return explorable_distances


def static_strategy_with_potential_distance(
    current_location,
    current_direction,
    frontier_map,
    robot_obstacle_map,
    ground_truth_obstacle_map,
    map_size,
    max_steps=1000,
    num_directions=16,
    save_path="static_strategy_potential_distance.gif",
):
    """
    Static strategy for exploring the map using potentially explorable distance with animation.

    Args:
        current_location (tuple): Initial (x, y) location of the robot.
        current_direction (float): Initial direction of the robot in radians.
        frontier_map (torch.Tensor): Frontier map indicating explored areas.
        robot_obstacle_map (torch.Tensor): Robot's obstacle map.
        ground_truth_obstacle_map (torch.Tensor): Ground truth obstacle map.
        map_size (tuple): (width, height) of the map.
        max_steps (int): Maximum steps before termination.
        num_directions (int): Number of angular sections to consider.
        save_path (str): Path to save the generated animation.

    Returns:
        tuple: Updated (frontier_map, robot_obstacle_map).
    """
    # Move tensors to the correct device
    frontier_map = frontier_map.to(device)
    robot_obstacle_map = robot_obstacle_map.to(device)
    ground_truth_obstacle_map = ground_truth_obstacle_map.to(device)

    # Log data for animation
    locations, directions, frontier_maps, robot_obstacle_maps = [], [], [], []

    for step in range(max_steps):
        print(f"Step {step} at {current_location}")
        # Log current state for animation
        locations.append(current_location)
        directions.append(current_direction)
        frontier_maps.append(frontier_map.clone().cpu())
        robot_obstacle_maps.append(robot_obstacle_map.clone().cpu())

        # Calculate potentially explorable distances
        explorable_distances = calculate_potentially_explorable_distance(
            current_location, current_direction, frontier_map, robot_obstacle_map, MAX_DETECTION_DIST, num_directions
        )

        # Choose the best direction
        best_direction_idx = torch.argmax(torch.tensor(explorable_distances)).item()
        angle_increment = 2 * torch.pi / num_directions
        best_direction = current_direction + best_direction_idx * angle_increment

        # Simulate robot movement towards the best direction
        linear_velocity = 1.0  # Move at constant speed
        angular_velocity = best_direction - current_direction
        action = (linear_velocity, angular_velocity)

        next_location, next_direction, collision = simulate_robot_step(
            current_location,
            current_direction,
            action,
            elapsed_time=1.0,
            map_size=map_size,
            binary_map=ground_truth_obstacle_map,
        )

        # Update the maps
        frontier_map, robot_obstacle_map = update_maps(
            frontier_map,
            robot_obstacle_map,
            next_location,
            next_direction,
            ground_truth_obstacle_map,
            MAX_DETECTION_DIST,
            MAX_DETECTION_ANGLE,
        )

        # Handle collision
        if collision:
            print("Collision detected. Adjusting direction...")
            current_direction += torch.pi / 4  # Turn 45 degrees to avoid obstacle
        else:
            # Update state
            current_location = next_location
            current_direction = next_direction

    # Create and save the animation
    animate_robot_progress(
        frontier_maps,
        robot_obstacle_maps,
        ground_truth_obstacle_map.cpu().numpy(),
        locations,
        directions,
        max_detection_dist=MAX_DETECTION_DIST,
        max_detection_angle=MAX_DETECTION_ANGLE,
        save_path=save_path,
    )

    return frontier_map, robot_obstacle_map

# Example inputs
if __name__ == "__main__":
    # Training loop
    dataset = load_processed_dataset(0, TRAIN_DATASET_SIZE)
    train_id = "STATIC" + str(datetime.now()).replace(" ", ".")
    reward_history = []
    for episode in range(MAX_EPISODES):
        ground_truth_obstacle_map, frontier_map, robot_obstacle_map = create_environment(
            dataset, episode % TRAIN_DATASET_SIZE
        )
        map_size = ground_truth_obstacle_map.shape
        current_location = generate_random_free_location(ground_truth_obstacle_map)
        current_direction = random.uniform(0, 2 * torch.pi)


        # Run the static strategy with animation
        updated_frontier_map, updated_robot_obstacle_map = static_strategy_with_potential_distance(
            current_location,
            current_direction,
            frontier_map,
            robot_obstacle_map,
            ground_truth_obstacle_map,
            map_size=map_size,
            save_path="exploration_animation.gif",  # Save animation as a GIF
        )

    print("Exploration Complete!")

