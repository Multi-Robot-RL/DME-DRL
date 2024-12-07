import torch
from simulate import simulate_robot_step
from hyperparam import MAX_DETECTION_DIST, MAX_DETECTION_ANGLE


def static_strategy(
    current_location,
    current_direction,
    frontier_map,
    robot_obstacle_map,
    ground_truth_obstacle_map,
    map_size,
    max_steps=1000,
    device="cpu",
):
    """
    Static strategy for exploring the map without AI.

    Args:
        current_location (tuple): Initial (x, y) location of the robot.
        current_direction (float): Initial direction of the robot in radians.
        frontier_map (torch.Tensor): Frontier map indicating explored areas.
        robot_obstacle_map (torch.Tensor): Robot's obstacle map.
        ground_truth_obstacle_map (torch.Tensor): Ground truth obstacle map.
        map_size (tuple): (width, height) of the map.
        max_steps (int): Maximum steps before termination.
        device (str): Device for computations ("cpu" or "cuda").

    Returns:
        tuple: Updated (frontier_map, robot_obstacle_map).
    """
    # Move tensors to the correct device
    frontier_map = frontier_map.to(device)
    robot_obstacle_map = robot_obstacle_map.to(device)
    ground_truth_obstacle_map = ground_truth_obstacle_map.to(device)

    for step in range(max_steps):
        # Find the closest unexplored cell
        unexplored_mask = frontier_map == 0
        if torch.sum(unexplored_mask) == 0:
            print("Exploration complete!")
            break

        unexplored_indices = torch.nonzero(unexplored_mask, as_tuple=False)
        distances = torch.sqrt(
            (unexplored_indices[:, 0] - current_location[0]) ** 2
            + (unexplored_indices[:, 1] - current_location[1]) ** 2
        )
        target_index = unexplored_indices[distances.argmin()]
        target_location = (target_index[0].item(), target_index[1].item())

        # Calculate direction to the target location
        dx = target_location[0] - current_location[0]
        dy = target_location[1] - current_location[1]
        target_direction = torch.atan2(dy, dx).item()

        # Simulate robot movement towards the target
        linear_velocity = 1.0  # Move at constant speed
        angular_velocity = target_direction - current_direction
        action = (linear_velocity, angular_velocity)

        next_location, next_direction, collision = simulate_robot_step(
            current_location,
            current_direction,
            action,
            elapsed_time=1.0,
            map_size=map_size,
            binary_map=ground_truth_obstacle_map,
            randomness_std=0.0,  # No randomness for deterministic strategy
            device=device,
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
            device=device,
        )

        # Handle collision
        if collision:
            print("Collision detected. Adjusting direction...")
            current_direction += torch.pi / 4  # Turn 45 degrees to avoid obstacle
        else:
            # Update state
            current_location = next_location
            current_direction = next_direction

    return frontier_map, robot_obstacle_map
