import torch
from model import device


def simulate_robot_step(
    location, direction, action, elapsed_time, map_size, binary_map, randomness_std=0.1
):
    """
    Simulate the robot's next state while considering obstacles in the binary map (CUDA-enabled).

    Args:
        location (tuple): Current (x, y) coordinates of the robot on the map.
        direction (float): Current direction of the robot in radians.
        action (tuple): (linear_velocity, angular_velocity) from the model.
        elapsed_time (float): Time elapsed since the last step (in seconds).
        map_size (tuple): (width, height) of the map.
        binary_map (torch.Tensor): Binary map where 1 indicates obstacles and 0 indicates free space.
        randomness_std (float): Standard deviation for Gaussian noise.
        device (str): Device to run the simulation on ("cuda" or "cpu").

    Returns:
        tuple: Next (location, direction) of the robot.
        bool: Whether the robot collided with an obstacle.
    """
    # Move inputs to device
    binary_map = binary_map.to(device)
    x, y = torch.tensor(location, device=device, dtype=torch.float32)
    linear_velocity, angular_velocity = action
    width, height = map_size

    # Add Gaussian noise to velocities
    noisy_linear_velocity = linear_velocity + torch.normal(
        mean=0, std=randomness_std * abs(linear_velocity), size=(1,), device=device
    )
    noisy_angular_velocity = angular_velocity + torch.normal(
        mean=0, std=randomness_std * abs(angular_velocity), size=(1,), device=device
    )

    # Update direction (radians)
    next_direction = direction + noisy_angular_velocity * elapsed_time
    next_direction %= 2 * torch.pi  # Ensure direction is within [0, 2π]

    # Update location based on noisy linear velocity and elapsed time
    next_x = x + noisy_linear_velocity * elapsed_time * torch.cos(next_direction)
    next_y = y + noisy_linear_velocity * elapsed_time * torch.sin(next_direction)

    # Ensure the robot stays within map boundaries
    next_x = torch.clamp(next_x, 0, width - 1)
    next_y = torch.clamp(next_y, 0, height - 1)

    # Check if the next location collides with an obstacle
    next_i, next_j = int(next_x.item()), int(next_y.item())  # Convert to map indices
    if binary_map[next_i, next_j] == 1:  # Obstacle detected
        return (location, direction, True)

    # No collision; update the robot's state
    return (next_x.item(), next_y.item()), next_direction.item(), False


def update_maps(
    frontier_map,
    robot_obstacle_map,
    location,
    direction,
    ground_truth_obstacle_map,
    max_detection_dist,
    max_detection_angle,
):
    """
    Update the frontier map and the robot's obstacle map based on sensor readings (CUDA-enabled).

    Args:
        frontier_map (torch.Tensor): Frontier map (0 = unexplored, 1 = explored).
        robot_obstacle_map (torch.Tensor): Current obstacle map (0 = free space, 1 = obstacle).
        location (tuple): Current (x, y) location of the robot.
        direction (float): Current direction of the robot in radians.
        ground_truth_obstacle_map (torch.Tensor): Ground truth obstacle map (1 = obstacle, 0 = free space).
        max_detection_dist (float): Maximum detection range of the sensor.
        max_detection_angle (float): Maximum detection angle (radians).
        device (str): Device to run the simulation on ("cuda" or "cpu").

    Returns:
        tuple: Updated (frontier_map, robot_obstacle_map).
    """
    # Move maps to device
    frontier_map = frontier_map.to(device)
    robot_obstacle_map = robot_obstacle_map.to(device)
    ground_truth_obstacle_map = ground_truth_obstacle_map.to(device)

    # Dimensions of the maps
    height, width = frontier_map.shape

    # Robot location
    x_robot, y_robot = location
    x_robot, y_robot = int(x_robot), int(y_robot)

    # Create grid for raycasting
    num_rays = 360  # Raycasting resolution
    ray_angles = torch.linspace(
        direction - max_detection_angle / 2,
        direction + max_detection_angle / 2,
        steps=num_rays,
        device=device,
    )

    # Raycasting steps
    ray_steps = torch.linspace(0, max_detection_dist, steps=100, device=device)

    # Create copies of the maps to update
    updated_frontier_map = frontier_map.clone()
    updated_robot_obstacle_map = robot_obstacle_map.clone()

    # Perform raycasting
    for ray_angle in ray_angles:
        ray_dx = ray_steps * torch.cos(ray_angle)
        ray_dy = ray_steps * torch.sin(ray_angle)

        for step_dx, step_dy in zip(ray_dx, ray_dy):
            x = int(x_robot + step_dx.item())
            y = int(y_robot + step_dy.item())

            # Ensure the point is within the map boundaries
            if x < 0 or x >= height or y < 0 or y >= width:
                break

            # Check the ground truth obstacle map
            if ground_truth_obstacle_map[x, y] == 1:  # Obstacle detected
                updated_frontier_map[x, y] = 1  # Mark as explored
                updated_robot_obstacle_map[x, y] = 1  # Update robot's obstacle map
                break  # Stop raycasting further along this ray

            # Update the frontier map for free space
            updated_frontier_map[x, y] = 1  # Mark as explored
            updated_robot_obstacle_map[x, y] = 0  # Confirm it's free space

    return updated_frontier_map, updated_robot_obstacle_map
