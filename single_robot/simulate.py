import numpy as np


def simulate_robot_step(
    location, direction, action, elapsed_time, map_size, binary_map, randomness_std=0.1
):
    """
    Simulate the robot's next state while considering obstacles in the binary map.

    Args:
        location (tuple): Current (x, y) coordinates of the robot on the map.
        direction (float): Current direction of the robot in radians.
        action (tuple): (linear_velocity, angular_velocity) from the model.
        elapsed_time (float): Time elapsed since the last step (in seconds).
        map_size (tuple): (width, height) of the map.
        binary_map (np.ndarray): Binary map where 1 indicates obstacles and 0 indicates free space.
        randomness_std (float): Standard deviation for Gaussian noise.

    Returns:
        tuple: Next (location, direction) of the robot.
        bool: Whether the robot collided with an obstacle.
    """
    # Unpack location and action
    x, y = location
    linear_velocity, angular_velocity = action
    width, height = map_size

    # Add Gaussian noise to velocities
    noisy_linear_velocity = linear_velocity + np.random.normal(
        0, randomness_std * abs(linear_velocity)
    )
    noisy_angular_velocity = angular_velocity + np.random.normal(
        0, randomness_std * abs(angular_velocity)
    )

    # Update direction (radians)
    next_direction = direction + noisy_angular_velocity * elapsed_time
    next_direction %= 2 * np.pi  # Ensure direction is within [0, 2π]

    # Update location based on noisy linear velocity and elapsed time
    next_x = x + noisy_linear_velocity * elapsed_time * np.cos(next_direction)
    next_y = y + noisy_linear_velocity * elapsed_time * np.sin(next_direction)

    # Ensure the robot stays within map boundaries
    next_x = max(0, min(next_x, width - 1))
    next_y = max(0, min(next_y, height - 1))

    # Check if the next location collides with an obstacle
    next_i, next_j = int(next_y), int(next_x)  # Convert to map indices
    if binary_map[next_i, next_j] == 1:  # Obstacle detected
        return (
            location,
            direction,
            True,
        )  # Return the current state and collision status

    # No collision; update the robot's state
    return (next_x, next_y), next_direction, False


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
    Update the frontier map and the robot's obstacle map based on sensor readings.

    Args:
        frontier_map (np.ndarray): Original frontier map (0 = unexplored, 1 = explored).
        robot_obstacle_map (np.ndarray): Current obstacle map maintained by the robot (0 = free space, 1 = obstacle).
        location (tuple): Current (x, y) location of the robot.
        direction (float): Current direction of the robot in radians.
        ground_truth_obstacle_map (np.ndarray): Ground truth obstacle map (1 = obstacle, 0 = free space).
        max_detection_dist (float): Maximum detection range of the sensor.
        max_detection_angle (float): Maximum detection angle (radians).

    Returns:
        tuple: Updated (frontier_map, robot_obstacle_map).
    """
    # Dimensions of the maps
    height, width = frontier_map.shape

    # Robot location
    x_robot, y_robot = location
    x_robot, y_robot = int(x_robot), int(y_robot)

    # Create copies of the maps to update
    updated_frontier_map = frontier_map.copy()
    updated_robot_obstacle_map = robot_obstacle_map.copy()

    # Iterate through angles within the detection cone
    num_rays = 360  # Raycasting resolution (increase for finer updates)
    for ray in range(num_rays):
        # Calculate the angle of the current ray
        ray_angle = (
            direction
            - (max_detection_angle / 2)
            + (ray / num_rays) * max_detection_angle
        )

        # Raycast along this angle
        for step in np.linspace(
            0, max_detection_dist, num=100
        ):  # Fine resolution along the ray
            # Calculate the position of the current step
            dx = step * np.cos(ray_angle)
            dy = step * np.sin(ray_angle)
            x, y = int(x_robot + dx), int(y_robot + dy)

            # Ensure the point is within the map boundaries
            if x < 0 or x >= width or y < 0 or y >= height:
                break

            # Check the ground truth obstacle map
            if ground_truth_obstacle_map[y, x] == 1:  # Obstacle detected
                updated_frontier_map[y, x] = 1  # Mark as explored
                updated_robot_obstacle_map[y, x] = 1  # Update robot's obstacle map
                break  # Stop raycasting further along this ray

            # Update the frontier map for free space
            updated_frontier_map[y, x] = 1  # Mark as explored
            updated_robot_obstacle_map[y, x] = 0  # Confirm it's free space

    return updated_frontier_map, updated_robot_obstacle_map
