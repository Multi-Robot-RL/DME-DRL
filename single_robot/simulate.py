import numpy as np


def simulate_robot_step(
    location, direction, action, elapsed_time, map_size, randomness_std=0.1
):
    """
    Simulate the robot's next state based on current state, action, elapsed time, and randomness.

    Args:
        location (tuple): Current (x, y) coordinates of the robot on the map.
        direction (float): Current direction of the robot in radians.
        action (tuple): (linear_velocity, angular_velocity) from the model.
        elapsed_time (float): Time elapsed since the last step (in seconds).
        map_size (tuple): (width, height) of the map.
        randomness_std (float): Standard deviation for Gaussian noise.

    Returns:
        tuple: Next (location, direction) of the robot.
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

    return (next_x, next_y), next_direction
