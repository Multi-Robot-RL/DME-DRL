import numpy as np

from hyperparam import ROOM_SPLIT

import numpy as np

def generate_feature_vector(
    frontier_map,
    obstacle_map,
    robot_position,
    robot_direction,
    frontier_threshold,
    obstacle_threshold,
):
    """
    Generate the feature vector from frontier and obstacle maps using vectorized operations.

    Args:
        frontier_map (np.ndarray): 2D array representing explored areas.
        obstacle_map (np.ndarray): 2D array representing obstacles.
        robot_position (tuple): Tuple of (x, y) coordinates representing the robot's position.
        robot_direction (float): Robot's current direction in radians.
        ROOM_SPLIT (int): Number of angular sections around the robot.
        frontier_threshold (float): Threshold above which an area is considered explored.
        obstacle_threshold (float): Threshold above which a cell is considered an obstacle.

    Returns:
        np.ndarray: Flattened feature vector for the ROOM_SPLIT sections.
    """
    # Get map dimensions
    map_size_x, map_size_y = frontier_map.shape

    # Robot position
    robot_x, robot_y = robot_position

    # Create grid coordinates
    x_coords, y_coords = np.meshgrid(
        np.arange(map_size_x), np.arange(map_size_y), indexing="ij"
    )

    # Calculate relative positions to the robot
    dx = x_coords - robot_x
    dy = y_coords - robot_y
    distances = np.sqrt(dx**2 + dy**2)

    # Calculate angles relative to the robot
    angles = np.arctan2(dy, dx)
    angles[angles < 0] += 2 * np.pi  # Normalize to [0, 2π)

    # Mask for the robot's current position (avoid self-reference)
    mask_robot_position = (dx == 0) & (dy == 0)

    # Mask out-of-bound areas (optional if needed)
    mask_valid = distances > 0

    # Combine valid mask
    valid_mask = mask_valid & ~mask_robot_position

    # Angular increments for each section
    angle_increment = 2 * np.pi / ROOM_SPLIT
    start_angles = np.arange(ROOM_SPLIT) * angle_increment
    end_angles = start_angles + angle_increment

    # Preallocate metrics
    total_area = np.zeros(ROOM_SPLIT)
    unexplored_area = np.zeros(ROOM_SPLIT)
    nearest_obstacle_distance = np.full(ROOM_SPLIT, np.inf)

    # Compute masks for angular sections
    for section, (start_angle, end_angle) in enumerate(zip(start_angles, end_angles)):
        angle_mask = (angles >= start_angle) & (angles < end_angle)
        combined_mask = valid_mask & angle_mask

        # Total area in this section
        total_area[section] = np.sum(combined_mask)

        # Unexplored area
        unexplored_area[section] = np.sum(combined_mask & (frontier_map < frontier_threshold))

        # Nearest obstacle distance
        obstacle_distances = distances[combined_mask & (obstacle_map > obstacle_threshold)]
        if obstacle_distances.size > 0:
            nearest_obstacle_distance[section] = obstacle_distances.min()

    # Handle sections with no obstacles
    nearest_obstacle_distance[nearest_obstacle_distance == np.inf] = 0.1  # Add epsilon

    # Calculate unexplored ratio
    unexplored_ratio = np.divide(
        unexplored_area, total_area, out=np.zeros_like(unexplored_area), where=total_area > 0
    )

    # Flatten the feature vector
    feature_vector = np.concatenate(
        [
            total_area,
            1 / nearest_obstacle_distance,
            unexplored_area,
            1 / unexplored_ratio,
            np.array(robot_position),
            frontier_map.shape,
            np.array([robot_direction]),
        ]
    )

    return feature_vector

