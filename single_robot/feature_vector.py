import numpy as np


def generate_feature_vector(
    frontier_map,
    obstacle_map,
    robot_position,
    room_split,
    frontier_threshold,
    obstacle_threshold,
):
    """
    Generate the feature vector from frontier and obstacle maps using vectorized operations.

    Args:
        frontier_map (np.ndarray): 2D array representing explored areas.
        obstacle_map (np.ndarray): 2D array representing obstacles.
        robot_position (tuple): Tuple of (x, y) coordinates representing the robot's position.
        room_split (int): Number of angular sections around the robot.
        frontier_threshold (float): Threshold above which an area is considered explored.
        obstacle_threshold (float): Threshold above which a cell is considered an obstacle.

    Returns:
        np.ndarray: Flattened feature vector for the room_split sections.
    """
    # Get map dimensions
    map_size_x, map_size_y = frontier_map.shape

    # Robot position
    robot_x, robot_y = robot_position

    # Angular increment for each section
    angle_increment = 2 * np.pi / room_split

    # Create grid coordinates
    x_coords, y_coords = np.meshgrid(np.arange(map_size_x), np.arange(map_size_y), indexing='ij')

    # Calculate relative positions to the robot
    dx = x_coords - robot_x
    dy = y_coords - robot_y
    distances = np.sqrt(dx**2 + dy**2)

    # Calculate angles relative to the robot
    angles = np.arctan2(dy, dx)
    angles[angles < 0] += 2 * np.pi  # Normalize to [0, 2π)

    # Mask for the robot's current position (avoid self-reference)
    mask_robot_position = (dx == 0) & (dy == 0)

    # Initialize the feature vector
    feature_vector = []

    # Iterate through each angular section
    for section in range(room_split):
        start_angle = section * angle_increment
        end_angle = (section + 1) * angle_increment

        # Create a mask for the current angular section
        angle_mask = (angles >= start_angle) & (angles < end_angle)

        # Combine masks for section and exclude the robot's position
        combined_mask = angle_mask & ~mask_robot_position

        # Metrics for the current section
        total_area = np.sum(combined_mask)
        unexplored_area = np.sum(combined_mask & (frontier_map < frontier_threshold))
        obstacle_distances = distances[combined_mask & (obstacle_map > obstacle_threshold)]

        # Nearest obstacle distance
        nearest_obstacle_distance = obstacle_distances.min() if obstacle_distances.size > 0 else 0

        # Calculate the unexplored ratio
        unexplored_ratio = unexplored_area / total_area if total_area > 0 else 0

        # Append the features for this section
        feature_vector.extend([total_area, nearest_obstacle_distance, unexplored_area, unexplored_ratio])

    # Convert feature vector to numpy array
    return np.array(feature_vector)
