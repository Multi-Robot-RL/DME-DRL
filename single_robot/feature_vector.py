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
    Generate the feature vector from frontier and obstacle maps.

    Args:
        frontier_map (np.ndarray): 2D array representing explored areas.
        obstacle_map (np.ndarray): 2D array representing obstacles.
        robot_position (tuple): Tuple of (x, y) coordinates representing the robot's position.
        room_split (int): Number of angular sections around the robot.
        frontier_threshold (float): Threshold above which an area is considered explored.
        obstacle_threshold (float): Threshold above which a cell is considered an obstacle.

    Returns:
        np.ndarray: Flattened feature vector for the ROOM_SPLIT sections.
    """
    # Get map dimensions
    map_size_x, map_size_y = frontier_map.shape

    # Robot position
    robot_x, robot_y = robot_position

    # Angular increment for each section
    angle_increment = 2 * np.pi / room_split

    # Initialize the feature vector
    feature_vector = []

    # Iterate through each angular section
    for section in range(room_split):
        start_angle = section * angle_increment
        end_angle = (section + 1) * angle_increment

        # Initialize section-specific metrics
        total_area = 0
        unexplored_area = 0
        nearest_obstacle_distance = np.inf

        # Traverse the map to calculate metrics for the current section
        for i in range(map_size_x):
            for j in range(map_size_y):
                # Calculate relative position to the robot
                dx, dy = i - robot_x, j - robot_y
                distance = np.sqrt(dx**2 + dy**2)

                # Skip the robot's current position
                if distance == 0:
                    continue

                # Calculate the angle relative to the robot
                angle = np.arctan2(dy, dx)
                if angle < 0:
                    angle += 2 * np.pi  # Normalize to [0, 2π)

                # Check if the cell falls within the current angular section
                if start_angle <= angle < end_angle:
                    total_area += 1

                    # Check unexplored area
                    if frontier_map[i, j] < frontier_threshold:
                        unexplored_area += 1

                    # Check obstacle distance
                    if obstacle_map[i, j] > obstacle_threshold:
                        nearest_obstacle_distance = min(
                            nearest_obstacle_distance, distance
                        )

        # Handle case where no obstacle is found
        if nearest_obstacle_distance == np.inf:
            nearest_obstacle_distance = 0  # No obstacles in this section

        # Calculate the unexplored ratio
        unexplored_ratio = unexplored_area / total_area if total_area > 0 else 0

        # Append the features for this section
        feature_vector.extend(
            [total_area, nearest_obstacle_distance, unexplored_area, unexplored_ratio]
        )

    # Convert feature vector to numpy array
    return np.array(feature_vector)
