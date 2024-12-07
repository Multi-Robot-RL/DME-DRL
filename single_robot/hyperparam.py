import numpy as np
ROOM_SPLIT = 200
HIDDEN_DIM = 1024
OUTPUT_DIM = 2

MAX_LINEAR_VELOCITY = 5
MAX_ANGULAR_VELOCITY = 1

# Parameters
MAX_EPISODES = 500  # Number of episodes
MAX_STEPS = 100  # Max steps per episode
GAMMA = 0.99  # Discount factor for rewards
LR = 1e-4  # Learning rate
EXPLORATION_STD = 0.1  # Exploration noise standard deviation
MAX_LINEAR_VELOCITY = 2.0
MAX_ANGULAR_VELOCITY = 0.5
MAX_DETECTION_DIST = 10
MAX_DETECTION_ANGLE = np.pi / 3  # 60 degrees field of view

# Reward settings
COLLISION_PENALTY = -10
EXPLORATION_REWARD = 1

# DATASET
TRAIN_DATASET_SIZE = 10
