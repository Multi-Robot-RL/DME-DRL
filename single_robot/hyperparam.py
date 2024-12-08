import numpy as np

ROOM_SPLIT = 20
HIDDEN_DIM = 1024
OUTPUT_DIM = 2

MAX_LINEAR_VELOCITY = 30
MAX_ANGULAR_VELOCITY = 0.1

# Parameters
MAX_EPISODES = 500  # Number of episodes
MAX_STEPS = 500  # Max steps per episode
GAMMA = 0.99  # Discount factor for rewards
LR = 1e-3  # Learning rate
EXPLORATION_STD = 5  # Exploration noise standard deviation
MAX_DETECTION_DIST = 200
MAX_DETECTION_ANGLE = np.pi / 3  # 60 degrees field of view

# Reward settings
COLLISION_PENALTY = -5
EXPLORATION_REWARD = 1e2

# DATASET
TRAIN_DATASET_SIZE = 1
