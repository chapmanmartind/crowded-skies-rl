# This file holds all the constants for the game
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
HEADER_HEIGHT = 0

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Compression of 1000:20 -> 50:1
NUM_WIDTH_BINS = 20
NUM_HEIGHT_BINS = 12
NUM_PLAYER_ACTIONS = 3  # no-op, up, down
PLAYER_ACTION_DICT = {'NO-OP': 0, 'UP': 1, 'DOWN': 2}

NUM_EPISODES = 100000

ALPHA = .1
GAMMA = .99

MODEL_SAVE_PATH = "Q_table_model.npy"
