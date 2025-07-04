# This file holds all the constants for the game
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
HEADER_HEIGHT = 100

TITLE = "Crowded Skies"

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

PLAYER_WIDTH = 80
PLAYER_HEIGHT = 30
ENEMY_WIDTH = 40
ENEMY_HEIGHT = 20

GRAVITY_CONST = 0.5
FORCE_CONST = 1.5

PLAYER_IMG_PATH = "images/jet4.png"
EXHAUST_IMG_PATH = "images/jet4_exhaust.png"
ENEMYSTRAIGHTMISSILE_IMG_PATH = "images/EnemyStraightMissile.png"


# Compression of 1000:20 -> 50:1
NUM_WIDTH_BINS = 20
NUM_HEIGHT_BINS = 12
NUM_PLAYER_ACTIONS = 3  # no-op, up, down
PLAYER_ACTION_DICT = {'NO-OP': 0, 'UP': 1, 'DOWN': 2}

ALPHA = .1
GAMMA = .99

