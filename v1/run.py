# Test mode. 0 if training, 1 if testing

from game import Game
from model import Model
import numpy as np

mode = 1
model = Model(mode)
model.test()
