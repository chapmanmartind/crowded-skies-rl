# In this model we will implement the Q-Learning algorithm

from game import Game
from constants import (SCREEN_WIDTH, SCREEN_HEIGHT,
                       NUM_WIDTH_BINS, NUM_HEIGHT_BINS, NUM_PLAYER_ACTIONS,
                       NUM_EPISODES)
import numpy as np

# Train mode, 1 if training, 0 if testing
TRAIN = 1


class Model():
    def __init__(self, mode, num_width_bins, num_height_bins,
                 num_player_actions,
                 num_episodes, train):
        # Initializing the model
        self.mode = mode
        # For this version of the game the player cannot move left/right
        self.num_player_x_bins = 0
        self.num_player_y_bins = num_height_bins
        self.num_enemy_x_bins = num_width_bins
        self.num_enemy_y_bins = num_height_bins
        self.num_player_actions = num_player_actions
        self.num_episodes = num_episodes
        self.train = train

        # If we are in training mode we want the game not to be in human
        # or render mode, and we want to instantiate a new Q_Table
        if self.train:
            human_mode = False
            render_mode = False
            self.game = Game(human_mode, render_mode)
            self.table = Q_Table(self.num_player_x_bins,
                                 self.num_player_y_bins,
                                 self.num_enemy_x_bins, self.num_enemy_y_bins,
                                 self.num_player_actions)
        else:
            # TODO: test mode
            pass

    def train_model(self):
        self.game.reset()

        for episode_num in range(self.num_episodes):
            self.run_episode(episode_num)
            pass

    def run_episode(self, episode_num):

        # observation: [player_pos, enemy_pos, enemies_survived, game_over,
        # victory]
        (player_pos, enemy_pos, enemies_survived, game_over,
         victory) = Game.get_observation()
        action = self.choose_action(episode_num)
        observation_new = self.game_step()
        update = self.calculate_update(observation)
        pass

    def game_step(self):
        # Runs a single step of the game and returns the observation
        Game.update()
        # The observation comes in the form of
        # [player_pos, enemy_pos, game_over, victory]
        observation = Game.get_observation()
        return observation

    def calculate_update(self, observation):
        # The overall update function is given by
        # Q(s,a) <- Q(s, a) + alpha[R + gamma*M - Q(s, a)]
        # Where Q(s, a) is the former Q-value estimation
        # alpha is the learning rate
        # R is the immediate reward
        # gamma is the discount rate
        # M is the Q-value of the optimal next state

        (player_pos, enemy_pos, enemies_survived, game_over,
         victory) = observation
        # The model always gets a raw reward based on how many enemies it has
        # survived to incentivize moving forward
        # and then subtracts from value if it lost
        # and adds more if it won
        raw_reward = enemies_survived
        if game_over:
            # (victory - 0.5) * 20 because victory is 0 or 1
            # which is then converted to -10 or 10
            raw_reward += (victory - 0.5) * 20

        query = self.table.get(player_pos, enemy_pos, action)

    def choose_action(episode_num):
        # We need to choose an action from the table according to
        # the epsilon-greedy policy
        # Epsilon represents the odds of choosing a random move

        # The value of epsilon here is defined as a negative exponential
        # as a function of the episode number
        epsilon = np.exp(-5 * (episode_num / 1000000))
        if epsilon > np.rand(0, 1):
            # choose a random action
            # TODO
            pass
        else:
            # choose the best actions
            # TODO
            pass

class Q_Table():
    # Separating Q-table from model for the benefit of future implementations
    # The Q_Table class will hold the table and handle updating it
    def __init__(self, num_player_x_bins, num_player_y_bins,
                 num_enemy_x_bins, num_enemy_y_bins,
                 num_player_actions):
        # Initializing the Q-Table object
        self.num_player_x_bins = num_player_x_bins
        self.num_player_y_bins = num_player_y_bins
        self.num_enemy_x_bins = num_enemy_x_bins
        self.num_enemy_y_bins = num_enemy_y_bins
        self.num_player_actions = num_player_actions
        self.init_table()
        pass

    def init_table(self):
        # Initalizes the table itself
        # For this implementation the table will have dimension
        # player_y_bin, missile_x_bin, missile_y_bin, action
        table = np.zeros(self.num_player_x_bins, self.num_player_y_bins,
                         self.num_enemy_x_bins, self.num_enemy_y_bins,
                         self.num_player_actions)
        # It is possible for any of the number of bins above to be 0
        # So we have to squeeze the table to delete this dimension
        self.table = np.squeeze(table)
        pass
