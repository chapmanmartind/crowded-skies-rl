# In this model we will implement the Q-Learning algorithm

from game import Game
from constants import (SCREEN_WIDTH, SCREEN_HEIGHT,
                       NUM_WIDTH_BINS, NUM_HEIGHT_BINS,
                       NUM_PLAYER_ACTIONS, PLAYER_ACTION_DICT,
                       NUM_EPISODES, ALPHA, GAMMA, MODEL_SAVE_PATH)
import numpy as np


class Model():
    def __init__(self, mode):
        # Initializing the model

        # mode is 0 for training, 1 for test
        self.mode = mode
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        # For this version of the game the player cannot move left/right
        self.num_height_bins = NUM_HEIGHT_BINS
        self.num_width_bins = NUM_WIDTH_BINS
        self.num_player_actions = NUM_PLAYER_ACTIONS
        self.player_action_dict = PLAYER_ACTION_DICT
        self.num_episodes = NUM_EPISODES
        # This is the learning rate
        self.alpha = ALPHA
        # This is the expected future return discount rate
        self.gamma = GAMMA
        self.model_save_path = MODEL_SAVE_PATH

        # For this version the player cannot move in the x direction
        num_player_x_bins = 0
        num_player_y_bins = self.num_height_bins
        # Technically the enemy starts off of the screen
        # so we need to allocate one more bin to take that into account
        num_enemy_x_bins = self.num_width_bins + 1
        num_enemy_y_bins = self.num_height_bins

        if not self.mode:
            # If we are in training mode we want the game not to be in human
            # or render mode, and we want to instantiate a new Q_Table

            human_mode = False
            render_mode = False
            self.game = Game(human_mode, render_mode)
            self.table = Q_Table(num_player_x_bins,
                                 num_player_y_bins,
                                 num_enemy_x_bins, num_enemy_y_bins,
                                 self.num_player_actions)
        else:
            # If we are in testing mode we want the game not to be in human
            # but to be in render mode (so we can watch)
            # and we want to load an existing Q table

            human_mode = False
            render_mode = True
            self.game = Game(human_mode, render_mode)
            self.table = Q_Table(num_player_x_bins,
                                 num_player_y_bins,
                                 num_enemy_x_bins, num_enemy_y_bins,
                                 self.num_player_actions)
            self.table.load_table(self.model_save_path)

    def train(self):

        for episode_num in range(self.num_episodes):
            # We need to reset the game after it is done to prepare for the
            # next episode
            self.game.reset()

            while not self.game.game_over:
                self.run_episode(episode_num)

            if (episode_num % 1000) == 0:
                print(f"{(episode_num / self.num_episodes) * 100}% done with training")

        self.table.save(self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

    def run_episode(self, episode_num):
        # observation: [player_pos, enemy_pos, enemies_survived, game_over,
        # victory]

        # Getting the current observation before the new episode
        # This is the current state s
        # I am only getting the current player and enemy positions
        # To minimize confusion
        (player_pos, enemy_pos, _, _, _) = self.game.get_observation()
        # We have to bin the positions for the table to use them
        player_pos_binned, enemy_pos_binned = self.bin_positions(player_pos,
                                                                 enemy_pos)
        # This is the action a that you take while in state s
        action = self.choose_action(episode_num, player_pos_binned,
                                    enemy_pos_binned)
        # Applying the new action
        self.game_step(action)
        # We have to distinguish the current position from the new position
        # because
        # the current position is used to calculate the value of the current
        # state
        # whereas the new position is used to predict the maximum possible
        # Q-value
        # immedaitely achievable from the current state
        (player_pos_new, enemy_pos_new, enemies_survived, game_over,
         victory) = self.game.get_observation()
        player_pos_new_binned, enemy_pos_new_binned = self.bin_positions(
                                                                player_pos_new,
                                                                enemy_pos_new)
        # Getting the new Q-Table value from the update calculation
        Q_val_updated = self.calculate_update(action,
                                              player_pos_binned,
                                              enemy_pos_binned,
                                              player_pos_new_binned,
                                              enemy_pos_new_binned,
                                              enemies_survived, game_over,
                                              victory)
        self.table.set(player_pos_binned, enemy_pos_binned, action,
                       Q_val_updated)

    def game_step(self, action):
        # Runs a single step of the game

        self.game.update(action)

    def calculate_update(self, action,
                         player_pos_binned, enemy_pos_binned,
                         player_pos_new_binned, enemy_pos_new_binned,
                         enemies_survived, game_over, victory):
        # The overall update function is given by
        # Q(s,a) <- Q(s, a) + alpha[R + gamma*M - Q(s, a)]
        # Where Q(s, a) is the former Q-value estimation
        # alpha is the learning rate
        # R is the immediate reward
        # gamma is the discount rate
        # M is the Q-value of the optimal next state

        # The model always gets a raw reward based on how many enemies it has
        # survived to incentivize moving forward
        # and then subtracts from value if it lost
        # and adds more if it won
        raw_reward = enemies_survived
        if game_over:
            # (victory - 0.5) * 20 because victory is 0 or 1
            # which is then converted to -10 or 10
            raw_reward += (victory - 0.5) * 20

        Q_val = self.table.get(player_pos_binned, enemy_pos_binned, action)
        possible_values = []
        for key in self.player_action_dict.keys():
            possible_action = self.player_action_dict[key]
            possible_value = self.table.get(player_pos_new_binned,
                                            enemy_pos_new_binned,
                                            possible_action)
            possible_values.append(possible_value)
        M = np.max(possible_value)
        # Calculating the updated Q-value based on the formula described above
        Q_val_updated = (Q_val +
                         self.alpha * (raw_reward + self.gamma * M - Q_val))
        return Q_val_updated

    def choose_action(self, episode_num, player_pos_binned, enemy_pos_binned):
        # During training we need to choose an action from the table according to
        # the epsilon-greedy policy
        # Epsilon represents the odds of choosing a random move
        # During test we choose the best action (the highest Q-value) from the
        # current state.
        # To do both of these in the same function during testing we can pass in
        # episode_num = 100 * self.num_episodes which will make the odds of the
        # random action triggering arbitrarily close to 0

        # The value of epsilon here is defined as a negative exponential
        # as a function of the episode number
        epsilon = np.exp(-5 * (episode_num / self.num_episodes))

        action = None

        if epsilon > np.random.rand():
            # choose a random action
            action = np.random.randint(0, self.num_player_actions)
        else:
            # choose the best action
            no_op_val = self.table.get(player_pos_binned, enemy_pos_binned,
                                   self.player_action_dict['NO-OP'])
            up_val = self.table.get(player_pos_binned, enemy_pos_binned,
                                self.player_action_dict['UP'])
            down_val = self.table.get(player_pos_binned, enemy_pos_binned,
                                  self.player_action_dict['DOWN'])

            # This isn't best practice because I am hard-coding the actions
            # with their values rather than using the action dictionary
            # but doing otherwise is too much of a headache
            action = self.argmax_random_on_ties([no_op_val, up_val, down_val])

        return action

    def bin_positions(self, player_pos, enemy_pos):
        # Puts the player and enemy positions into their correct bins

        x_compression_ratio = self.screen_width / self.num_width_bins
        y_compression_ratio = self.screen_height / self.num_height_bins

        player_x_bin = (np.floor(player_pos[0] / x_compression_ratio)
                        .astype(int))
        player_y_bin = (np.floor(player_pos[1] / y_compression_ratio)
                        .astype(int))
        enemy_x_bin = (np.floor(enemy_pos[0] / x_compression_ratio)
                       .astype(int))
        enemy_y_bin = (np.floor(enemy_pos[1] / y_compression_ratio)
                       .astype(int))

        return (player_x_bin, player_y_bin), (enemy_x_bin, enemy_y_bin)

    def argmax_random_on_ties(self, arr):
        # The built-in np.argmax is problematic because, in the event of a tie,
        # it always picks the first indexed item. This is could potentially
        # cause issues in the Q-table which is initially initialized to all 0
        # and so there would be lots of ties. So, the "random" action would be
        # heavily biased towards the first action in the array, no-op.

        actions = np.asarray(arr)
        max_action = np.max(actions)
        # We need the indexies of the max elements to randomly select from them
        candidates = np.where(actions == max_action)
        # np.where returns a tuple, so we only want the first item
        action = np.random.choice(candidates[0])

        return action

    def test(self):
        # Tests the model by playing the game

        self.game.reset()
        while not self.game.game_over:
            (player_pos, enemy_pos, _, _, _) = self.game.get_observation()
            player_pos_binned, enemy_pos_binned = self.bin_positions(player_pos, enemy_pos)

            # During testing
            # we need to pass in an arbitrarily high episode (obviously there is only 1 episode)
            # to always trigger a deterministic best action
            # See the description of self.choose_action() for more details
            fake_episode = self.num_episodes * 100
            action = self.choose_action(fake_episode, player_pos_binned, enemy_pos_binned)
            self.game_step(action)

        print("GAME OVER")
        if self.game.victory:
            print("The model WON")
        else:
            print("The model LOST")

class Q_Table():
    # Separating Q-table from model for the benefit of future implementations
    # The Q_Table class will hold the table and handle updating it
    def __init__(self, num_player_x_bins, num_player_y_bins,
                 num_enemy_x_bins, num_enemy_y_bins,
                 num_player_actions):
        # Initializing the Q-Table object
        # I am just hard-coding for now that player_x_bins is 0

        self.num_player_x_bins = 0
        self.num_player_y_bins = num_player_y_bins
        self.num_enemy_x_bins = num_enemy_x_bins
        self.num_enemy_y_bins = num_enemy_y_bins
        self.num_player_actions = num_player_actions

        # Always initialize a fresh table. This may be overwritten
        self.init_table()

    def init_table(self):
        # Initalizes the table itself
        # For this implementation the table will have dimension
        # player_y_bin, missile_x_bin, missile_y_bin, action

        # Ignoring player_x_bins for now
        # The dimensions must be passed in as a tuple
        table = np.zeros((self.num_player_y_bins,
                         self.num_enemy_x_bins, self.num_enemy_y_bins,
                         self.num_player_actions))
        # It is possible for any of the number of bins above to be 0
        # So we have to squeeze the table to delete this dimension
        # self.table = np.squeeze(table)

        self.table = table

    def get(self, player_pos_binned, enemy_pos_binned,
            action):
        # This gets a Q-value based on the player's coordinates, the enemy's
        # coordinates, and the player's action

        (player_x_bin, player_y_bin) = player_pos_binned
        (enemy_x_bin, enemy_y_bin) = enemy_pos_binned

        # Ignoring player x bins
        return self.table[player_y_bin, enemy_x_bin, enemy_y_bin,
                          action]

    def set(self, player_pos_binned, enemy_pos_binned,
            action, value):
        # This sets a Q-value based on the player's coordinates, the enemy's
        # coordinates, and the player's action

        (player_x_bin, player_y_bin) = player_pos_binned
        (enemy_x_bin, enemy_y_bin) = enemy_pos_binned

        # Ignoring player_x_bin
        self.table[player_y_bin, enemy_x_bin, enemy_y_bin,
                   action] = value

    def save(self, path):
        # Saves its own table to a file

        np.save(path, self.table)

    def load_table(self, path):
        # Loads a saved Q-table from an existing table
        # at a .npy path

        table = np.load(path)
        self.table = table
