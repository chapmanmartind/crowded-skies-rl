# In this model we will implement the Q-Learning algorithm

from game import Game
from constants import (SCREEN_WIDTH, SCREEN_HEIGHT, NUM_WIDTH_BINS, NUM_HEIGHT_BINS,
                       OBSERVATION_LEN, REPLAY_BUFFER_CAPACITY, BATCH_SIZE, GAMMA,
                       NUM_PLAYER_ACTIONS, PLAYER_ACTION_DICT)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torchmetrics import MeanMetric
import random
import sys


class DQN():
    def __init__(self, testing, render, num_episodes, model_save_path,
                 channels=[OBSERVATION_LEN, 128, 128, NUM_PLAYER_ACTIONS], deterministic=True):

        # This Deep Quality Network class will manage collecting data, training, and move execution

        # Testing is True for testing, False for training
        self.testing = testing
        # Render is True for rendering the screen and gameplay, False for no render (training and evaluation)
        self.render = render
        self.num_episodes = num_episodes
        self.model_save_path = model_save_path
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.num_player_actions = NUM_PLAYER_ACTIONS
        self.observation_len = OBSERVATION_LEN
        self.replay_buffer_capacity = REPLAY_BUFFER_CAPACITY
        # The DQN will utilize a replay buffer which is an array of the following
        # state, action, reward, following state, and done (a boolean that indicates if the game finished after this state)
        # I am going to define a blank replay and fill the replay buffer with this. I know this is not ideal and trains the model on false data
        # However, by doing this I don't have the handle the logic of only selecting from parts of the buffer that have been written to
        # A single state comprises
        # - player position and velocity,
        # - enemy position, velocity, and type for every enemy on the screen
        # - number of frames
        # - number of enemies spawned
        # - out of bounds
        # - collision
        # - game over
        # - victory

        # The replay buffer object will be a list of dictionaries. Each dictionary represets a transition
        # transition = {state: val, action: val, reward: val, following_state: val, done: val}
        self.replay_buffer = [0] * self.replay_buffer_capacity
        # The full buffer flag will be used later when we only want to train after the buffer has been filled
        self.num_frames = 0

        self.network = Network(channels, model_save_path)
        if self.testing:
            # If we are testing we will load the model from the path
            self.network.load()

        self.human_mode = False
        no_enemies = False

        game = Game(self.human_mode, self.render, no_enemies)
        self.game = game

    def train(self):
        # Training the DQN
        # In this model we are collecting data and training simultaenously. We collect a batch size's worth of data
        # then randomly sample the training buffer and use that data as our training data
        # Note that all functions work on flattened observations and not as they come from the game

        loss_avg = MeanMetric()

        # Episode should start at 1 because 0th episode isn't logical, it's the first episode
        # Also useful later to prevent checking loss on 0th episode
        for episode in range(1, self.num_episodes):
            self.game.reset()

            while not self.game.game_over:
                # Here we are updating the gameplay buffer by playing the game with the network's current weights

                self.num_frames += 1

                # I am going to carefully document the shapes of the tensors in this section for clarity and debugging
                # I want all the values to be 1D arrays for ease when I unpack the dictionary

                state = torch.tensor(self.game.get_observation())  # 30,

                # The network predicts the rewards associated with each move in this state
                rewards = self.network(state).detach().clone()  # 3,

                # We pick our next action according to our epsilon greedy strategy
                # The action is always going to be a torch tensor so we detach and clone
                action = self.choose_action(rewards, episode)  # 1,

                self.game.update(action.item())
                state_after = torch.tensor(self.game.get_observation())  # 30,

                reward = torch.tensor([self.calculate_reward(state_after)])  # 1,
                print(reward.item())
                game_over_int = torch.tensor([self.game.game_over])  # 1,

                transition = {'state': state,
                              'action': action,
                              'reward': reward,
                              'state_after': state_after,
                              'game_over_int': game_over_int}

                # We want the replay buffer to be overwritten in a round robin pattern
                idx = self.num_frames % self.replay_buffer_capacity - 1
                self.replay_buffer[idx] = transition

                # We only want to train if the replay buffer has already been completely filled
                # and we have enough new replays
                if (self.num_frames > self.replay_buffer_capacity) and (not self.num_frames % self.batch_size):
                    loss = self.update_network()
                    loss_avg.update(loss)

            if not (episode % 100):
                print(f"The average loss is {loss_avg.compute().item()}")
                print(f"{episode / self.num_episodes * 100}% done with training")

        self.network.save()

    def test(self):
        self.game.reset()
        # Setting episode to high to make sure always best move
        episode = 1000000
        while not self.game.game_over:
            state = torch.tensor(self.game.get_observation())
            rewards = self.network(state)
            action = self.choose_action(rewards, episode)
            self.game.update(action.item())

    def choose_action(self, rewards, episode):
        # This function takes in rewards as a tensor and episode as an int
        # and outputs the action as a tensor

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
        epsilon = torch.exp(torch.tensor(-5 * (episode / self.num_episodes)))
        action = None

        if epsilon > torch.rand(1).item():
            # choose a random action
            action = torch.randint(0, self.num_player_actions, (1, ))
        else:
            # choose the best action
            # This isn't best practice because I am hard-coding the actions with their values rather than using
            # the action dictionary but doing otherwise is too much of a headache
            # Converting to torch.tensor to use tensors throughout
            action = torch.tensor([self.argmax_random_on_ties(rewards)])

        return action

    def argmax_random_on_ties(self, arr):
        # The built-in np.argmax is problematic because, in the event of a tie it always picks the first indexed item. This is could potentially
        # cause issues in the buffer which is initially initialized to all 0 and so there would be lots of ties. So, the "random" action would be
        # heavily biased towards the first action in the array, no-op. We therefore want a truly random tiebreaker
        # This function outputs an int, not a tensor

        max_values = torch.max(arr)
        # We need the indexies of the max elements to randomly select from them
        candidate_idxs = torch.where(arr == max_values)[0]
        # torch.where returns the indexes of the values that have the max value
        action = random.choice(candidate_idxs)

        return action

    def calculate_reward(self, state):
        # Calculates the raw reward given a state
        # Note that a single state comprises
        # - player position and velocity,
        # - enemy position, velocity, and type for every enemy on the screen
        # - number of frames
        # - number of enemies spawned
        # - out of bounds
        # - collision
        # - game over
        # - victory
        # Note that this function works on a FLATTENED state

        # In this function we only care about non-character items
        frames, enemies_spawned, oob, collision, game_over, victory = state[-6:]
        # victory is 0 by default so only give bonus if game is over
        victorious = (game_over and victory)
        # There are 50k frames in a completed game so we don't want to weight it too highly
        # This weighting doesn't have any real science behind it
        reward = (.001 * frames) + (5 * enemies_spawned) - (100 * oob) - (100 * collision) + (200 * victorious)

        return reward

    def update_network(self):
        # Calculates a network update and applies backpropagation

        # Using torch because it is good practice in case I want to use GPUs
        # We want to randomly sample batch_size samples from the buffer to preserve the expectation of the training set
        sample_idxs = torch.randperm(self.replay_buffer_capacity)[:self.batch_size]  # 32,
        samples = [self.replay_buffer[i] for i in sample_idxs]  # 32 dictionaries

        # Untangling the dictionary into tensors for parallelization
        states = torch.stack([sample['state'] for sample in samples])  # 32 x 30
        actions = torch.stack([sample['action'] for sample in samples])  # 32 x 1
        rewards = torch.stack([sample['reward'] for sample in samples])  # 32 x 1
        state_afters = torch.stack([sample['state_after'] for sample in samples])  # 32 x 30
        game_over_ints = torch.stack([sample['game_over_int'] for sample in samples])  # 32 x 1

        # It will be useful to invert the game over ints
        inverted_game_over_ints = ~ game_over_ints  # 32 x 1

        # We want to get the predicted rewards for our model for the actions we took
        # then we want to compare these rewards against the actual rewards we got, modified by
        # a prediction of future rewards
        outputs = self.network(states)  # 32 x 3 because network's output dim is 3
        # We then want to index into these outputs and select only the values prescriped by actions (indexes)
        predictions = outputs.gather(1, actions)  # 32 x 1

        # The network outputs the predicted Q values for making a move in a given state
        # We will use this to form the target. The logical flaw in this is that we are using the network's
        # prediction as the target for which the network will evaluate itself against
        max_future_values, _ = torch.max(self.network(state_afters), dim=1, keepdim=True)  # 32 x 3 --> 32 x 1

        # I want only want to add the future value factor if the game is NOT already over. If the game
        # is already over then the future value factor is 0
        # This is why we created inverted_game_over_ints to work as a mask
        targets = rewards + self.gamma * (max_future_values * inverted_game_over_ints)

        loss = self.network.update(targets, predictions)

        return loss

    def flatten_observation(self, observation):
        # It it helpful to have a function that recursively flattens our observation arrays

        result = []
        for item in observation:
            if isinstance(item, (list, tuple)):
                result.extend(self.flatten_observation(item))  # recursive flattening
            else:
                result.append(item)
        return result


class Network(nn.Module):
    # In this version of the model we are using a Deep Quality Network (DQN)
    # The network itself is a standard multilayer perceptron - a fully connected feed forward network
    def __init__(self, channels, model_save_path):
        super(Network, self).__init__()
        # The input state is very small so we can use a fully connected network
        # There is no need for any convolutional layers. Especially because we are not passing in any image data
        self.model_path = model_save_path

        self.linear1 = nn.Linear(channels[0], channels[1])
        self.linear2 = nn.Linear(channels[1], channels[2])
        self.linear3 = nn.Linear(channels[2], channels[3])

        self.relu = nn.ReLU()

        self.L2 = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x1 = self.relu(self.linear1(x))
        x2 = self.relu(self.linear2(x1))

        # It would be incorrect to use a ReLU after the final linear layer because Q values below 0 are valid
        # ReLU clips values below 0 to 0
        x3 = self.linear3(x2)
        return x3

    def loss(self, prediction, target):
        # Calculates the L2 loss for a prediction and a target
        loss = self.L2(prediction, target)
        return loss

    def update(self, prediction, target):
        # Updates the gradient based on the loss

        loss = self.loss(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save(self):
        torch.save(self.state_dict(), self.model_path)

    def load(self):
        self.load_state_dict(torch.load(self.model_path))


testing = False
render = False
num_episodes = 10000
model_save_path = "test.pth"
model = DQN(testing, render, num_episodes, model_save_path)
model.train()
