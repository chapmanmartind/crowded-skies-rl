# In this model we will implement the Q-Learning algorithm

from game import Game
from constants import (SCREEN_WIDTH, SCREEN_HEIGHT, HEADER_HEIGHT,
                       OBSERVATION_LEN, REPLAY_BUFFER_CAPACITY, BATCH_SIZE, GAMMA, TAU,
                       NUM_PLAYER_ACTIONS)
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import random


class DQN():
    def __init__(self, testing, render, num_episodes, model_save_path,
                 channels=[OBSERVATION_LEN, 128, 128, NUM_PLAYER_ACTIONS], deterministic=True):

        # This Deep Quality Network class will manage collecting data, training, and move execution

        # Testing is True for testing, False for training
        self.testing = testing

        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        self.header_height = HEADER_HEIGHT

        # Render is True for rendering the screen and gameplay, False for no render (training and evaluation)
        self.render = render
        self.num_episodes = num_episodes
        self.model_save_path = model_save_path
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        self.num_player_actions = NUM_PLAYER_ACTIONS
        self.observation_len = OBSERVATION_LEN
        self.replay_buffer_capacity = REPLAY_BUFFER_CAPACITY
        # The DQN will utilize a replay buffer which is an array of the following
        # state, action, reward, following state, and done
        # (a boolean that indicates if the game finished after this state)
        # I am going to define a blank replay and fill the replay buffer with this.
        # I know this is not ideal and trains the model on false data
        # However, by doing this I don't have the handle the logic of
        # only selecting from parts of the buffer that have been written to
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

        # This will be used in testing
        self.losses = []
        self.episode_lens = []

        self.network = Network(channels, model_save_path)
        self.target_network = Network(channels, "")
        self.target_network.load_state_dict(self.network.state_dict())

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

        # We are going to define a previous reward to hold the reward from the previous frame
        # This will allow us to calculate reward jumps
        previous_reward = torch.tensor([0])

        personal_best = 10000

        # Episode should start at 1 because 0th episode isn't logical, it's the first episode
        # Also useful later to prevent checking loss on 0th episode
        for episode in range(1, self.num_episodes):
            self.game.reset()

            while not self.game.game_over:
                # Here we are updating the gameplay buffer by playing the game with the network's current weights

                self.num_frames += 1

                # I am going to carefully document the shapes of the tensors in this section for clarity and debugging
                # I want all the values to be 1D arrays for ease when I unpack the dictionary

                state = torch.as_tensor(self.game.get_observation(), dtype=torch.float32)  # 30,

                # The network predicts the rewards associated with each move in this state
                rewards = self.network(state).detach().clone()  # 3,

                # We pick our next action according to our epsilon greedy strategy
                # The action is always going to be a torch tensor so we detach and clone
                action = self.choose_action(rewards, episode)  # 1,

                self.game.update(action.item())
                state_after = torch.as_tensor(self.game.get_observation(), dtype=torch.float32)  # 30,

                reward = torch.tensor([self.calculate_reward(state_after)])  # 1,

                game_over_int = torch.tensor([self.game.game_over])  # 1,

                transition = {'state': state,
                              'action': action,
                              'reward': reward,
                              'state_after': state_after,
                              'game_over_int': game_over_int}

                # We are going to implemented a weighted transition assignment to set more entries of the replay buffer
                # to important (large reward jump) transitions
                self.update_replay_buffer(transition, reward, previous_reward)

                # We only want to train if the replay buffer has already been completely filled
                # and we have enough new replays
                # TRYING OUT self.batch size / 4 CHANGE FOR MORE FREQUENT TRAINING
                if (self.num_frames > self.replay_buffer_capacity) and (not self.num_frames % (self.batch_size / 4)):
                    loss = self.update_network()
                    self.losses.append(loss)

                    # Updating the network via Soft (Polyak) updates to avoid sudden network changes
                    for p, p_bar in zip(self.network.parameters(), self.target_network.parameters()):
                        p_bar.data.mul_(1 - self.tau)
                        p_bar.data.add_(self.tau * p.data)

                previous_reward = reward

            # Adding the frame which the game ended at to see game length progression
            self.episode_lens.append(self.game.frame)

            if self.game.frame > personal_best:
                print("\n New personal best: ", self.game.frame)
                personal_best = self.game.frame

            if not (episode % 100):
                print(f"The recent avg game length is {np.sum(self.episode_lens[-100:]) / 100}")
                print(f"The recent avg loss is {np.sum(self.losses[-100:]) / 100}")
                print(f"{episode / self.num_episodes * 100}% done with training")
                print("\n\n")

        self.network.save()
        print("Training completed and network saved")
        print("\nOverall personal best: ", personal_best)
        return 1

    def test(self):
        # This function runs the model on the weights

        self.game.reset()
        # Setting episode to high to make sure always best move
        episode = 1000

        # If we are rendering we don't want to exit automatically
        if self.render:
            while 1:
                state = torch.tensor(self.game.get_observation())
                rewards = self.network(state)
                action = self.choose_action(rewards, episode)
                self.game.update(action.item())

        else:
            while not self.game.game_over:
                state = torch.tensor(self.game.get_observation())
                rewards = self.network(state)
                action = self.choose_action(rewards, episode)
                self.game.update(action.item())

        print("Game ended at frame: ", self.game.frame)

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

        # x will represent what percent of training we are finished with
        x = episode / self.num_episodes

        # Based on testing this is the best epsilon pattern.
        # Try starting at .25 and decreasing to .01
        # Apparently we always want a minimum randomness of .01 to explore possibilities
        epsilon = max(.25 - (x / 4), .01)

        # I want to give the model a decent amount of space at the end to train its final version
        # This section will also be used during testing
        if (episode / self.num_episodes) > .9:
            epsilon = 0

        action = None

        if epsilon > torch.rand(1).item():
            # choose a random action
            action = torch.randint(0, self.num_player_actions, (1, ))
        else:
            # choose the best action

            # Instead of using my random function I want to use the built in random which biases towards
            # the first element
            # Which is most likely to be no-op, and then up, and then down. This is exactly the subtle bias I want
            # keepdim = True because I want it to be a 1D array like in the previous action
            action = torch.argmax(rewards, keepdim=True)

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

        player_pos = state[:2]

        # Adding distance from center as part of reward because having trouble with going out of bounds
        distance_from_center = abs(player_pos[1] - 0.5)

        frames, enemies_spawned, oob, collision, game_over, victory = state[-6:]
        # victory is 0 by default so only give bonus if game is over
        victorious = (game_over and victory)
        # This weighting is based on extensive testing. I'm not sure why this is what works
        not_dead = not collision
        reward = (.1 * not_dead) + (-0.1 * distance_from_center) + (-1.0 * collision) + (1.0 * victorious)

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

        # We want to use game_over_ints to create a mask. If the game is over we want that section to be 0, 1 otherwise
        mask = 1 - game_over_ints.float()  # 32 x 1
        # We want to get the predicted rewards for our model for the actions we took
        # then we want to compare these rewards against the actual rewards we got, modified by
        # a prediction of future rewards
        outputs = self.network(states)  # 32 x 3 because network's output dim is 3
        # We then want to index into these outputs and select only the values prescriped by actions (indexes)
        predictions = outputs.gather(1, actions)  # 32 x 1
        # The network outputs the predicted Q values for making a move in a given state
        # We will use this to form the target. The logical flaw in this is that we are using the network's
        # prediction as the target for which the network will evaluate itself against
        # Now trying to use target_network
        max_future_values, _ = torch.max(self.target_network(state_afters).detach().clone(), dim=1, keepdim=True)
        # 32 x 3 --> 32 x 1

        # I want only want to add the future value factor if the game is NOT already over. If the game
        # is already over then the future value factor is 0
        # This is why we created inverted_game_over_ints to work as a mask
        targets = rewards + self.gamma * (max_future_values * mask)  # 32 x 1

        loss = self.network.update(predictions, targets)

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

    def update_replay_buffer(self, transition, reward, previous_reward):
        # We want to weight transitions with large reward jumps more heavily than transitions with small reward jumps
        # Large reward jumps indicate that this particular move is more important than previous moves

        # We want the replay buffer to be overwritten in a round robin pattern
        idx = (self.num_frames - 1) % self.replay_buffer_capacity
        self.replay_buffer[idx] = transition

        # To provide more weight to large jump transitions
        # we are going to set extra replays from the buffer to these transitions
        reward_weighting = int(abs((reward.item() - previous_reward.item()) * 10))
        extra_transitions = max(0, reward_weighting)
        idxs = random.sample(range(self.replay_buffer_capacity), extra_transitions)
        for idx in idxs:
            self.replay_buffer[idx] = transition


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

        self.L1 = nn.SmoothL1Loss()
        # Trying an extremely slow learning rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00005)

    def forward(self, x):
        x1 = self.relu(self.linear1(x))
        x2 = self.relu(self.linear2(x1))

        # It would be incorrect to use a ReLU after the final linear layer because Q values below 0 are valid
        # ReLU clips values below 0 to 0
        x3 = self.linear3(x2)
        return x3

    def loss(self, prediction, target):
        # Calculates the L2 loss for a prediction and a target
        loss = self.L1(prediction, target)
        return loss

    def update(self, prediction, target):
        # Updates the gradient based on the loss

        loss = self.loss(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()

        # Trying gradient clipping for stability
        clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    def save(self):
        torch.save(self.state_dict(), self.model_path)

    def load(self):
        self.load_state_dict(torch.load(self.model_path))
