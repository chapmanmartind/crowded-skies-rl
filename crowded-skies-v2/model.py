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
import random

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
        # So a single replay has the form [[state], action, reward, [following state], done]
        # And each state has the form [[(x, y), (xv, yv)], [[(xe, ye), (xev, yev), etype] x 4], f, n, b, c, g, v]
        # which will be unraveled into an array of length 30 = OBSERVATION_LEN
        # In total each replay will have length 2 * OBSERVATION_LEN + 3
        # The replay buffer will hold flattened (1D) observations
        self.replay_buffer = torch.zeros(self.replay_buffer_capacity, 2 * self.observation_len + 3)
        self.network = Network(channels, model_save_path)

        self.human_mode = False
        no_enemies = False

        game = Game(self.human_mode, self.render, no_enemies)
        self.game = game

    def train(self):
        # Training the DQN
        # In this model we are collecting data and training simultaenously. We collect a batch size's worth of data
        # then randomly sample the training buffer and use that data as our training data
        # Note that all functions work on flattened observations and not as they come from the game

        replay_num = 0
        for episode in range(self.num_episodes):
            self.game.reset()

            while not self.game.game_over:
                # Here we are updating the gameplay buffer by playing the game with the network's current weights
                # Get the initial observation to form our transition
                # The state will now come as a flat tensor of length 30
                state = torch.tensor(self.game.get_observation())
                # The network predicts the rewards associated with each move in this state
                rewards = self.network(state)
                # We pick our next action according to our epsilon greedy strategy
                # The action is always going to be a torch tensor so we detach and clone
                action = torch.tensor(self.choose_action(rewards, episode)).detach().clone().unsqueeze(0)
                self.game.update(action.item())
                state_after = torch.tensor(self.game.get_observation())
                reward = torch.tensor(self.calculate_reward(state_after)).unsqueeze(0)

                # A single transition has the form [[state], action, reward, [following state], done]
                # The single values have to be unsqueezed into arrays
                transition = [state, action, reward, state_after, torch.tensor(int(self.game.game_over)).unsqueeze(0)]
                transition_flat = torch.cat(transition)
                self.replay_buffer[replay_num] = transition_flat

                # Now we want to train if we have hit enough new replays
                if not (replay_num % self.batch_size):
                    self.update_network()
                replay_num += 1
                replay_num = replay_num % self.replay_buffer_capacity

            if not (episode % 100):
                print(f"{episode / self.num_episodes * 100}% done with training")

        self.network.save()

    def choose_action(self, rewards, episode):
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
            action = torch.randint(0, self.num_player_actions, (1, )).item()
        else:
            # choose the best action
            # This isn't best practice because I am hard-coding the actions with their values rather than using
            # the action dictionary but doing otherwise is too much of a headache
            # Converting to torch.tensor to use tensors throughout
            action = self.argmax_random_on_ties(torch.tensor([rewards[0], rewards[1], rewards[2]]))

        return action

    def argmax_random_on_ties(self, arr):
        # The built-in np.argmax is problematic because, in the event of a tie it always picks the first indexed item. This is could potentially
        # cause issues in the buffer which is initially initialized to all 0 and so there would be lots of ties. So, the "random" action would be
        # heavily biased towards the first action in the array, no-op. We therefore want a truly random tiebreaker
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
        sample_idxs = torch.randperm(len(self.replay_buffer))[:self.batch_size]
        samples = self.replay_buffer[sample_idxs]
        
        targets = torch.empty(self.batch_size)
        predictions = torch.empty(self.batch_size)

        for i in range(len(samples)):
            sample = samples[i]
            # We need to process each sample
            # If the sample is the final frame then we cannot calculate the target reward value
            # Otherwise we use a similar formula to in Q Table model
            # target = r + gamma * (max next move Q val)
            # [state, action, reward, state_after, done] = sample
            # Note that state has length self.observation_len
            state = sample[:self.observation_len]

            # We only want the value of the action, not the tensor itself
            # Action has to be an int also because we are using it as an index
            action = int(sample[self.observation_len].item())
            reward = sample[self.observation_len + 1].item()
            state_after = sample[self.observation_len + 2: self.observation_len + 2 + self.observation_len]
            done = sample[self.observation_len + 2 + self.observation_len:].item()
            target = None
            if done:
                target = reward
            else:
                # The network outputs the predicted Q values for making a move in a given state
                # We will use this to form the target. The logical flaw in this is that we are using the network
                # prediction as the target for which the network will evaluate itself against
                # We need to convert the output of the network to a np array before taking its max
                Q_max = torch.max(self.network(state_after)).item()
                
                target = reward + self.gamma * Q_max

            # prediction is the model's prediction of the Q value for taking the action we took in the state we were in
            prediction = self.network(state)[action]

            # And we compare that to the actual reward augmented by the predicted future reward
            targets[i] = target
            predictions[i] = prediction

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

testing = True
render = False
num_episodes = 100
model_save_path = "test.pth"
model = DQN(testing, render, num_episodes, model_save_path)
model.train()
