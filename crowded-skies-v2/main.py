from game import Game
from model import DQN
import sys
import numpy as np
import matplotlib.pyplot as plt


def run_play():
    # Runs the game to be played by the user

    human_mode = True
    render_mode = True
    no_enemies = False

    game = Game(human_mode, render_mode, no_enemies)
    while 1:
        game.update()
    if game.victory:
        print("You won!")
    else:
        print("You lost! Try again")


def run_test(model_save_path):
    # Runs the model with the weights at model_save_path

    testing = True
    render = True
    num_episodes = 1
    model = DQN(testing, render, num_episodes, model_save_path)
    model.test()
    if model.game.victory:
        print("The model won!")
    else:
        print("The model lost!")


def run_train(num_episodes):
    # Trains the model for num_episodes
    # and saves to models/{num_episodes/1000}k_episodes.pth

    testing = False
    render = False
    model_save_path = f"models/{num_episodes/1000}k_episodes.pth"
    model = DQN(testing, render, num_episodes, model_save_path)
    model.train()

    x = np.arange(model.num_episodes - 1)
    y = model.episode_lens

    window = 1000

    # Can only graph if data is long enough
    if len(y) >= window:
        # compute the rolling mean
        y_smooth = np.convolve(y, np.ones(window)/window, mode='valid')
        x_smooth = x[window-1:]

        plt.figure(figsize=(12, 4))
        plt.plot(x_smooth, y_smooth, linewidth=1.5)
        plt.xlabel('Episode')
        plt.ylabel(f'{window}-episode moving avg')
        plt.title('Smoothed Episode Length')
        plt.savefig('models/episode_len.png')
        plt.show()

        x = np.arange(len(model.losses))
        y = np.log(np.array(model.losses))
        # compute the rolling mean
        y_smooth = np.convolve(y, np.ones(window)/window, mode='valid')
        x_smooth = x[window-1:]

        plt.figure(figsize=(12, 4))
        plt.plot(x_smooth, y_smooth, linewidth=1.5)
        plt.xlabel('Time')
        plt.ylabel(f'{window}-episode moving avg')
        plt.title('Smoothed Loss')
        plt.savefig('models/loss.png')
        plt.show()

    print("DONE TRAINING")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("\nPlease pass in the command you would like to run.\n ")
        print("Try 'play' to play the game yourself")
        print("Try 'run' to watch the model run on its pretrained weights\n")
        print("You can also try 'train' to train the model. Pass in a number for the number of episodes to train for")
        print("Note that training the model for 500k episodes took a week!")
        print("If you want to test the model on non-default weights, pass in a path to 'run' ")
        print("Otherwise the model will default to the pretrained weights\n")

    elif sys.argv[1] == "play":
        run_play()

    elif sys.argv[1] == "run":
        if len(sys.argv) > 2:
            model_save_path = sys.argv[2]
        else:
            model_save_path = "models/pretrained_500k_episodes.pth"

        run_test(model_save_path)

    elif sys.argv[1] == "train":
        if len(sys.argv) > 2:
            num_episodes = int(sys.argv[2])
            run_train(num_episodes)
        else:
            print("Please provide the number of episodes to train for as an argument")

    else:
        print("Please provide a recognized argument to python3 main.py. Try 'play', or 'run'")
