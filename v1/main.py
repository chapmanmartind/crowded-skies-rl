from model import Model
from game import Game
import sys


if __name__ == "__main__":

    path = "models/Q_table_100000_episodes.npy"
    if len(sys.argv) < 2:
        print("Please pass in the command you would like to run.\n ")
        print("Try 'play' to play the game yourself or 'run' to run the model on the game.\n")
        print("To run the model you can pass in a second argument to specify the model path\n")
        print(" or omit a second argument to use the default path")

    elif sys.argv[1] == "play":
        human_mode = True
        render_mode = True
        game = Game(human_mode, render_mode)
        while not game.game_over:
            game.update()
        if game.victory:
            print("You won!")
        else:
            print("You lost! Try again")

    elif sys.argv[1] == "run":
        testing = True
        render = True
        # Just setting deterministic = True for now
        deterministic = True
        if len(sys.argv) > 2:
            path = sys.argv[2]
        model = Model(testing, render, 1, path, deterministic)
        model.test()
        if model.game.victory:
            print("The model won!")
        else:
            print("The model lost")
