from model import Model


def generate_models(num_episodes_arr):
    # Takes in a list of numbers of episodes and trains a model
    # for that many episodes
    # saves the model at models/"Q_table_" + NUM_EPISODES + "_episodes.npy"

    # We are training and not rendering
    testing = False
    render = False

    for num_episodes in num_episodes_arr:
        path = f"models/Q_table_{num_episodes}_episodes.npy"
        print(f"Training model for {num_episodes} episodes")
        print(f"The model will be saved to {path}")

        model = Model(testing, render, num_episodes, path)
        model.train()


def test_models(model_path_arr):
    # Gets the winrate of a series of models
    # Runs each model for 1000 episodes
    # I am testing these with deterministic=False because otherwise
    # each model will have either a 100% victory rate or a 0% victory rate

    # We want to test but not render. Rendering is extremely slow
    testing = True
    render = False
    deterministic = True
    num_test_episodes = 100
    point_arr = []
    for path in model_path_arr:
        points = 0
        # Setting the episode element of Model to 0 so as to not
        # get confused with the num_test_epsidodes above

        model = Model(testing, render, 1, path, deterministic)
        for episode in range(num_test_episodes):
            model.test()
            # Victory is 1 for win, 0 for lose
            victory = model.game.victory
            points += victory

        point_arr.append(points / num_test_episodes)

    return point_arr


#generate_models([0, 100, 1000, 10000, 100000])
model_path_arr = ["models/Q_table_0_episodes.npy", "models/Q_table_100_episodes.npy",
                  "models/Q_table_1000_episodes.npy", "models/Q_table_10000_episodes.npy",
                  "models/Q_table_100000_episodes.npy"]
point_arr = test_models(model_path_arr)
print(point_arr)
