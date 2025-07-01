# CROWDED SKIES

A personal project to learn Reinforcement Learning and improve my software engineering skills
This project comprises a simple projectile dodging game and a Q-Learning Reinforcement model on top of it.
The model was built from scratch and does not use any RL libraries. The game is built on top of pygame, a minimal game library
You can play the game or watch the model play it. You can also train new "weights" for the model for yourself. 

# Install

1. Clone the repo:
   ```bash
   git clone https://github.com/chapmanmartind/crowded-skies-v1.git
   cd crowded-skies-v1
```

2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

# Play the game or watch the model play

To play the game run
```bash 
python3 main.py play```

To watch the model play the game run
```bash
python3 main.py run
```

To watch a specific model (number of training episodes) play the game run
```bash
python3 main.py run {model_path}
```

There are several models in the models directory. "Q_table_0_episodes.npy" corresponds to completely random behavior and serves as a baseline. 

See game.py for the game code, model.py for the model code, and testing.py for some auxilliary training and testing functions.