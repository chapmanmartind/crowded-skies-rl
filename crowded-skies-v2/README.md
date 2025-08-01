# CROWDED SKIES

A personal project to learn Reinforcement Learning and improve my software engineering skills.
This project comprises a simple projectile dodging game and a Deep Q-Learning (DQN) Reinforcement model on top of it.
The model was built from scratch and does not use any RL libraries. The game is built on top of pygame, a minimal game library.
You can play the game or watch the model play it. You can also train new "weights" for the model for yourself, and watch the model play on those weights.

# Install

1. Clone the repo:
```bash
    git clone https://github.com/chapmanmartind/crowded-skies-v2.git
    cd crowded-skies-v2
```

2. Install dependencies
```bash
    pip install -r requirements.txt
```

# Play the game or watch the model play

To play the game yourself run
```bash 
    python3 main.py play
```

To watch the model play the game run
```bash
    python3 main.py run
```

To watch a specific model (number of training episodes) play the game run
```bash
    python3 main.py run {model_path}
```

To train the model run
```bash
    python3 main.py train {number of episodes}
```

The default model is "pretrained_500k_episodes.pth". Training the model for 500k episodes took me a week on my Mac without GPU usage.