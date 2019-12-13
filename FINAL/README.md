# Comparision of SNN and ANN in Reinforcement Learning (Pong-v0)

Final project of computational cognitive neuroscience

## Setup

Some needed library

```
pip3 install -r requirements.txt
```

## Training

To train the agent playing Pong-v0 (OpenAI Gym).

### ANN

Train the agnet by config

```
python3 agent_play_ann.py ./configs/policy_gradient.yaml
```

### SNN

Train the agnet by config

```
python3 agent_play_snn.py ./configs/example_snn.yaml
```

## Evaluation

Temporarily no
