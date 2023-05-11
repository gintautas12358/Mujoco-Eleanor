## Run PopSAN

- `python RL/popsan/sac_cuda_norm.py --env {env_name}`

It generated a param/ and a runs/ folder for the following agent

## Tensorboard

- `tensorboard --logdir ./runs`

## Agent evaluation

Command for classical gym environments:
- `python RL/test_record_classic_agent.py`

Command for custom gym environments:
- `python RL/test_record_agent.py`


