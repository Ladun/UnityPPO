# UnityRL

This repository is a reinforcement algorithm implementation for unity ml-agents


# Support

## Algorithm
- ppo
- sac (under development)

## Environment
- Unity
- gym


# Train
```
python -m src.train -c=/path/to/config_file

# example
python -m src.train -c=configs/ppo/cralwer.yaml
```

# inference

```
python -m src.infer -c=/path/to/config_file

# example
python -m src.infer -c=configs/ppo/cralwer.yaml
```

# Directory Hierarchy
    .
    ├── configs
    ├── outputs
    ├── src
    │   ├── algorithm
    │   ├── environment
    │   ├── utils
    │   ├── infer.py
    │   └── train.py