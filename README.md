# Meta Reinforcement Learning for Smart Building Control Using Sinergym

This repository contains the implementation of Meta-Reinforcement Learning (Meta-RL) for optimizing HVAC control in smart buildings using the Sinergym simulation environment. The project explores the Reptile algorithm, a Meta-RL technique, to improve adaptability across different building and weather conditions. The goal is to reduce energy consumption while maintaining thermal comfortability.

## Key Features

- **Meta-RL Framework**: Implements the Reptile algorithm for fast adaptation to new tasks.

- **Task Distributions**: Two approaches are explored:
  - **Building-based**: Training on different building types.
  - **Weather-based**: Training on varying weather conditions.

- **Inner-Loop DQN**: Uses Deep Q-Networks (DQN) for policy optimization within the Meta-RL framework.

- **Sinergym Integration**: Leverages the Sinergym environment, based on EnergyPlus, for realistic building simulations.

run the agent.py

to train using "weather" task distribution:
```bash
python agent.py reptile --train --mode weather
```

to train using "building" task distribution:
```bash
python agent.py reptile --train --mode building
```

to test the algorithm on the weather task distribution:
```bash
python agent.py reptile --mode weather
```

test on building distribution:
```bash
python agent.py reptile --mode building
```

You need to train before you can test. 
