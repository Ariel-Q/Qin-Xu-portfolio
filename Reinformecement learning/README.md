
# Decentralized Multi-Agent Reinforcement Learning in Overcooked-AI

## Overview

This project explores decentralized multi-agent reinforcement learning (MARL) in the **Overcooked-AI** environment. Agents are trained to collaborate on cooking tasks under two main settings:

- **Homogeneous**: Two reinforcement learning (RL) agents train independently with the same algorithm.
- **Heterogeneous**: One RL agent is paired with a random-acting heuristic agent.

We implemented four RL algorithms from scratch:

- **Q-learning**
- **Deep Q-Network (DQN)**
- **Proximal Policy Optimization (PPO)**
- **Trust Region Policy Optimization (TRPO)**

The project investigates the effectiveness of each method under decentralized training conditions in two kitchen layouts of varying complexity.

---

## Project Structure

```
.
├── Project_within_proposal (1).ipynb        # Main training and evaluation notebook (partial layout)
├── kitchen_Project_within_proposal (1).ipynb # Additional environment customization and agent logic
├── COMP_579_Project_Report_complete.pdf     # Full project report with methodology, results, and discussion
└── README.md                                # Project summary and usage instructions
```

---

## Setup

### Dependencies

Make sure to install the following Python packages:

```bash
pip install numpy matplotlib torch gym tqdm
```

You will also need the Overcooked-AI Gym environment:

```bash
git clone https://github.com/ilvieira/overcooked-gym
cd overcooked-gym
pip install -e .
```

---

## How to Run

1. **Open the notebook**: Use Jupyter or VSCode to run `Project_within_proposal (1).ipynb`.
2. **Choose the layout**: Select between `simple_l` and `simple_kitchen` layouts.
3. **Select agent setting**:
   - `2-agent`: both agents are RL-based
   - `1-agent`: one agent is RL, one is a random heuristic
4. **Run training and evaluation**: The notebook supports training for all four RL algorithms with predefined hyperparameters.

---

## Key Features

- Full implementation of Q-learning, DQN, PPO, and TRPO from scratch
- Evaluation in decentralized settings without parameter sharing or centralized training
- Comparative analysis of algorithm robustness under sparse rewards and non-stationary partner behavior
- Grid-based Overcooked layouts simulating varying levels of task complexity

---

## Results Summary

- **Q-learning** showed reliable convergence in simple layouts but required extensive training for complex environments.
- **DQN** succeeded in simple layouts but struggled to generalize in more dynamic settings.
- **PPO and TRPO** consistently failed in decentralized settings due to conservative policy updates and lack of coordination signals.
- All algorithms failed to learn when paired with a randomly acting partner, highlighting the challenge of non-stationarity.

See the `COMP_579_Project_Report_complete.pdf` for detailed plots and explanations.

---

## Acknowledgements

This project uses the [Overcooked-Gym](https://github.com/ilvieira/overcooked-gym) wrapper for OpenAI Gym, originally designed to test cooperative behavior in constrained spatial tasks.

---

## Author

**Qin (Ariel) Xu**  
School of Information Science  
McGill University  
qin.xue@mail.mcgill.ca
