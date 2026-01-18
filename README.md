# Neural-Sheet Agent for the Collapsing Bridge Environment

This repository provides an implementation of the Collapsing Bridge grid-based environment together with a neural-sheet encoder and a linear value-based agent. The environment models irreversible failures, action-dependent dynamics, and partial observability. The agent represents states on a sparse 2D neural sheet and learns a value function over these activations. The code targets safe and explainable reinforcement learning experiments in a compact, configurable setting.

## Features

### Collapsing Bridge environment (`bridge_env.py`)

- Gymnasium-compatible episodic grid world with walls, water, and bridge tiles that collapse into water after the agent leaves them.
- Object delivery task in which the agent picks up an object and delivers it to a fixed goal while avoiding catastrophic water cells.
- Time-of-day flag (morning/evening) that moves the object between two corners, so the optimal behaviour depends on a global condition.
- Partial observability through a 3×3 perceptual window centered on the agent, plus position, time, and object-carry indicators.
- Optional nondeterministic transitions and an “impossible” layout variant for stress-testing robustness and safety.

### Neural-sheet encoder (`neural_sheet_encoder.py`)

- Maps a continuous input vector in the unit hypercube to a 2D neural sheet with biologically inspired structure.
- Uses Gaussian population codes per input dimension, a sparse random projection to the sheet, and recurrent lateral inhibition.
- Applies percentile-based sparsification to produce a sparse code that lives on the sheet.
- Provides a Numba-accelerated implementation of the inhibition dynamics together with a pure NumPy fallback.
- Centralises configuration of dimensionality, sheet size, sparsity level, and dynamics in `settings.py`.

### Neural-sheet Q agent (`bridge_neural_sheet_agent.py`)

- Encodes `BridgeEnv` observations as a fixed 69-dimensional vector that matches the structure of the environment’s observation space.
- Feeds this vector into the neural-sheet encoder and flattens the sheet activation into a feature vector for value approximation.
- Implements a linear action-value function over neural-sheet features with epsilon-greedy exploration.
- Provides separate routines for first-visit Monte Carlo updates and one-step temporal-difference updates.
- Includes utilities for saving and restoring agent state, random number generator state, and encoder parameters.

### Training and evaluation pipeline (`train_bridge_neural_sheet.py`)

- High-level training loop that instantiates the environment, encoder, and agent based on a `TrainConfig` structure.
- Uses first-visit Monte Carlo regression by default, with reward scaling and simple clipping for numerical stability.
- Maintains a buffer of successful episodes and replays them periodically to address sparse rewards.
- Integrates off-policy updates from fixed expert trajectories stored in separate files for morning and evening scenarios.
- Periodically evaluates the greedy policy on four fixed cases (train/test × morning/evening) and supports early stopping when all succeed.
- Writes logs to the console and to an output directory along with a consolidated state file for later reuse.

### Representation and topology tests

- `neural_sheet_encoder_test_3d.py`: tests the topological structure of the encoder in low dimensions, using line segments and vertices of squares, cubes, and hypercubes as probes.
- `neural_sheet_encoder_test_69d.py`: studies similarity on full 69-dimensional bridge-style inputs, including continuous position changes, mixed discrete/continuous perturbations, and varying numbers of local cell changes.

### Configuration and utilities (`settings.py` and helpers)

- Central configuration of input dimensionality, number of detectors per dimension, lateral inhibition kernel, sparsity percentile, and noise injection.
- Small utility functions and helpers for discretising observations, loading expert trajectories, and logging to multiple output streams.

## Citation

If you use this environment, the neural-sheet encoder, or the agent in your research, please cite:

> Florin Leon, *Sparse Neural Code Representations for Reinforcement Learning with Linear Action-Value Function Approximation*, 2026.

## License

This project is provided under the MIT License.
