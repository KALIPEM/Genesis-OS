# Genesis-OS

This repository contains simple examples for running a toy universe simulation and training a tiny neural network. It also showcases experimental spin-based neural network components.

## Installation

Ensure Python 3 is installed. Install the required packages with:

```bash
pip install torch numpy
```

## Running the universe simulation

Execute `universe.py` to launch a short random walk simulation:

```bash
python universe.py
```

Example output:

```
Step 0: 0.123
Step 1: -0.456
...
Simulation complete!
```

## Training the basic spin neural network

Run the training script:

```bash
python train_spin_network.py
```

Expected output (truncated):

```
Epoch 0 Loss: 0.9
Epoch 20 Loss: 0.2
...
Training complete!
```

## Spin Network

The `SpinNeuron` rotates its input by a Fibonacci-scaled golden angle before applying a linear transformation. `SpinNetwork` stacks these neurons to form a feed-forward model. A simple training example can be found in `examples/train_spin.py`.

These scripts are minimal demonstrations and can be extended for more complex experiments.
