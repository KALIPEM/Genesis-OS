# Genesis-OS

This repository contains experimental neural network components.

## Spin Network

The `SpinNeuron` rotates its input by a Fibonacci-scaled golden angle before
applying a linear transformation. `SpinNetwork` stacks these neurons to form a
feed-forward model. A simple training example can be found in
`examples/train_spin.py`.
