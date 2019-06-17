# Adaptive Learning Rate Clipping (ALRC)

Repository for the [paper](link goes here) "Adatpive Learning Rate Clipping Stabilizes Learning". 

This repository contains source code for CIFAR-10 supersampling experiments with squared and countic errors. There is also an implementation of the ALRC algorithm in `alrc.py`. Some source code for partial-STEM experiments is [here](https://github.com/Jeffrey-Ede/partial-STEM).

# Description

ALRC is a simple, computationally inexpensive algorithm that stabilizes learning by limiting loss spikes. It can be applied to any neural network trained with stochastic gradient descent. In practice, it improves the training of neural networks where learning is destabilized by loss spikes and otherwise has little effect.

If you are unsure whether to use ALRC, you should. It is computationally inexpensive, designed to complement existing learning algorithms and, at worst, will do nothing. Large improvements in the rate of convergence are seen when learning otherwise has high loss spikes.

# Example

ALRC can be applied like any other neural network layer

```python
loss = my_loss_fn( ... ) #Apply neural network and infer loss
loss = alrc(loss) #Apply ALRC to stabilize learning with default parameters
```

ALRC is clipping is robust to hyperparamer choices. The only hyperparameter that needs to be changed is an initial estimate for the first two raw moments of the loss function. If you're usure, run your neural network for 10+ iterations without ALRC to get rough estimates. Any reasonable overestimates are fine: even if they are an order of magnitude too high, the ALRC algorithm will decay them to the correct values. Don't underestimate.

```python
#Roughly estimate the first two raw moments of the loss function
mu1_start_estimate = 1.
mu2_start_estimate = 1.5

#It's fine to overestimate
overestimate_factor = 3 
mu1_start_estimate *= overestimate_factor
mu2_start_estimate *= overestimate_factor

loss = my_loss_fn( ... ) #Apply neural network and infer loss
loss = alrc(loss, mu1_start=mu1_start_estimate, mu2_start=mu2_start_estimate) #Apply ALRC
```

Note that `mu2_start` should be larger than `mu1_start`.
