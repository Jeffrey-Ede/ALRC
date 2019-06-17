# ALRC

Repository for the [paper](link goes here) "Adatpive Learning Rate Clipping Stabilizes Learning". 

ALRC is a simple, computationally inexpensive algorithm that stabilizes learning by limiting the magnitude of backpropagated losses. It can be applied to any neural network trained with stochastic gradient descent. In practice, it improves the training of neural networks where learning is destabilized by high errors and otherwise has little effect.

This repository contains source code for CIFAR-10 supersampling experiments with squared and countic errors. There is also an implementation of the ALRC algorithm in `alrc.py`. Some source code for partial-STEM experiments is [here](https://github.com/Jeffrey-Ede/partial-STEM).


If you are unsure whether to use ALRC, you should. It is computationally inexpensive and, at worst, will do nothing. Large improvements in the rate of convergence are seen when learning otherwise has high loss spikes.

ALRC can be applied like any other neural network layer

```python
loss = my_loss_fn( ... ) #Apply neural network and infer loss
loss = alrc(loss) #Apply ALRC to stabilize learning with default parameters
```

ALRC is clipping is robust to hyperparamer choices. The only hyperparameter that needs to be changed is an initial estimate for the threshold a number of standard deviations above the losses. If you're usure, run your neural network for 10+ iterations without ALRC to get an estimate. Anything a bit higher is fine, even if it is more than an order of magnitude too high, as the ALRC algorithm will decay it to the correct value.

```python
overestimate_factor = 3 #It's fine to overestimate the initial loss
loss_start_estimate = 1. #Rough initial loss estimate...
std_dev_estimate =  1 #Rough initial loss standard deviation estimate...
t = threshold_estimate = overestimate_factor*(loss_start_estimate + num_stddev*std_dev_estimate)

loss = my_loss_fn( ... ) #Apply neural network and infer loss
loss = alrc(loss, num_std_dev=num_stddev, mu1_start=t, mu2_start=t**2+1) #Apply ALRC
```

Note that `mu2_start` should be larger than `mu1_start`.
