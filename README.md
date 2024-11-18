Work in progress...

The goal is to construct a UNet-based surrogate model for the Cahn-Hilliard phase field dynamics,
and augment that surrogate with predictive uncertainties. Preliminary work has already established that
UNets can serve as useful surrogate models for this type of system. However, I am not aware of any related
UQ work for gauging the fidelity of these surrogates. To alleviate the computational cost of
UQ, I plan to implement the loss projected posterior sampler from,

(1) Miani, Marco, Hrittik Roy, and Søren Hauberg. "Bayes without Underfitting:
Fully Correlated Deep Learning Posteriors via Alternating Projections."
arXiv preprint arXiv:2410.16901 (2024).

in PyTorch, where jacobian-based operations (JVPs and VJPs) can be performed using `torch.func`.
In their paper, the authors' have already produced a codebase in JAX, which perhaps has several advantages.

At the moment, I am finding that my current implementation is still quite costly --
primarily due to iterating over the training dataset (in batches) during each cycle of
the alternating projections algorithm. Currently looking into torch scripting to see if
it can speed things up.

Also, I am out of colab gpu compute units.
