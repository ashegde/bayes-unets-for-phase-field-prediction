The Cahn-Hilliard equation describes the separation of a binary mixture into pure components. This is a canonical phase field model, and has been studied extensively in the literature. In the above code, we construct a UNet surrogate model to emulate the Cahn-Hilliard dynamics. A common numerical challenge in simulating phase fields is that they require integration with tiny stepsizes. To overcome this, several recent works have employed neural network-based models to coarsen the timescale -- i.e., to take larger time steps:

[1] Oommen, Vivek, et al. "Learning two-phase microstructure evolution using neural operators and autoencoder architectures." npj Computational Materials 8.1 (2022): 190.

[2] Oommen, Vivek, et al. "Rethinking materials simulations: Blending direct numerical simulations with neural operators." npj Computational Materials 10.1 (2024): 145.

[3] Bonneville, Christophe, et al. "Accelerating Phase Field Simulations Through a Hybrid Adaptive Fourier Neural Operator with U-Net Backbone." arXiv preprint arXiv:2406.17119 (2024).

This repo illustrates the idea of 'time coarsening' on a fairly simple system. There are several caveats. Due to computational limitations (google colab), I am using just low resolution simulations and fairly limited data. Moreover, the underlying simulation is quite fast -- so in this particular setting, building a surrogate may not actually buy us much in terms of the computational cost vs. accuracy tradeoff. Moreover, the overall intent here is to (eventually) explore different strategies for uncertainty quantification on a model problem.

To get things up and running on colab, first clone this repo and then run:

```
!python prepare_dataset.py --n_train 100 --init_noise_scale 0.1
!python train_model.py --log_freq 100 --n_epochs 20
!python test_prediction.py
```

The first command builds a training, validation, and test dataset using the simulator. The simulator time step is set to `dt=0.01`s and the training horizon corresponds to the first `500` time steps. Additionally, note that the initial field for each simulation is generated according to a random rule. One can observe that this rule can generate quite diverse behaviors. The second command trains a UNet model with certain hardwired settings -- e.g., predictions with the UNet leap `25` time steps forward (relative to the simulator), resulting in a coarsened time step of `.25`s. The final command generates animations comparing the UNet surrogate to the simulator on test data (unseen during training). Note that we only use the simulator for the first `50` time steps since the dynamics are quite fast. Therefore, the UNet is first initialized by the simulator at time `t=0.50`s, and then subsequently operates independently. Note that the test simulations run for `10.0`s, which is double the time horizon used for training. Hence, everything after `5.0`s is extrapolation.

An example result is shown below.


https://github.com/user-attachments/assets/cc8f79d9-6543-4d02-814f-11df1d76d898


It is interesting to observe, in this and other examples, that the surrogate retains plausible physical behavior (at least with respect to the eyeball norm), even when deviating from the underlying physics. Naturally, this is also alarming and it demonstrates that failures can occur with no clear signal. 

Work in progress: incorporating uncertainty quantification using the strategies described in the reference below:

(1) Miani, Marco, Hrittik Roy, and Søren Hauberg. "Bayes without Underfitting:
Fully Correlated Deep Learning Posteriors via Alternating Projections."
arXiv preprint arXiv:2410.16901 (2024).

While this is aspect currently unfinished, the code can be found in the pipelines/sampling`folder. The main challenge is the computational cost, which is exacerbated by the fact that I am currently using the free edition of google colab. The cost arises due to iterating over the training dataset (in batches) during each cycle of
the alternating projections algorithm. Time permitting, it would also be interesting to incorporate other strategies for UQ, such as "evidential deep learning". While these methods are not strictly ``Bayesian'' and do not necessarily account for model/parameter uncertainty, they may still be practically useful -- e.g., in understanding when the model is less confident and may deviate from the underlying physics.
