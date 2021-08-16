# MPhil Project 2021
## Deep Ensembles for Better Uncertainty Quantification

This repository contains a documented and extendable framework for experimenting with deep ensembles and their variations, as well as end-to-end deep neural network based mixtures of experts.

Parts of the code are based on the repositories [here](https://github.com/google-research/google-research/tree/master/uq_benchmark_2019) and [here](https://github.com/davidmrau/mixture-of-experts/), with references also included in the files affected.

### Methods Available

The repository offers implementations of ResNet-20 and LeNet5 network architectures as well as a simple feed-forward neural network, as well as version of the same architectures with dropout layers instrted after each non-final convolutional of fully-connected layer to choose from as base predictors.

They can be used in several different "methods". First, we provide implementations of well-established baseline methods. The first of these is training the base predictor as a stand-alone predictor by standard MAP training (`single` for the `--method` argument). The second option trains an n-member deep ensemble, using sequential forward and backward passes for the individual predictors (`ensemble`). The third method is MC Dropout, where networks are trained with dropout layers inserted after each non-final layer and a set dropout probability. Predictions are then made by keeping dropout enabled and averaging the results of n forward passes (`mcdrop`).

We also provide implementations of diversity-regularised deep ensemble variations. The first of these uses negative correlation learning with a set scaling parameter (`ncensemble`). The second method uses scaled mean pairwise cross entrypy between ensemble members as a regularisation term (`ceensemble`).

Lastly, implementations for end-to-end mixture of experts training are provided. The experts use the specified base predictor architecture, while the gating network can be chosen separately from a simple MLP, small convolutional network, and the same architecture as is used for the experts. Mixtures of experts can either be trained using a single optimiser for all components (`moe`) or using one optimiser for the experts and a separate one for the gating network (`moe2step`), with epochs involving two distinct passes over the training data - one to train the experts, followed by one to train only the gating network. 3 distinct loss functions can be used to train mixtures of experts: ensemble loss - cross-entropy using the combined prediction, sum loss - using a sum of individual losses weighted by the gating output, or log-sum-exp - using this function to approximate the minimum individual loss. The latter does not use gating weights in its computation and thus should only be used in conjunction with the 2-step training procedure.

### Usage
As the codebase was produced first and foremost for an MPhil project, it is focused on enabling full experiments to be run in an easy and customisable way. The entry point for neural network training is `main.py`

To easily use the DNN training implementation, a user would have to update the `constants.py` file to contain paths to the desirable checkpoint, data and logging directories.

[Weights and Biases](https://wandb.ai/) is used as a logger throughout, with project and run names, as well as user possible to specify within the main configuration.

The script is controlled via the following arguments:
* General and logging:
  * `-h`, `--help`: shows a help message.
  * `--log`: when present, log training statistics to wandb.
  * `--project PROJECT`: project name for logging.
  * `--user USER`: user name for logging.
  * `--run-name RUN_NAME`   wandb run name.
* Data:
  * `--data-dir DATA_DIR`: directory the relevant datasets can be found in. Default `.\data`.
  * `--dataset-type {cifar10,cifar100,mnist}`: dataset to use. Default `mnist`.
  * `--corrupted-test`: when present, additionally test on a corrupted (shifted) testing set. If omitted, will only use the standar one.
  * `--validation-fraction VALIDATION_FRACTION`: Fraction of the training set to be held out for validation, in cases where a dedicated set is unavailable. Default `0.1`.
  * `--log-subdir LOG_SUBDIR`: logging subdirectory, to allow organisation by experiment.
  * `--detailed-eval`: whether to log detailed testing results in an accessible numpy format. Will be saved in the same directory as checkpoints. 
* General method and training config:
  * `--method {single,ensemble,mcdrop,ncensemble,ceensemble,moe,moe2step}`: method to run. Default `single`.
  * `--n N`: size of the ensemble to be trained. In case of MC Dropout - number of fowward passes to average. Default `5`.
  * `--model {lenet,mlp,resnet}`: model architecture to be used. Default `lenet`.
  * `--reg-weight REG_WEIGHT`: scaling factor for custom loss regularisation, initial value. In MoE controls the load balancing scaling. Default `0.5`.
  * `--reg-min REG_MIN`: lower bound on the regularisation scaling constant. Default `0`
  * `--reg-decay REG_DECAY`: exponential decay factor for regularisation weight. Default `1` (no decay).
  * `--dropout DROPOUT`: dropout rate for models that use it. Default `0.5`.
  * `--scheduler {step,exp,multistep,multistep-ext,multistep-adam` if set, will use a learning rate scheduler, of the type specified.
  * `--scheduler-step SCHEDULER_STEP`: for `step` scheduler, the number of epochs between learning rate reductions. Default `20`.
  * `--scheduler-rate SCHEDULER_RATE`: for all scheduler, the factor to multiply the learning rate by upon reduction. Default `0.1`.
  * `--optimizer {adam,sgd}`: which optimizer to use. SGD will default to momentum of 0.9. Default `adam`.
  * `--batch-size BATCH_SIZE`: batch size to use in training. Default `128`.
  * `--epochs EPOCHS`: maximum number of epochs to train for. Default `100`. 
  * `--lr LR`: initial learning rate. Default `1e-3`
  * `--weight-decay WEIGHT_DECAY`: l2 regularisation (weight decay) scaling. Default `0` (no weight decay).
  * `--cpu`: if ommited present, train on a CPU
  * `--checkpoint`: if present, save final chekpoint.
  * `--num-workers NUM_WORKERS`: number of CPU cores to load data on. Default `0`.
  * `--early-stop-tol EARLY_STOP_TOL`: when specified, will be used as the number of epochs without improvement allowed before early stopping.
  * `--seed SEED`: random seed. If provided, experiments run with fixed rng.
* Mixture of experts specific config:
  * `--predict-gated`: if present, a MoE model should use gating in predictions. Otherwise, a mean (ensemble) prediciton is used.
  * `--moe-type {dense,fixed,fixed-class,sparse}`: type of a MoE model. Dense uses a gating network to determine weights for averaging, fixed is a dummy with fixed allocatons. Default `dense`.
  * `--moe-gating {same,simple,mcd_simple,mcdc_simple,mcd_lenet,mcd_conv,conv,mcd_resnet}`: type of a gating network to use in a MoE model. Same sets the network to have the same architecture as experts. Simple is an MLP of with a single 100-unit hidden layer. Default `same`.
  * `--moe-loss {ens,sum,lsexp}`: the type of training loss to use for MoE models (expert step if 2-step training used). Default `ens`.
  * `--moe-topk MOE_TOPK`: for sparse MoE gating, the number of experts to use. Default `1`.
* Bayesian gating approximation config:
  * `--gating-laplace`: if present, the run should use post-hoc laplace prroximation for the gating network.
  * `--laplace-precision LAPLACE_PRECISION`: prior precision for the Laplace approximation. If not specified, it will be fitted.
  * `--entropy-threshold ENTROPY_THRESHOLD`: if present, will use uniform gating on samples with gating output entropy above the threshold.

To download the relevant datasets and their corrupted versions on the Ubuntu OS (likely other Linux systems as well, however it has not beet tested) the `download_data.sh` script can be used, supplying the path to the directory data should be stored in as the first argument. Note this script requires torchvision to be installed to obtain MNIST, numpy and pickle modules for processing the corrupted CIFAR datasets.

Similarly, additional post-training evaluation can be run via `evaluate.py`. See the help message for more detailed usage information.


### Extendability

The repository can be extended by addind additional methods, metrics, and network architectures. 

New methods should extend the `BaseTrainer` class, which provides basic training and testing loop functionality. Applying the framework in the context of regression is possible, but would require re-implementing the train, vlaidate and test functionality to remove the tracking of accuracy.

New model architectures can be added to the file `methods\models.py`.

The relevant selector functions (in `util.py`) mapping from `main.py` arguments to appropriate trainer instantiations should also be updated. 