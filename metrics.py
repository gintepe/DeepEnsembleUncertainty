"""
Calibration Metrics.
Adapted from the code provided alongside [1].

#### References:
[1]: Ovadia, Yaniv, et al. "Can you trust your model's uncertainty? 
     Evaluating predictive uncertainty under dataset shift." 
     arXiv preprint arXiv:1906.02530 (2019).
"""

import numpy as np
import torch
import scipy


def disagreement_and_correctness(predictions, gt):
  
  if predictions is None:
    return 0, 0, None

  counts = np.zeros((len(predictions), len(predictions)))
  correct = 0

  for i, pred in enumerate(predictions):
    _, predicted = torch.max(pred, 1)
    correct += (predicted == gt).sum().item()
    for j in range(i+1, len(predictions)):
      _, predicted_other = torch.max(predictions[j], 1)
      count = torch.sum(predicted != predicted_other).item()
      counts[i, j] += count
      counts[j, i] += count

  return np.sum(counts)/(len(predictions)*(len(predictions) - 1)), correct/(len(predictions)), counts

# Additional metrics needed have default implementations:
# * NLL for classification is equivalent to the cross entropy, commonly used as loss
# * Entropy has standard implementations in scipy

def basic_cross_entropy(probs, gt):
  """
  Implements cross entropy loss, for raw probability values
  """
  nll = torch.nn.NLLLoss()
  return nll(torch.log(probs), gt)

def wrap_ece(bins):
  """ convenience wrapper for the ECE computation when bins are fixed """
  return lambda prob, gt: expected_calibration_error_multiclass(
                                                prob.cpu().numpy(), 
                                                gt.cpu().numpy(), 
                                                bins)

def wrap_brier():
  """ convenience wrapper for the Brier score computation when bins are fixed """
  return lambda prob, gt: np.mean(brier_scores(gt.cpu().numpy(), prob.cpu().numpy()))

def bin_predictions_and_accuracies(probabilities, ground_truth, bins=10):
  """
  A helper function which histograms a vector of probabilities into bins.
  
  Parameters
  -----
    probabilities: A numpy vector of N probabilities assigned to each prediction
    ground_truth: A numpy vector of N ground truth labels in {0,1}
    bins: Number of equal width bins to bin predictions into in [0, 1], or an
      array representing bin edges.
  
  Returns
  -----
    bin_edges: Numpy vector of floats containing the edges of the bins
      (including leftmost and rightmost).
    accuracies: Numpy vector of floats for the average accuracy of the
      predictions in each bin.
    counts: Numpy vector of ints containing the number of examples per bin.
  """
  _validate_probabilities(probabilities)

  if len(probabilities) != len(ground_truth):
    raise ValueError(
        'Probabilies and ground truth must have the same number of elements.')

  if [v for v in ground_truth if v not in [0., 1., True, False]]:
    raise ValueError(
        'Ground truth must contain binary labels {0,1} or {False, True}.')

  if isinstance(bins, int):
    num_bins = bins
  else:
    num_bins = bins.size - 1

  # Ensure probabilities are never 0, since the bins in np.digitize are open on
  # one side.
  probabilities = np.where(probabilities == 0, 1e-8, probabilities)
  counts, bin_edges = np.histogram(probabilities, bins=bins, range=[0., 1.])
  indices = np.digitize(probabilities, bin_edges, right=True)
  accuracies = np.array([np.mean(ground_truth[indices == i])
                         for i in range(1, num_bins + 1)])
  return bin_edges, accuracies, counts


def bin_centers_of_mass(probabilities, bin_edges):
  probabilities = np.where(probabilities == 0, 1e-8, probabilities)
  indices = np.digitize(probabilities, bin_edges, right=True)
  return np.array([np.mean(probabilities[indices == i])
                   for i in range(1, len(bin_edges))])


def expected_calibration_error(probabilities, ground_truth, bins=15):
  """
  Compute the expected calibration error of a set of preditions in [0, 1].
  
  Parameters
  -----
    probabilities: A numpy vector of N probabilities assigned to each prediction
    ground_truth: A numpy vector of N ground truth labels in {0,1, True, False}
    bins: Number of equal width bins to bin predictions into in [0, 1], or
      an array representing bin edges.
  
  Returns
  -----
    Float: the expected calibration error.
  """

  probabilities = probabilities.flatten()
  ground_truth = ground_truth.flatten()
  bin_edges, accuracies, counts = bin_predictions_and_accuracies(
      probabilities, ground_truth, bins)
  bin_centers = bin_centers_of_mass(probabilities, bin_edges)
  num_examples = np.sum(counts)

  ece = np.sum([(counts[i] / float(num_examples)) * np.sum(
      np.abs(bin_centers[i] - accuracies[i]))
                for i in range(bin_centers.size) if counts[i] > 0])
  return ece


def accuracy_top_k(probabilities, labels, top_k):
  """
  Computes the top-k accuracy of predictions.
  A prediction is considered correct if the ground-truth class is among the k
  classes with the highest predicted probabilities.
  
  Parameters
  -----
    probabilities: Array of probabilities of shape [num_samples, num_classes].
    labels: Integer array labels of shape [num_samples].
    top_k: Integer. Number of highest-probability classes to consider.
  
  Returns
  -----
    float: Top-k accuracy of predictions.
  """
  _, ground_truth = _filter_top_k(probabilities, labels, top_k)
  return ground_truth.any(axis=-1).mean()


def _filter_top_k(probabilities, labels, top_k):
  """Extract top k predicted probabilities and corresponding ground truths."""

  labels_one_hot = np.zeros(probabilities.shape)
  labels_one_hot[np.arange(probabilities.shape[0]), labels] = 1

  if top_k is None:
    return probabilities, labels_one_hot

  # Negate probabilities for easier use with argpartition (which sorts from
  # lowest)
  negative_prob = -1. * probabilities

  ind = np.argpartition(negative_prob, top_k-1, axis=-1)
  top_k_ind = ind[:, :top_k]
  rows = np.expand_dims(np.arange(probabilities.shape[0]), axis=1)
  lowest_k_negative_probs = negative_prob[rows, top_k_ind]
  output_probs = -1. * lowest_k_negative_probs

  labels_one_hot_k = labels_one_hot[rows, top_k_ind]
  return output_probs, labels_one_hot_k


def get_multiclass_predictions_and_correctness(probabilities, labels, top_k=1):
  """Returns predicted class, correctness boolean vector."""
  _validate_probabilities(probabilities, multiclass=True)

  if top_k == 1:
    class_predictions = np.argmax(probabilities, -1)
    top_k_probs = probabilities[np.arange(len(labels)), class_predictions]
    is_correct = np.equal(class_predictions, labels)
  else:
    top_k_probs, is_correct = _filter_top_k(probabilities, labels, top_k)

  return top_k_probs, is_correct


def expected_calibration_error_multiclass(probabilities, labels, bins=15,
                                          top_k=1):
  """
  Computes expected calibration error from Guo et al. 2017.
  For details, see https://arxiv.org/abs/1706.04599.
  
  Parameters
  -----
    probabilities: Array of probabilities of shape [num_samples, num_classes].
    labels: Integer array labels of shape [num_samples].
    bins: Number of equal width bins to bin predictions into in [0, 1], or
      an array representing bin edges.
    top_k: Integer or None. If integer, use the top k predicted
      probabilities in ECE calculation (can be informative for problems with
      many classes and lower top-1 accuracy). If None, use all classes.
  
  Returns
  -----
    float: Expected calibration error.
  """
  top_k_probs, is_correct = get_multiclass_predictions_and_correctness(
      probabilities, labels, top_k)
  top_k_probs = top_k_probs.flatten()
  is_correct = is_correct.flatten()
  return expected_calibration_error(top_k_probs, is_correct, bins)


def compute_accuracies_at_confidences(labels, probs, thresholds):
  """
  Compute accuracy of samples above each confidence threshold.
  
  Parameters
  -----
    labels: Array of integer categorical labels.
    probs: Array of categorical probabilities.
    thresholds: Array of floating point probability thresholds in [0, 1).
  
  Returns
  -----
    accuracies: Array of accuracies over examples with confidence > T for each T
        in thresholds.
    counts: Count of examples with confidence > T for each T in thresholds.
  """
  assert probs.shape[:-1] == labels.shape

  predict_class = probs.argmax(-1)
  predict_confidence = probs.max(-1)

  shape = (len(thresholds),) + probs.shape[:-2]
  accuracies = np.zeros(shape)
  counts = np.zeros(shape)

  eq = np.equal(predict_class, labels)
  for i, thresh in enumerate(thresholds):
    mask = predict_confidence >= thresh
    counts[i] = mask.sum(-1)
    accuracies[i] = np.ma.masked_array(eq, mask=~mask).mean(-1)
  return accuracies, counts


def brier_scores(labels, probs=None, logits=None):
  """
  Compute elementwise Brier score.
  
  Parameters
  -----
    labels: Tensor of integer labels shape [N1, N2, ...]
    probs: Tensor of categorical probabilities of shape [N1, N2, ..., M].
    logits: If `probs` is None, class probabilities are computed as a softmax
      over these logits, otherwise, this argument is ignored.
  
  Returns
  -----
    Tensor of shape [N1, N2, ...] consisting of Brier score contribution from
    each element. The full-dataset Brier score is an average of these values.
  """
  assert (probs is None) != (logits is None)
  if probs is None:
    probs = scipy.special.softmax(logits, axis=-1)
  nlabels = probs.shape[-1]
  flat_probs = probs.reshape([-1, nlabels])
  flat_labels = labels.reshape([len(flat_probs)])

  plabel = flat_probs[np.arange(len(flat_labels)), flat_labels]

  # adding 1 to match the paper and conputations in general
  out = np.square(flat_probs).sum(axis=-1) - 2 * plabel + 1
  return out.reshape(labels.shape)


def get_quantile_bins(num_bins, probs, top_k=1):
  """
  Find quantile bin edges.
  
  Parameters
  -----
    num_bins: int, number of bins desired.
    probs: Categorical probabilities of shape [num_samples, num_classes].
    top_k: int, number of highest-predicted classes to consider in binning.
  
  Returns
  -----
    Numpy vector, quantile bin edges.
  """
  edge_percentiles = np.linspace(0, 100, num_bins+1)

  if len(probs.shape) == 1:
    probs = np.stack([probs, 1-probs]).T

  if top_k == 1:
    max_probs = probs.max(-1)
  else:
    unused_labels = np.zeros(probs.shape[0]).astype(np.int32)
    max_probs, _ = _filter_top_k(probs, unused_labels, top_k)

  bins = np.percentile(max_probs, edge_percentiles)
  bins[0], bins[-1] = 0., 1.
  return bins


def _validate_probabilities(probabilities, multiclass=False):
  if np.max(probabilities) > 1. or np.min(probabilities) < 0.:
    raise ValueError('All probabilities must be in [0,1].')
  if multiclass and not np.allclose(1, np.sum(probabilities, axis=-1),
                                    atol=1e-5):
    raise ValueError(
        'Multiclass probabilities must sum to 1 along the last dimension.')
