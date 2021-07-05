import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import wandb
from methods.mcdropout.models import MCDropout, LeNet5MCDropout
from methods.moe.gate_models import get_gating_network

LOSS_COEF=0

def cv_squared(x):
    """The squared coefficient of variation of a sample.
    Useful as a loss to encourage a positive distribution to be more uniform.
    Epsilons added for numerical stability.
    Returns 0 for an empty Tensor.
    Args:
    x: a `Tensor`.
    Returns:
    a `Scalar`.
    """
    eps = 1e-10
    # if only num_experts = 1
    if x.shape[0] == 1:
        return torch.Tensor([0])
    return x.float().var() / (x.float().mean()**2 + eps)


class DenseBasicMoE(nn.Module):
    """
    Class implementing a naiive approach for mixture of experts, with forward passes performed on all expert networks
    and gating applied afterwards. 
    """
    def __init__(self, network_class, gate_type='same', data_feat=28*28, n=5, k=5, dropout_p=0.1, **kwargs):
        """
        Initialise the model.

        Parameters
        ----------
        - network_class (type): type of network to use in the ensemble.
        - gate_type (str): which gating network should be used.
        - data_feat (int): number of sample input features
        - n (int): number of experts
        - k (int): number of highest-weighted expert predictions to use. Here for compatability, irrelevant for the dense 
          network as a weighted average of all experts is used.
        """
        super().__init__()
        self.experts = nn.ModuleList([network_class(**kwargs) for i in range(n)])
        self.n = n
        #TODO this is not necessarily the best appproach, but for now sort of works since it takes the same input
        # overall it might make sense to have this be a simple MLP
        # for now it can be conditionally set to be a fixed MLP
        self.gating_network = get_gating_network(network_class, gate_type, data_feat, n, dropout_p)

    def forward(self, x, labels=None, loss_coef=LOSS_COEF):
        """
        Compute combined and individual predictions for x.
        
        Parameters
        ---------
        - x (torch.Tensor): input data
        - labels (torch.Tensor): ground truth labels, for compatability with other approaches.

        Returns
        ---------
        - combined_pred (torch.Tensor): combined predixtion of the mixture of experts.
        - preds (list[torch.Tensor]): predictions of the individual experts.
        - part_sizes (np.ndarray): number of samples in batch x each individual expert 
          is assigned a non-zero weight for.
        - part_sizes_by_label (list[np.ndarray]): for compatability. 
          Number of samples in batch x each individual expert 
          is assigned a non-zero weight for, organised by ground truth label. Compatability is limited 
          to 10 labels, for a mixture of experts classifying samples into more categories, only the 
          first 10 will be considered.
        """
        preds = [net(x) for net in self.experts]
        weights = nn.functional.softmax(self.gating_network(x), dim=-1)

        importance = weights.sum(0)
        
        loss = cv_squared(importance)# + cv_squared(load)
        loss *= loss_coef



        combined_pred = torch.sum(
                            nn.functional.softmax(
                                torch.stack(
                                    preds, dim=0), 
                                dim=-1) * torch.unsqueeze(weights.T, -1), 
                            dim=0)
        
        weight_mask = weights > 0.1
        part_sizes = weight_mask.sum(0).cpu().numpy()
        disp = SparseDispatcher(self.n, weight_mask, labels)


        return combined_pred, preds, part_sizes, disp.part_sizes_by_label(), loss

    def forward_dense(self, x):
        """
        Compute combined and individual predictions for x.
        """
        out = self.forward(x)
        return out[0], out[1]


class DenseFixedMoE(nn.Module):
    """
    Dense naiive mixture of experts with a non-trainable gating network. 
    """
    def __init__(self, network_class, gate_type='same', data_feat=28*28, n=5, k=1, dropout_p=0.1, gate_by_class=True, **kwargs):
        """
        Initialise the model.

        Parameters
        ----------
        - network_class (type): type of network to use in the ensemble.
        - gate_type (str): which gating network should be used.
        - data_feat (int): number of sample input features
        - n (int): number of experts
        - k (int): number of highest-weighted expert predictions to use.
        - gate_by_class (bool): whether the fixed gating should choose the same experts for all elements of the same class in training.
        """
        super().__init__()
        self.experts = nn.ModuleList([network_class(**kwargs) for i in range(n)])
        self.k = k
        self.n = n
        self.gate_by_class = gate_by_class
        self.gating_network = get_gating_network(network_class, gate_type, data_feat, n, dropout_p)

        for param in self.gating_network.parameters():
            param.requires_grad = False

    def forward(self, x, labels=None):
        """
        Compute combined and individual predictions for x.
        
        Parameters
        ---------
        - x (torch.Tensor): input data
        - labels (torch.Tensor): ground truth labels, optional.

        Returns
        ---------
        - combined_pred (torch.Tensor): combined predixtion of the mixture of experts.
        - preds (list[torch.Tensor]): predictions of the individual experts.
        - part_sizes (np.ndarray): number of samples in batch x each individual expert 
          is assigned a non-zero weight for.
        - part_sizes_by_label (list[np.ndarray]): Number of samples in batch x each individual expert 
          is assigned a non-zero weight for, organised by ground truth label. Compatability is limited 
          to 10 labels, for a mixture of experts classifying samples into more categories, only the 
          first 10 will be considered.
        """
        preds = [net(x) for net in self.experts]
        
        if self.gate_by_class and labels is not None:
            weights = self.class_based_gating(labels)
        else:
            weights = nn.functional.softmax(self.gating_network(x), dim=-1)
            # top 1 selection, no conditional compute
            # max_weights = torch.max(weights, dim=-1, keepdims=True)[0]
            # weights = torch.where(weights == max_weights, 1., 0.)

            top_k_weights, top_k_indices = weights.topk(self.k, dim=-1)
            zeros = torch.zeros_like(weights, requires_grad=False)
            weights = zeros.scatter(1, top_k_indices, 1/self.k)


        combined_pred = torch.sum(
                            nn.functional.softmax(
                                torch.stack(
                                    preds, dim=0), 
                                dim=-1) * torch.unsqueeze(weights.T, -1), 
                            dim=0)

        part_sizes = (weights > 0).sum(0).cpu().numpy()

        disp = SparseDispatcher(self.n, weights, labels)
        
        return combined_pred, preds, part_sizes, disp.part_sizes_by_label(), 0
    
    def forward_dense(self, x):
        """
        Compute combined and individual predictions for x.
        """
        out = self.forward(x)
        return out[0], out[1]

    def class_based_gating(self, ys, prob_class_gated=0.95):
        """
        Compute gating weights for a fixed, ground-truth class-based assongment.

        Currently done by assigning the expert indices ass class_label modulo number_of_experts.
        """
        if self.k >= self.n:
            return torch.ones((ys.size[0], self.n)) / self.k
        else:
            idxs = torch.stack([(ys + i) % self.n for i in range(self.k)], dim=1).long()
            zeros = torch.zeros((ys.shape[0], self.n), requires_grad=False).to(ys.device)
            weights = zeros.scatter(1, idxs, 1/self.k)
            return weights
            


################### Sparse Implementation ###################################
# Based on David Rau's repository @ https://github.com/davidmrau/mixture-of-experts/ 


class SparseMoE(nn.Module):

    """
    Sparsely gated mixture of experts, based on the sparse MoE layer implementation
    in https://github.com/davidmrau/mixture-of-experts/ .
    For now with load balancing capabilities and probabilistic gating removed.
    """

    def __init__(self,  network_class, gate_type='same', data_feat=28*28, n=5, k=4, dropout_p=0.1, **kwargs):
        """
        Initialise the model.

        Parameters
        ----------
        - network_class (type): type of network to use in the ensemble.
        - gate_type (str): which gating network should be used.
        - data_feat (int): number of sample input features
        - n (int): number of experts
        - k (int): number of highest-weighted expert predictions to use.
        - gate_by_class (bool): whether the fixed gating should choose the same experts for all elements of the same class in training.
        """

        super(SparseMoE, self).__init__()
        # self.noisy_gating = noisy_gating
        self.num_experts = n
        # self.input_size = input_size
        self.k = k
        
        # instantiate experts
        self.experts = nn.ModuleList([network_class(**kwargs) for i in range(n)])
        self.gating_network = get_gating_network(network_class, gate_type, data_feat, n, dropout_p)

        # self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        # self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        # self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        assert(self.k <= self.num_experts)


    def custom_softmax(self, x):
        # means = torch.mean(x, 1, keepdim=True)
        x_exp = torch.exp(x)# - means)
        # norm = x.clone().detach().sum(1, keepdims=True)
        norm = x_exp.sum(1, keepdims=True)

        return x_exp / norm

    def top_k_gating(self, x):
        """
        Simple top-k gating using the gating network.

        Parameters
        ---------
        - x (torch.Tensor): input data

        Returns
        ---------
        - gates (torch.Tensor): a tensor of shape (num_samples, num_experts) containing the final 
          gating weights for the epert networks. Each row will have at most self.k non-zero entries,
          which must sum to 1.
        """
        # ------------------------ original version with scatter used ---------------------------------        
        # gating_out = self.gating_network(x)

        # top_k_logits, top_k_indices = gating_out.topk(self.k, dim=-1)
        # # top_k_logits = top_logits[:, :self.k]
        # # top_k_indices = top_indices[:, :self.k]

        # wandb.log({"gating_out": wandb.Histogram(gating_out.detach().cpu())})

        # # top_k_gates = self.softmax(top_k_logits)
        # # print((top_k_logits > 0).sum(), top_k_logits.max(), top_k_logits.min())
        # top_k_gates = self.custom_softmax(top_k_logits)
        # # print((top_k_gates > 0).sum(), (top_k_gates == 0).sum(), top_k_gates.min(), top_k_gates.max())
        # # top_k_gates = top_k_logits

        # zeros = torch.zeros_like(gating_out, requires_grad=True)
        
        # gates = zeros.scatter(1, top_k_indices, top_k_gates) #--- does not seem to work for backprop, at least for k=1
        
        # # if self.noisy_gating and self.k < self.num_experts:
        # #     load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        # # else:
        #     # load = self._gates_to_load(gates)

        # ------------------------ version without the use of the scatter function ---------------------------------
        # ------------------------ uncomment the full thing to use ---------------------------------
        
        gating_out = self.gating_network(x)

        wandb.log({"gating_out": wandb.Histogram(gating_out.detach().cpu())})

        gating_out = self.softmax(gating_out)

        top_k_logits, top_k_indices = gating_out.topk(self.k, dim=-1)
        zeros = torch.zeros_like(gating_out, requires_grad=True)
        factors = zeros.scatter(1, top_k_indices, 1)

        # this will be problematic since the non-zero entries won't add up to 1
        gates = gating_out * factors
        # TODO figure out a replacement/different gating training
        # so we are using sum-to-one normalisation, not a softmax here because I can't figure out how to make the
        # latter not affect the zeros and their placement
        # despite this, the softmax was kind of needed to not have one of the top values be zero! 
        # let's try to put it back in at the start
        gates = gates / gates.sum(dim=-1, keepdims=True)

        # -------------------- Returning load computation option
        load = self._gates_to_load(gates)

        return gates, load

    def forward_dense(self, x):
        """
        Run all the experts in the network, rather than doing it sparsely to save computation
        It is useful for inter-compatability with other methods that measure disagreement etc, also allows
        for non-gated predictions to be run easily

        Parameters
        ---------
        - x (torch.Tensor): input data

        Returns
        ---------
        - combined_pred (torch.Tensor): combined predixtion of the mixture of experts.
        - preds (list[torch.Tensor]): predictions of the individual experts. For compatability, as 
          the full network is not run here and they are not available.
        """

        gates, load = self.top_k_gating(x)
        preds = [net(x) for net in self.experts]

        combined_pred = torch.sum(
                            nn.functional.softmax(
                                torch.stack(
                                    preds, dim=0), 
                                dim=-1) * torch.unsqueeze(gates.T, -1), 
                            dim=0)
        
        return combined_pred, preds



    def forward(self, x, loss_coef=LOSS_COEF, labels=None):
        """
        Compute the combined prediction for x.
        
        Parameters
        ---------
        - x (torch.Tensor): input data
        - labels (torch.Tensor): ground truth labels, for compatability with other approaches.

        Returns
        ---------
        - combined_pred (torch.Tensor): combined predixtion of the mixture of experts.
        - preds (list[torch.Tensor]): predictions of the individual experts. For compatability, as 
          the full network is not run here and they are not available.
        - part_sizes (np.ndarray): number of samples in batch x each individual expert 
          is assigned a non-zero weight for.
        - part_sizes_by_label (list[np.ndarray]): for compatability. 
          Number of samples in batch x each individual expert 
          is assigned a non-zero weight for, organised by ground truth label. Compatability is limited 
          to 10 labels, for a mixture of experts classifying samples into more categories, only the 
          first 10 will be considered.
        """
        gates, load = self.top_k_gating(x)
        
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = cv_squared(importance) + cv_squared(load)
        loss *= loss_coef

        # print((gates > 0).sum())

        dispatcher = SparseDispatcher(self.num_experts, gates, labels)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, None, np.array(dispatcher._part_sizes), dispatcher.part_sizes_by_label(), loss


    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    # def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
    #     """Helper function to NoisyTopKGating.
    #     Computes the probability that value is in top k, given different random noise.
    #     This gives us a way of backpropagating from a loss that balances the number
    #     of times each expert is in the top k experts per example.
    #     In the case of no noise, pass in None for noise_stddev, and the result will
    #     not be differentiable.
    #     Args:
    #     clean_values: a `Tensor` of shape [batch, n].
    #     noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
    #       normally distributed noise with standard deviation noise_stddev.
    #     noise_stddev: a `Tensor` of shape [batch, n], or None
    #     noisy_top_values: a `Tensor` of shape [batch, m].
    #        "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
    #     Returns:
    #     a `Tensor` of shape [batch, n].
    #     """

    #     batch = clean_values.size(0)
    #     m = noisy_top_values.size(1)
    #     top_values_flat = noisy_top_values.flatten()
    #     threshold_positions_if_in = torch.arange(batch) * m + self.k
    #     threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
    #     is_in = torch.gt(noisy_values, threshold_if_in)
    #     threshold_positions_if_out = threshold_positions_if_in - 1
    #     threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out), 1)
    #     # is each value currently in the top k.
    #     prob_if_in = self.normal.cdf((clean_values - threshold_if_in)/noise_stddev)
    #     prob_if_out = self.normal.cdf((clean_values - threshold_if_out)/noise_stddev)
    #     prob = torch.where(is_in, prob_if_in, prob_if_out)
    #     return prob


    # def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
    #     """Noisy top-k gating.
    #       See paper: https://arxiv.org/abs/1701.06538.
    #       Args:
    #         x: input Tensor with shape [batch_size, input_size]
    #         train: a boolean - we only add noise at training time.
    #         noise_epsilon: a float
    #       Returns:
    #         gates: a Tensor with shape [batch_size, num_experts]
    #         load: a Tensor with shape [num_experts]
    #     """
    #     clean_logits = x @ self.w_gate
    #     if self.noisy_gating:
    #         raw_noise_stddev = x @ self.w_noise
    #         noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
    #         noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
    #         logits = noisy_logits
    #     else:
    #         logits = clean_logits

    #     # calculate topk + 1 that will be needed for the noisy gates
    #     top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
    #     top_k_logits = top_logits[:, :self.k]
    #     top_k_indices = top_indices[:, :self.k]
    #     top_k_gates = self.softmax(top_k_logits)

    #     zeros = torch.zeros_like(logits, requires_grad=True)
    #     gates = zeros.scatter(1, top_k_indices, top_k_gates)

    #     if self.noisy_gating and self.k < self.num_experts:
    #         load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
    #     else:
    #         load = self._gates_to_load(gates)
    #     return gates, load

class SparseDispatcher(object):
    """
    Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates, labels=None):
        """Create a SparseDispatcher."""
        # should work-as is

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = sorted_experts[index_sorted_experts[:, 1],0]
        # calculate num samples that each expert gets
        self._part_sizes = list((gates > 0).sum(0).cpu().numpy())
        # print(self._part_sizes)
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

        self._labels = labels

        # the gates array has to have 0s for the experts to not be run, and non-zero comdbination weights for the ones we want to run

    def part_sizes_by_label(self):
        sizes_by_label = []
        if self._labels is not None:
            for i in range(10):
                sizes_label_i = (self._gates[torch.where(self._labels == i)] > 0).sum(0).cpu().numpy()
                sizes_by_label.append(sizes_label_i)

        return sizes_by_label

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """
        # should work-as is

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index]#.squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)


    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # this one probably needs a bit more change since the log/non-log space thing is questionable
        # though I suppose it makes sense if we are working under the assumption that the final activation 
        # will be a softmax --- softmax if just normalizing for sum-to-one in exp space, what happens if 
        # we sum things in exp space before?


        # apply exp to expert outputs, so we are not longer in log space
        # stitched = torch.cat(expert_out, 0).exp()

        # softmax instead
        stitched = nn.functional.softmax(torch.cat(expert_out, 0), dim=-1)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True).to(stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        
        # ---- for now removed to use probability averaging ----
        # add eps to all zero values in order to avoid nans when going back to log space
        # combined[combined == 0] = np.finfo(float).eps
        # back to log space
        # return combined.log()

        return combined


    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)