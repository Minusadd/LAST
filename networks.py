

import os

import torch
import numpy as np

import torchvision.transforms as T

from PIL import Image


from transformers import T5Tokenizer, T5EncoderModel

import pickle

#from headers import *
from gumbel import gumbel_softmax
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


use_cuda = torch.cuda.is_available()

from torch.autograd import Function

def create_padding_mask(padded_sequences, padding_value=-1):
    return (padded_sequences == padding_value)

def create_causal_mask(batch_size, seq_length):
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
    return mask


class StraightThroughEstimator(Function):
    @staticmethod
    def forward(ctx, input, discrete_input):
        ctx.save_for_backward(discrete_input)
        return discrete_input

    @staticmethod
    def backward(ctx, grad_output):
        discrete_input, = ctx.saved_tensors
        return grad_output, None


ste = StraightThroughEstimator.apply

class CombineOutputs(Function):

    @staticmethod
    def forward(ctx, a, b):
        modified_b = b
        idx = (modified_b[:, 0] == 1).nonzero(as_tuple=True)[0]
        range_tensor = torch.arange(len(a), device=a.device)
        closest_indices = torch.searchsorted(idx, range_tensor, right=True) - 1
        c = a[idx[closest_indices]]

        ctx.save_for_backward(a, modified_b, idx)
        return c

    @staticmethod
    def backward(ctx, grad_c):
        a, b, idx = ctx.saved_tensors

        # Calculate grad_a
        range_tensor = torch.arange(len(a), device=a.device)
        closest_indices = torch.searchsorted(idx, range_tensor, right=True) - 1
        result = idx[closest_indices]
        grad_a = torch.zeros_like(a)
        grad_a.index_add_(0, result, grad_c)

        new_row = torch.zeros((1, a.shape[1])).to(a.device)
        temp_a = torch.cat((new_row, a[idx[closest_indices]][:-1]), dim=0).to(a.device)


        grad_b_0 = ((a - temp_a) * grad_c).sum(dim=1)
        grad_b_1 = -grad_b_0  # -(a * grad_c).sum(dim=1) #(a[idx[closest_indices]] * grad_c).sum(dim=1)
        # Calculate grad_b
        grad_b = torch.stack((grad_b_0, grad_b_1), dim=1)
        return grad_a, grad_b


combine_outputs = CombineOutputs.apply


def combine_tensors(tensor1, tensor2):
    # Check if both tensors have the same shape
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape"

    # Define the target vectors for comparison
    target_zero_one = torch.tensor([0, 1]).to(tensor1.device)
    target_one_zero = torch.tensor([1, 0]).to(tensor1.device)

    # Find the positions where both tensors have [0, 1] or [1, 0]
    both_zero_one = torch.all(tensor1 == tensor2, dim=-1) & torch.all(tensor1 == target_zero_one, dim=-1)
    both_one_zero = torch.all(tensor1 == tensor2, dim=-1) & torch.all(tensor1 == target_one_zero, dim=-1)

    # Create the combined tensor using element-wise operations
    combined = both_one_zero.unsqueeze(-1).float().detach() * tensor1 + ((both_zero_one | ~both_one_zero).unsqueeze(
        -1).float() * target_zero_one).detach()

    return combined

class Obsencoder(torch.nn.Module):
    def __init__(self, output_size=256):
        super(Obsencoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.fc = torch.nn.Linear(128 * 2 * 2, output_size)
        self.activation_layer = torch.nn.Tanh()

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Convolutional layer #relu?
        x = F.relu(self.conv2(x))  # Convolutional layer
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.activation_layer(self.fc(x))  # Fully connected layer
        return x

class Lanencoder_skill(torch.nn.Module):
    def __init__(self,output_size):
        super(Lanencoder_skill, self).__init__()
        self.fc = torch.nn.Linear(768, output_size)
        self.activation_layer = torch.nn.Tanh()

    def forward(self, x):
        x = self.activation_layer(self.fc(x))  # Fully connected layer
        return x

class Lanencoder_goal(torch.nn.Module):
    def __init__(self,output_size):
        super(Lanencoder_goal, self).__init__()
        self.fc = torch.nn.Linear(768, output_size)
        self.activation_layer = torch.nn.Tanh()

    def forward(self, x):
        x = self.activation_layer(self.fc(x))  # Fully connected layer
        return x

class Actionencoder(torch.nn.Module):
    def __init__(self,output_size=64):
        super(Actionencoder, self).__init__()
        self.fc = torch.nn.Linear(101, output_size)
        self.activation_layer = torch.nn.Tanh()

    def forward(self, x):
        x = self.activation_layer(self.fc(x))  # Fully connected layer
        return x

class VariationalNetwork(torch.nn.Module):

    # def __init__(self, input_size, hidden_size, z_dimensions, number_layers=4, z_exploration_bias=0., b_exploration_bias=0.,  batch_size=1):
    def __init__(self, args, obsenc, lanenc, actenc, img_emb_size=256, action_size = 89 + 12, act_emb_size=64, hidden_size=128, num_subpolicies=50, num_objects=89, g_dimensions=256, number_layers=3, d_model=512):
        # Ensures inheriting from torch.nn.Module goes nicely and cleanly.
        # super().__init__()
        super(VariationalNetwork, self).__init__()
        self.args = args
        self.obs_enc = obsenc
        self.lang_enc = lanenc
        self.act_enc = actenc
        self.input_size = img_emb_size + act_emb_size + g_dimensions + 2
        self.hidden_size = hidden_size
        self.output_size = num_subpolicies + num_objects
        self.num_layers = number_layers
        self.num_subpolicies = num_subpolicies
        self.num_objects = num_objects
        self.batch_size = 128
        self.k_exploration_bias = 0.
        self.m_exploration_bias = 0.
        self.k_probability_factor = 0.01
        self.m_probability_factor = 0.01
        self.d_model = d_model
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.input_projection = torch.nn.Linear(self.input_size, self.d_model)
        self.output_projection = torch.nn.Linear(self.d_model, 2 * self.hidden_size)


        # Transform to output space - Latent z and Latent b.
        # THIS OUTPUT LAYER TAKES 2*HIDDEN SIZE as input because it's bidirectional.
        self.termination_output_layer = torch.nn.Linear(2 * self.hidden_size, 2)

        # Softmax activation functions for Bernoulli termination probability and latent z selection .
        self.batch_softmax_layer = torch.nn.Softmax(dim=-1)
        self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=-1)

        self.subpolicy_output_layer = torch.nn.Linear(self.hidden_size + g_dimensions, self.num_subpolicies) #+ g_dimensions
        self.k_first_layer = torch.nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.activation_layer = torch.nn.Tanh()
        self.hidden_activation = F.relu
        self.variance_activation_layer = torch.nn.Softplus()
        self.variance_activation_bias = 0.

        self.variance_factor = 0.01


    def forward(self, state_input, action_input, lang_input, presampled_m, epsilon, new_z_selection=True, var_epsilon=0., deterministic=False,
                sampled_k=None, sampled_z=None, temperature=1.0, constrain_m=True):

        # padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        # padding_mask = create_padding_mask(padded_sequences)
        b, seq, _, _, _ = state_input.size()
        state_input = state_input.view(-1, state_input.shape[2], state_input.shape[3], state_input.shape[4])
        state_transform = self.obs_enc(state_input)  # Fully connected layer
        state_transform = state_transform.view(b, seq, -1)
        # # TODO RELU
        lang_input = lang_input.view(-1, lang_input.shape[2])
        lang_transform =  self.lang_enc(lang_input)

        lang_transform = lang_transform.view(b, seq, -1)
        action_input = action_input.view(-1, action_input.shape[2]).to(torch.float32)
        act_transform = self.act_enc(action_input)
        act_transform = act_transform.view(b, seq, -1)
        mask = ~((presampled_m != 2).any(dim=-1))
        presampled_m = presampled_m.to(torch.float32)
        input = torch.cat([state_transform, act_transform, lang_transform, presampled_m], dim=-1)

        input_transform = self.input_projection(input)
        outputs_transform = self.transformer_encoder(input_transform, src_key_padding_mask=mask)
        outputs = self.output_projection(outputs_transform)

        # outputs, hidden = self.gru(input)
        # outputs = outputs * (~mask).unsqueeze(2)

        variational_k_pre = self.hidden_activation(self.k_first_layer(outputs))
        # FIXME Damping factor for probabilities to prevent washing out of bias.
        variational_m_preprobabilities = self.termination_output_layer(outputs) * self.m_probability_factor

        variational_m_preprobabilities[:, 0, 0] += self.m_exploration_bias

        sampled_ms, variational_m_probabilities = gumbel_softmax(variational_m_preprobabilities,
                                                                temperature=temperature, hard=True,
                                                                deterministic=deterministic, return_prob=True)
        #sampled_m = sampled_m.squeeze(0)


        k_final_input = variational_k_pre
        #TODO
        variational_k_preprobabilities = self.subpolicy_output_layer(
            torch.cat([k_final_input, lang_transform], dim=-1))
        #variational_k_preprobabilities = self.subpolicy_output_layer(k_final_input)
        presampled_k_indexes, variational_k_probabilities = gumbel_softmax(variational_k_preprobabilities,
                                                                         temperature=temperature,
                                                                         hard=True,
                                                                         deterministic=deterministic, return_prob=True)

        #presampled_k = presampled_k_index.squeeze(0)#.squeeze(1)
        sampled_ms[:, 0, 0] = 1
        sampled_ms[:, 0, 1] = 0

        if not constrain_m:
            sampled_m_indexes = sampled_ms # combine_tensors(latent_ms, presampled_m)
        else:
            sampled_m_indexes = combine_tensors(sampled_ms, presampled_m)
            variational_m_probabilities = presampled_m[:, :, :1] * variational_m_probabilities
            vec = (variational_m_probabilities[:] == torch.tensor([0., 0.]).to(sampled_ms.device)).all(dim=-1)
            variational_m_probabilities[vec] = torch.tensor([0., 1.]).to(sampled_ms.device)

        sampled_k_indexes = []


        for bb in range(sampled_ms.shape[0]):
            sampled_k_index = combine_outputs(presampled_k_indexes[bb], sampled_m_indexes[bb])
            sampled_k_indexes.append(sampled_k_index)

        sampled_k_indexes = torch.stack(sampled_k_indexes).view(b, seq, -1)
        sampled_k_index_last = torch.cat(
            ((torch.zeros((b, 1, sampled_k_indexes.shape[2]))).to(input.device), sampled_k_indexes[:, :-1, :]),
            dim=1).detach()
        #print(variational_m_probabilities)
        k_prob = sampled_k_index_last * variational_m_probabilities[:, :,
                                        1:] + variational_k_probabilities * variational_m_probabilities[:, :, :1]
        variational_k_logprobabilities = torch.log(k_prob + 1e-30)
        sampled_k_entropys = -(k_prob * variational_k_logprobabilities).sum(dim=-1, keepdim=True)

        sampled_k_codelen = - (sampled_k_indexes * variational_k_logprobabilities).sum(dim=-1, keepdim=True)

        return sampled_k_indexes, sampled_m_indexes.view(b, seq, -1), k_prob, variational_m_probabilities, presampled_k_indexes.view(b, seq, -1), sampled_k_codelen, sampled_k_entropys





class DiscretePolicyNetwork(torch.nn.Module):

    def __init__(self, args, obsenc, lanenc, actenc, img_emb_size=256, action_size=89 + 12, act_emb_size=64,
                 hidden_size=128, num_subpolicies=50, num_objects=89, g_dimensions=256, number_layers=3, d_model=512):


        super(DiscretePolicyNetwork, self).__init__()

        self.args = args
        self.obs_enc = obsenc
        self.lang_enc = lanenc
        self.act_enc = actenc
        self.hidden_size = hidden_size
        self.output_size = num_subpolicies + num_objects
        self.num_layers = number_layers
        self.num_subpolicies = num_subpolicies
        self.num_objects = num_objects
        self.batch_size = 16
        self.k_exploration_bias = 0.
        self.m_exploration_bias = 0.
        self.k_probability_factor = 0.01
        self.m_probability_factor = 0.01
        self.d_model = d_model
        self.input_size = img_emb_size + act_emb_size + g_dimensions + 2 + num_subpolicies

        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.input_projection = torch.nn.Linear(self.input_size, self.d_model)
        self.output_projection = torch.nn.Linear(self.d_model, self.hidden_size)

        self.termination_output_layer = torch.nn.Linear(self.hidden_size, 2)

        self.batch_softmax_layer = torch.nn.Softmax(dim=-1)
        self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=-1)

        self.subpolicy_output_layer = torch.nn.Linear(self.hidden_size, self.num_subpolicies)
        self.k_first_layer = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.activation_layer = torch.nn.Tanh()
        self.hidden_activation = F.relu
        self.variance_activation_layer = torch.nn.Softplus()
        self.variance_activation_bias = 0.

        self.variance_factor = 0.01


    def forward(self, state_input, action_input, goal_input, skill_input, m_input, presampled_m, epsilon, new_z_selection=True, var_epsilon=0., deterministic=False,
                sampled_k=None, sampled_z=None, temperature=1.0):

        b, seq, _, _, _ = state_input.size()
        state_input = state_input.view(-1, state_input.shape[2], state_input.shape[3], state_input.shape[4])
        state_transform = self.obs_enc(state_input).detach()  # Fully connected layer
        state_transform = state_transform.view(b, seq, -1)
        action_input = action_input.view(-1, action_input.shape[2]).to(torch.float32)
        act_transform = self.act_enc(action_input).detach()
        act_transform = act_transform.view(b, seq, -1)
        goal_input = goal_input.view(-1, seq, goal_input.shape[2])
        lang_transform = self.lang_enc(goal_input).detach()
        lang_transform = lang_transform.view(b, seq, -1)
        mask = ~((goal_input != -1).any(dim=-1))
        presampled_m = presampled_m.to(torch.float32)
        input = torch.cat([state_transform, act_transform, lang_transform, m_input, skill_input], dim=-1)

        input_transform = self.input_projection(input)
        outputs_transform = self.transformer_encoder(input_transform, mask=torch.ones(seq, seq).triu(diagonal=1).bool().to(input.device), src_key_padding_mask=mask)
        outputs = self.output_projection(outputs_transform)



        latent_k_preprobabilities = self.subpolicy_output_layer(self.hidden_activation(self.k_first_layer(outputs)))
        latent_m_preprobabilities = self.termination_output_layer(outputs) + self.m_exploration_bias
        # if mask_input is not None:

        latent_ms, latent_m_probabilities = gumbel_softmax(latent_m_preprobabilities, temperature=temperature,
                                                          hard=True, deterministic=deterministic, return_prob=True)

        latent_ks, latent_k_probabilities = gumbel_softmax(latent_k_preprobabilities, temperature=temperature,
                                                          hard=True,
                                                          deterministic=deterministic, return_prob=True)


        latent_ms[:, 0, 0] = 1
        latent_ms[:, 0, 1] = 0
        sampled_m_indexes = latent_ms#combine_tensors(latent_ms, presampled_m)
        sampled_k_indexes = []
        #TODO
        log_ks = []

        for bb in range(latent_ms.shape[0]):
            sampled_k_index = combine_outputs(latent_ks[bb], sampled_m_indexes[bb])
            log_k = (sampled_k_index * latent_k_probabilities).sum(dim=-1, keepdim=True)
            sampled_k_indexes.append(sampled_k_index)
            log_ks.append(log_k)
        sampled_k_indexes = torch.stack(sampled_k_indexes).view(b, seq, -1)
        sampled_k_index_last = torch.cat(
            ((torch.zeros((b, 1, sampled_k_indexes.shape[2]))).to(input.device), sampled_k_indexes[:, :-1, :]),
            dim=1).detach()

        k_prob = sampled_k_index_last * latent_m_probabilities[:, :,
                                        1:] + latent_k_probabilities * latent_m_probabilities[:, :, :1]
        return sampled_k_indexes, sampled_m_indexes.view(b, seq,
                                                         -1), k_prob, latent_m_probabilities  # cross_entropy


    def get_actions(self, state_input, action_input, goal_input, skill_input, m_input, presampled_m, greedy=False, temperature=1.0, mask_input=None, fix_m=False):
        # Input Format must be: Sequence_Length x 1 x Input_Size.
        b, seq, _, _, _ = state_input.size()
        state_input = state_input.view(-1, state_input.shape[2], state_input.shape[3], state_input.shape[4])
        state_transform = self.obs_enc(state_input).detach()  # Fully connected layer
        state_transform = state_transform.view(b, seq, -1)
        action_input = action_input.view(-1, action_input.shape[2]).to(torch.float32)
        act_transform = self.act_enc(action_input).detach()
        act_transform = act_transform.view(b, seq, -1)
        goal_input = goal_input.view(-1, seq, goal_input.shape[2])
        lang_transform = self.lang_enc(goal_input).detach()
        lang_transform = lang_transform.view(b, seq, -1)
        # input = torch.cat([state_transform, skill_input, lang_transform], dim=-1)
        mask = ~((goal_input != -1).any(dim=-1))
        if presampled_m is not None:
            presampled_m = presampled_m.to(torch.float32)
        input = torch.cat([state_transform, act_transform, lang_transform, m_input, skill_input], dim=-1)

        input_transform = self.input_projection(input)
        outputs_transform = self.transformer_encoder(input_transform, mask=torch.ones(seq, seq).triu(diagonal=1).bool().to(input.device), src_key_padding_mask=mask)

        outputs = self.output_projection(outputs_transform)

        latent_k_preprobabilities = self.subpolicy_output_layer(self.hidden_activation(self.k_first_layer(outputs)))
        latent_m_preprobabilities = self.termination_output_layer(outputs) + self.m_exploration_bias
        if mask_input is not None:
            latent_k_preprobabilities = latent_k_preprobabilities.masked_fill(mask_input == 0,
                                                                                          float('-1e6'))
        latent_ms, latent_m_probabilities = gumbel_softmax(latent_m_preprobabilities, temperature=temperature,
                                                           hard=True, deterministic=greedy, return_prob=True)

        latent_ks, latent_k_probabilities = gumbel_softmax(latent_k_preprobabilities, temperature=temperature,
                                                           hard=True,
                                                           deterministic=greedy, return_prob=True)
        # TODO

        latent_ms[:, 0, 0] = 1
        latent_ms[:, 0, 1] = 0
        sampled_m_indexes = latent_ms
        sampled_k_indexes = []

        #FIXME
        for bb in range(latent_ms.shape[0]):
            sampled_k_index = combine_outputs(latent_ks[bb], sampled_m_indexes[bb])
            sampled_k_indexes.append(sampled_k_index)
        return torch.stack(sampled_k_indexes).view(b, seq, -1).detach(), sampled_m_indexes.view(b, seq, -1).detach()

    def get_embedding(self, state_input, action_input, goal_input, skill_input, m_input, epsilon, rewards=None, presampled_m=None, new_z_selection=True, var_epsilon=0., deterministic=False,
                sampled_k=None, sampled_z=None, temperature=1.0, mask_input=None, online_training=False, fix_m=False):

        b, seq, _, _, _ = state_input.size()
        state_input = state_input.view(-1, state_input.shape[2], state_input.shape[3], state_input.shape[4])
        state_transform = self.obs_enc(state_input).detach()  # Fully connected layer
        state_transform = state_transform.view(b, seq, -1)
        action_input = action_input.view(-1, action_input.shape[2]).to(torch.float32)
        act_transform = self.act_enc(action_input).detach()
        act_transform = act_transform.view(b, seq, -1)
        goal_input = goal_input.view(-1, seq, goal_input.shape[2])
        lang_transform = self.lang_enc(goal_input).detach()
        lang_transform = lang_transform.view(b, seq, -1)
        mask = ~((rewards!= -1).any(dim=-1))
        input = torch.cat([state_transform, act_transform, lang_transform, m_input, skill_input], dim=-1)

        input_transform = self.input_projection(input)

        outputs_transform = self.transformer_encoder(input_transform,
                                                     mask=torch.ones(seq, seq).triu(diagonal=1).bool().to(input.device),
                                                     src_key_padding_mask=mask)

        outputs = self.output_projection(outputs_transform)
        if online_training:
            outputs = outputs.detach()
        latent_k_preprobabilities = self.subpolicy_output_layer(self.hidden_activation(self.k_first_layer(outputs)))#detach
        latent_m_preprobabilities = self.termination_output_layer(outputs) + self.m_exploration_bias#detach
        # if mask_input is not None:
        if mask_input is not None:
            latent_k_preprobabilities = latent_k_preprobabilities.masked_fill(mask_input == 0,
                                                                                          float('-1e6'))
        latent_ms, latent_m_probabilities = gumbel_softmax(latent_m_preprobabilities, temperature=temperature,
                                                           hard=True, deterministic=deterministic, return_prob=True)

        latent_ks, latent_k_probabilities = gumbel_softmax(latent_k_preprobabilities, temperature=temperature,
                                                           hard=True,
                                                           deterministic=deterministic, return_prob=True)


        latent_ms[:, 0, 0] = 1
        latent_ms[:, 0, 1] = 0
        if presampled_m is not None:
            if fix_m:
                sampled_m_indexes = presampled_m
            else:
                sampled_m_indexes = combine_tensors(latent_ms, presampled_m)
        else:
            sampled_m_indexes = latent_ms  # combine_tensors(latent_ms, presampled_m)
        sampled_k_indexes = []


        log_ms = (sampled_m_indexes * latent_m_probabilities).sum(dim=-1, keepdim=True)
        log_ks = []
        if presampled_m is not None:
            latent_m_probabilities = presampled_m[:, :, :1] * latent_m_probabilities
        for bb in range(latent_ms.shape[0]):
            sampled_k_index = combine_outputs(latent_ks[bb], sampled_m_indexes[bb])
            log_k = (sampled_k_index * latent_k_probabilities).sum(dim=-1, keepdim=True)
            sampled_k_indexes.append(sampled_k_index)
            log_ks.append(log_k)
        sampled_k_indexes = torch.stack(sampled_k_indexes).view(b, seq, -1)
        sampled_k_index_last = torch.cat(
            ((torch.zeros((b, 1, sampled_k_indexes.shape[2]))).to(input.device), sampled_k_indexes[:, :-1, :]),
            dim=1).detach()

        k_prob = sampled_k_index_last * latent_m_probabilities[:,:,1:] + latent_k_probabilities * latent_m_probabilities[:, :, :1]

        return sampled_k_indexes, sampled_m_indexes.view(b, seq,
                                                 -1), k_prob, latent_m_probabilities, outputs  # cross_entropy




class LowLevelPolicyNetwork(torch.nn.Module):

    def __init__(self, args, obsenc, lanenc, img_emb_size=256, action_size = 12, hidden_size=256, num_subpolicies=50, num_objects=89, g_dimensions=256,
                 number_layers=3, d_model=512, n_head=8,
                 repram=True, train=True, b_exploration_bias=0., batch_size=1, whether_latentb_input=False,
                 zero_z_dim=False, joint=False, small_init=False, causal_mask=False, include_goal=False):

        super(LowLevelPolicyNetwork, self).__init__()
        self.obs_enc = obsenc
        self.lang_enc = lanenc
        self.hidden_size = hidden_size

        self.g_dimensions = g_dimensions
        self.num_subpolicies = num_subpolicies
        self.output_size = action_size + num_subpolicies + g_dimensions
        self.num_objects = num_objects
        self.causal_mask = causal_mask
        self.num_layers = number_layers
        self.args = args
        self.batch_size = self.args.batch_size

        self.d_model = d_model
        self.include_goal = include_goal
        if self.include_goal:
            self.input_size = img_emb_size + num_subpolicies + g_dimensions
        else:
            self.input_size = img_emb_size + num_subpolicies
        model_args = dict(n_layer=self.num_layers, n_head=n_head, n_embd=d_model, block_size=512,
                          bias=False, vocab_size=None, dropout=0.0)

        # Transformer
        self.lang_projection = torch.nn.Linear(self.g_dimensions, self.num_subpolicies)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.input_projection = torch.nn.Linear(self.input_size, self.d_model)
        self.output_projection = torch.nn.Linear(self.d_model, self.hidden_size)

        if small_init:
            for name, param in self.mean_output_layer.named_parameters():
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    torch.nn.init.xavier_normal_(param, gain=0.0001)
        self.subpolicy_output_layer1 = torch.nn.Linear(self.hidden_size, action_size)
        self.subpolicy_output_layer2 = torch.nn.Linear(self.hidden_size, self.num_objects)

        self.activation_layer = torch.nn.Tanh()
        self.variance_activation_layer = torch.nn.Softplus()
        self.variance_activation_bias = 0.
        self.variance_factor = 0.01

    def forward(self, state_input, goal_input, skill_input=None, mask_input=None, lang_input=None, temperature=1.0, epsilon=0.001, deterministic=False):

        b, seq, _, _, _ = state_input.size()
        state_input = state_input.view(-1, state_input.shape[2], state_input.shape[3], state_input.shape[4])
        state_transform = self.obs_enc(state_input)  # Fully connected layer
        state_transform = state_transform.view(b, seq, -1)
        goal_input = goal_input.view(-1, seq, goal_input.shape[2])
        lang_transform = self.lang_enc(goal_input)
        lang_transform = lang_transform.view(b, seq, -1)
        if skill_input is None:
            lang_transform_skill = self.lang_enc(lang_input)
            lang_emb = self.lang_projection(lang_transform_skill)
            skill_input, _ = gumbel_softmax(lang_emb,temperature=temperature,
                                                                               hard=True,
                                                                               deterministic=deterministic,
                                                                               return_prob=True)
            skill_input = skill_input.view(b, seq, -1)

        if self.include_goal:
            input = torch.cat([state_transform, skill_input, lang_transform], dim=-1)
        else:
            input = torch.cat([state_transform, skill_input], dim=-1) #, lang_transform
        mask1 = ~((goal_input != -1).any(dim=-1))

        input_transform = self.input_projection(input)
        if not self.causal_mask:
            mask1 = torch.cat([mask1, torch.zeros(b, 1).to(input_transform.device)], dim=1)
            sentinel_token = torch.zeros(1, self.d_model).unsqueeze(0)
            input_transform = torch.cat((input_transform, sentinel_token.repeat(b, 1, 1).to(input_transform.device)), dim=1)
            mask2 = (torch.ones(seq + 1, seq + 1).tril(diagonal=-17).bool() | torch.ones(seq + 1, seq + 1).triu(
                diagonal=1).bool()).to(input_transform.device)

            outputs_transform = self.transformer_encoder(input_transform, mask=mask2, src_key_padding_mask=mask1)[:, :-1, :] #is_causal=False,
        else:
            outputs_transform = self.transformer_encoder(input_transform, mask=torch.ones(seq, seq).triu(diagonal=1).bool().to(input_transform.device), src_key_padding_mask=mask1)  # is_causal=False,
        lstm_outputs = self.output_projection(outputs_transform)

        #lstm_outputs, hidden = self.gru(input)

        #lstm_outputs = lstm_outputs * (~mask).unsqueeze(2)
        variational_k_preprobabilities1 = self.subpolicy_output_layer1(
            lstm_outputs)
        variational_k_preprobabilities2 = self.subpolicy_output_layer2(
            lstm_outputs)
        if mask_input is not None:
            variational_k_preprobabilities2 = variational_k_preprobabilities2.masked_fill(mask_input == 0, float('-1e6'))
            #variational_k_preprobabilities2 = variational_k_preprobabilities2 * mask_input

        presampled_k_index, variational_k_probabilities1 = gumbel_softmax(variational_k_preprobabilities1,
                                                                          temperature=temperature,
                                                                          hard=True,
                                                                          deterministic=True, return_prob=True)
        presampled_obj_index, variational_k_probabilities2 = gumbel_softmax(variational_k_preprobabilities2,
                                                                            temperature=temperature,
                                                                            hard=True,
                                                                            deterministic=True,
                                                                            return_prob=True)

        sampled_action = torch.cat([variational_k_preprobabilities1, variational_k_preprobabilities2], dim=-1)
        # Predict Gaussian means and variances.

        return sampled_action

    def get_actions(self, state_input, goal_input, skill_input=None, mask_input=None, lang_input=None, temperature=1.0, greedy=False):

        b, seq, _, _, _ = state_input.size()
        state_input = state_input.view(-1, state_input.shape[2], state_input.shape[3], state_input.shape[4])
        state_transform = self.obs_enc(state_input)  # Fully connected layer
        state_transform = state_transform.view(b, seq, -1)
        # # TODO RELU
        goal_input = goal_input.view(-1, goal_input.shape[2])
        lang_transform = self.lang_enc(goal_input)
        lang_transform = lang_transform.view(b, seq, -1)

        if skill_input is None:
            lang_transform_skill = self.lang_enc(lang_input)
            lang_emb = self.lang_projection(lang_transform_skill)
            skill_input, _ = gumbel_softmax(lang_emb,temperature=temperature,
                                                                               hard=True,
                                                                               deterministic=greedy,
                                                                               return_prob=True)
            skill_input = skill_input.view(b, seq, -1)
        if self.include_goal:
            input = torch.cat([state_transform, skill_input, lang_transform], dim=-1)
        else:
            input = torch.cat([state_transform, skill_input], dim=-1) #, lang_transform
        input_transform = self.input_projection(input)
        if not self.causal_mask:
            mask2 = (torch.ones(seq, seq).tril(diagonal=-6).bool() | torch.ones(seq, seq).triu(
                diagonal=1).bool()).to(input.device)
            outputs_transform = self.transformer_encoder(input_transform, mask=mask2)
        else:
            outputs_transform = self.transformer_encoder(input_transform, mask=torch.ones(seq, seq).triu(diagonal=1).bool().to(input.device))
        lstm_outputs = self.output_projection(outputs_transform)
        #lstm_outputs, hidden = self.gru(input)

        variational_k_preprobabilities1 = self.subpolicy_output_layer1(
            lstm_outputs)
        variational_k_preprobabilities2 = self.subpolicy_output_layer2(
            lstm_outputs)
        if mask_input is not None:
            variational_k_preprobabilities2 = variational_k_preprobabilities2.masked_fill(mask_input == 0, float('-1e6'))
        presampled_k_index, variational_k_probabilities1 = gumbel_softmax(variational_k_preprobabilities1,
                                                                          temperature=temperature,
                                                                          hard=True,
                                                                          deterministic=greedy, return_prob=True)
        presampled_obj_index, variational_k_probabilities2 = gumbel_softmax(variational_k_preprobabilities2,
                                                                            temperature=temperature,
                                                                            hard=True,
                                                                            deterministic=greedy,
                                                                            return_prob=True)

        sampled_action = torch.cat([presampled_k_index, presampled_obj_index], dim=-1)
        # Predict Gaussian means and variances.

        return sampled_action.squeeze(0).detach(), skill_input.squeeze(0).detach()

    def get_embedding(self, state_input, goal_input, skill_input=None, mask_input=None, lang_input=None, temperature=1.0, rewards=None, epsilon=0.001, deterministic=False, online_training=False):

        b, seq, _, _, _ = state_input.size()
        state_input = state_input.view(-1, state_input.shape[2], state_input.shape[3], state_input.shape[4])
        state_transform = self.obs_enc(state_input)  # Fully connected layer
        state_transform = state_transform.view(b, seq, -1)
        goal_input = goal_input.view(-1, seq, goal_input.shape[2])
        lang_transform = self.lang_enc(goal_input)
        lang_transform = lang_transform.view(b, seq, -1)

        if skill_input is None:
            lang_transform_skill = self.lang_enc(lang_input)
            lang_emb = self.lang_projection(lang_transform_skill)
            skill_input, _ = gumbel_softmax(lang_emb,temperature=temperature,
                                                                               hard=True,
                                                                               deterministic=deterministic,
                                                                               return_prob=True)
            skill_input = skill_input.view(b, seq, -1)

        if self.include_goal:
            input = torch.cat([state_transform, skill_input, lang_transform], dim=-1)
        else:
            input = torch.cat([state_transform, skill_input], dim=-1) #, lang_transform
        mask1 = ~((rewards!= -1).any(dim=-1))

        input_transform = self.input_projection(input)
        if not self.causal_mask:
            mask1 = torch.cat([mask1, torch.zeros(b, 1).to(input_transform.device)], dim=1)
            sentinel_token = torch.zeros(1, self.d_model).unsqueeze(0)
            input_transform = torch.cat((input_transform, sentinel_token.repeat(b, 1, 1).to(input_transform.device)), dim=1)
            mask2 = (torch.ones(seq + 1, seq + 1).tril(diagonal=-17).bool() | torch.ones(seq + 1, seq + 1).triu(
                diagonal=1).bool()).to(input_transform.device)

            outputs_transform = self.transformer_encoder(input_transform, mask=mask2, src_key_padding_mask=mask1)[:, :-1, :] #is_causal=False,
        else:
            outputs_transform = self.transformer_encoder(input_transform, mask=torch.ones(seq, seq).triu(diagonal=1).bool().to(input_transform.device), src_key_padding_mask=mask1)  # is_causal=False,
        lstm_outputs = self.output_projection(outputs_transform)
        if online_training:
            lstm_outputs = lstm_outputs.detach()

        variational_k_preprobabilities1 = self.subpolicy_output_layer1(
            lstm_outputs)
        variational_k_preprobabilities2 = self.subpolicy_output_layer2(
            lstm_outputs.detach())
        if mask_input is not None:
            variational_k_preprobabilities2 = variational_k_preprobabilities2.masked_fill(mask_input == 0, float('-1e6'))


        presampled_k_index, variational_k_probabilities1 = gumbel_softmax(variational_k_preprobabilities1,
                                                                          temperature=temperature,
                                                                          hard=True,
                                                                          deterministic=True, return_prob=True)
        presampled_obj_index, variational_k_probabilities2 = gumbel_softmax(variational_k_preprobabilities2,
                                                                            temperature=temperature,
                                                                            hard=True,
                                                                            deterministic=True,
                                                                            return_prob=True)


        return presampled_k_index, presampled_obj_index, variational_k_probabilities1, variational_k_probabilities2, lstm_outputs


class LowLevelMLPPolicyNetwork(torch.nn.Module):

    def __init__(self, args, obsenc, lanenc, img_emb_size=256, action_size = 12, hidden_size=256, num_subpolicies=50, num_objects=89, g_dimensions=256,
                 number_layers=3, d_model=512, n_head=8,
                 repram=True, train=True, b_exploration_bias=0., batch_size=1, whether_latentb_input=False,
                 zero_z_dim=False, joint=False, small_init=False):

        super(LowLevelMLPPolicyNetwork, self).__init__()
        self.obs_enc = obsenc
        self.lang_enc = lanenc
        self.hidden_size = hidden_size

        self.g_dimensions = g_dimensions
        self.num_subpolicies = num_subpolicies
        self.output_size = action_size + num_subpolicies
        self.num_objects = num_objects

        self.num_layers = number_layers
        self.args = args
        self.batch_size = self.args.batch_size

        self.d_model = d_model
        self.input_size = img_emb_size + num_subpolicies  + g_dimensions
        model_args = dict(n_layer=self.num_layers, n_head=n_head, n_embd=d_model, block_size=512,
                          bias=False, vocab_size=None, dropout=0.0)

        self.lang_projection = torch.nn.Linear(self.g_dimensions, self.num_subpolicies)
        self.input_projection = torch.nn.Linear(self.input_size, self.d_model)
        self.output_projection = torch.nn.Linear(self.d_model, self.hidden_size)
        self.fc1 = torch.nn.Linear(self.d_model, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.d_model)

        if small_init:
            for name, param in self.mean_output_layer.named_parameters():
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    torch.nn.init.xavier_normal_(param, gain=0.0001)
        self.subpolicy_output_layer1 = torch.nn.Linear(self.hidden_size, action_size)
        self.subpolicy_output_layer2 = torch.nn.Linear(self.hidden_size, self.num_objects)
        # self.first_layer = torch.nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.hidden_activation = F.relu
        self.activation_layer = torch.nn.Tanh()
        self.variance_activation_layer = torch.nn.Softplus()
        self.variance_activation_bias = 0.
        self.variance_factor = 0.01

    def forward(self, state_input, goal_input, skill_input=None, mask_input=None, lang_input=None, temperature=1.0, epsilon=0.001, deterministic=False):

        b, seq, _, _, _ = state_input.size()
        state_input = state_input.view(-1, state_input.shape[2], state_input.shape[3], state_input.shape[4])
        state_transform = self.obs_enc(state_input)  # Fully connected layer
        state_transform = state_transform.view(b, seq, -1)
        goal_input = goal_input.view(-1, seq, goal_input.shape[2])
        lang_transform = self.lang_enc(goal_input)
        if skill_input is None:
            lang_transform_skill = self.lang_enc(lang_input)
            lang_emb = self.lang_projection(lang_transform_skill)
            skill_input, _ = gumbel_softmax(lang_emb,temperature=temperature,
                                                                               hard=True,
                                                                               deterministic=deterministic,
                                                                               return_prob=True)
            skill_input = skill_input.view(b, seq, -1)
        lang_transform = lang_transform.view(b, seq, -1)

        #input = torch.cat([state_transform, skill_input], dim=-1)
        input = torch.cat([state_transform, skill_input, lang_transform], dim=-1)
        mask = ~((goal_input != -1).any(dim=-1))
        input_transform = self.input_projection(input)
        x1 = self.hidden_activation(self.fc1(input_transform))
        outputs = self.hidden_activation(self.fc2(x1))
        lstm_outputs = self.output_projection(outputs)

        #lstm_outputs, hidden = self.gru(input)

        #lstm_outputs = lstm_outputs * (~mask).unsqueeze(2)
        variational_k_preprobabilities1 = self.subpolicy_output_layer1(
            lstm_outputs)
        variational_k_preprobabilities2 = self.subpolicy_output_layer2(
            lstm_outputs)
        if mask_input is not None:
            variational_k_preprobabilities2 = variational_k_preprobabilities2.masked_fill(mask_input == 0, float('-1e6'))
            #variational_k_preprobabilities2 = variational_k_preprobabilities2 * mask_input


        sampled_action = torch.cat([variational_k_preprobabilities1, variational_k_preprobabilities2], dim=-1)
        # Predict Gaussian means and variances.

        return sampled_action

    def get_actions(self, state_input, goal_input, skill_input=None, mask_input=None, lang_input=None, temperature=1.0, greedy=False):

        b, seq, _, _, _ = state_input.size()
        state_input = state_input.view(-1, state_input.shape[2], state_input.shape[3], state_input.shape[4])
        state_transform = self.obs_enc(state_input)  # Fully connected layer
        state_transform = state_transform.view(b, seq, -1)
        # # TODO RELU
        goal_input = goal_input.view(-1, goal_input.shape[2])
        if skill_input is None:
            lang_transform_skill = self.lang_enc(lang_input)
            lang_emb = self.lang_projection(lang_transform_skill)
            skill_input, _ = gumbel_softmax(lang_emb, temperature=temperature,
                                            hard=True,
                                            deterministic=greedy,
                                            return_prob=True)
            skill_input = skill_input.view(b, seq, -1)
        lang_transform = self.lang_enc(goal_input)
        lang_transform = lang_transform.view(b, seq, -1)
        #input = torch.cat([state_transform, skill_input], dim=-1)
        input = torch.cat([state_transform, skill_input, lang_transform], dim=-1)
        input_transform = self.input_projection(input)
        x1 = self.hidden_activation(self.fc1(input_transform))
        outputs = self.hidden_activation(self.fc2(x1))
        lstm_outputs = self.output_projection(outputs)
        #lstm_outputs, hidden = self.gru(input)

        variational_k_preprobabilities1 = self.subpolicy_output_layer1(
            lstm_outputs)
        variational_k_preprobabilities2 = self.subpolicy_output_layer2(
            lstm_outputs)
        if mask_input is not None:
            variational_k_preprobabilities2 = variational_k_preprobabilities2.masked_fill(mask_input == 0, float('-1e6'))
        presampled_k_index, variational_k_probabilities1 = gumbel_softmax(variational_k_preprobabilities1,
                                                                          temperature=temperature,
                                                                          hard=True,
                                                                          deterministic=greedy, return_prob=True)
        presampled_obj_index, variational_k_probabilities2 = gumbel_softmax(variational_k_preprobabilities2,
                                                                            temperature=temperature,
                                                                            hard=True,
                                                                            deterministic=greedy,
                                                                            return_prob=True)

        sampled_action = torch.cat([presampled_k_index, presampled_obj_index], dim=-1)
        # Predict Gaussian means and variances.

        return sampled_action.squeeze(0), skill_input.squeeze(0)

class LowLevelPolicyNetworknoskill(torch.nn.Module):

    def __init__(self, args, obsenc, lanenc, img_emb_size=256, action_size = 12, hidden_size=256, num_subpolicies=50, num_objects=89, g_dimensions=256,
                 number_layers=3, d_model=512, n_head=8,
                 repram=True, train=True, b_exploration_bias=0., batch_size=1, whether_latentb_input=False,
                 zero_z_dim=False, joint=False, small_init=False, causal_mask=False, include_goal=False):

        super(LowLevelPolicyNetworknoskill, self).__init__()
        self.obs_enc = obsenc
        self.lang_enc = lanenc
        self.hidden_size = hidden_size

        self.g_dimensions = g_dimensions
        self.num_subpolicies = num_subpolicies
        self.output_size = action_size + num_subpolicies + g_dimensions
        self.num_objects = num_objects
        self.causal_mask = causal_mask
        self.num_layers = number_layers
        self.args = args
        self.batch_size = self.args.batch_size

        self.d_model = d_model
        self.include_goal = include_goal

        self.input_size = img_emb_size + g_dimensions


        model_args = dict(n_layer=self.num_layers, n_head=n_head, n_embd=d_model, block_size=512,
                          bias=False, vocab_size=None, dropout=0.0)
        # Create LSTM Network.
        #self.gru = torch.nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        # Transformer
        self.lang_projection = torch.nn.Linear(self.g_dimensions, self.num_subpolicies)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.input_projection = torch.nn.Linear(self.input_size, self.d_model)
        self.output_projection = torch.nn.Linear(self.d_model, self.hidden_size)

        if small_init:
            for name, param in self.mean_output_layer.named_parameters():
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    torch.nn.init.xavier_normal_(param, gain=0.0001)
        self.subpolicy_output_layer1 = torch.nn.Linear(self.hidden_size, action_size)
        self.subpolicy_output_layer2 = torch.nn.Linear(self.hidden_size, self.num_objects)
        # self.first_layer = torch.nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.activation_layer = torch.nn.Tanh()
        self.variance_activation_layer = torch.nn.Softplus()
        self.variance_activation_bias = 0.
        self.variance_factor = 0.01

    def forward(self, state_input, goal_input, mask_input=None, lang_input=None, temperature=1.0, epsilon=0.001, deterministic=False):

        b, seq, _, _, _ = state_input.size()
        state_input = state_input.view(-1, state_input.shape[2], state_input.shape[3], state_input.shape[4])
        state_transform = self.obs_enc(state_input)  # Fully connected layer
        state_transform = state_transform.view(b, seq, -1)
        goal_input = goal_input.view(-1, seq, goal_input.shape[2])
        lang_transform = self.lang_enc(goal_input)
        lang_transform = lang_transform.view(b, seq, -1)


        input = torch.cat([state_transform, lang_transform], dim=-1)

        mask1 = ~((goal_input != -1).any(dim=-1))

        input_transform = self.input_projection(input)
        if not self.causal_mask:
            mask1 = torch.cat([mask1, torch.zeros(b, 1).to(input_transform.device)], dim=1)
            sentinel_token = torch.zeros(1, self.d_model).unsqueeze(0)
            input_transform = torch.cat((input_transform, sentinel_token.repeat(b, 1, 1).to(input_transform.device)), dim=1)
            mask2 = (torch.ones(seq + 1, seq + 1).tril(diagonal=-17).bool() | torch.ones(seq + 1, seq + 1).triu(
                diagonal=1).bool()).to(input_transform.device)

            outputs_transform = self.transformer_encoder(input_transform, mask=mask2, src_key_padding_mask=mask1)[:, :-1, :] #is_causal=False,
        else:
            outputs_transform = self.transformer_encoder(input_transform, mask=torch.ones(seq, seq).triu(diagonal=1).bool().to(input_transform.device), src_key_padding_mask=mask1)  # is_causal=False,
        lstm_outputs = self.output_projection(outputs_transform)


        variational_k_preprobabilities1 = self.subpolicy_output_layer1(
            lstm_outputs)
        variational_k_preprobabilities2 = self.subpolicy_output_layer2(
            lstm_outputs)
        if mask_input is not None:
            variational_k_preprobabilities2 = variational_k_preprobabilities2.masked_fill(mask_input == 0, float('-1e6'))
            #variational_k_preprobabilities2 = variational_k_preprobabilities2 * mask_input

        presampled_k_index, variational_k_probabilities1 = gumbel_softmax(variational_k_preprobabilities1,
                                                                          temperature=temperature,
                                                                          hard=True,
                                                                          deterministic=True, return_prob=True)
        presampled_obj_index, variational_k_probabilities2 = gumbel_softmax(variational_k_preprobabilities2,
                                                                            temperature=temperature,
                                                                            hard=True,
                                                                            deterministic=True,
                                                                            return_prob=True)

        sampled_action = torch.cat([variational_k_preprobabilities1, variational_k_preprobabilities2], dim=-1)
        # Predict Gaussian means and variances.

        return sampled_action

    def get_actions(self, state_input, goal_input, mask_input=None, lang_input=None, temperature=1.0, greedy=False):

        b, seq, _, _, _ = state_input.size()
        state_input = state_input.view(-1, state_input.shape[2], state_input.shape[3], state_input.shape[4])
        state_transform = self.obs_enc(state_input)  # Fully connected layer
        state_transform = state_transform.view(b, seq, -1)
        # # TODO RELU
        goal_input = goal_input.view(-1, goal_input.shape[2])
        lang_transform = self.lang_enc(goal_input)
        lang_transform = lang_transform.view(b, seq, -1)


        input = torch.cat([state_transform, lang_transform], dim=-1)

        input_transform = self.input_projection(input)
        if not self.causal_mask:
            mask2 = (torch.ones(seq, seq).tril(diagonal=-6).bool() | torch.ones(seq, seq).triu(
                diagonal=1).bool()).to(input.device)
            outputs_transform = self.transformer_encoder(input_transform, mask=mask2)
        else:
            outputs_transform = self.transformer_encoder(input_transform, mask=torch.ones(seq, seq).triu(diagonal=1).bool().to(input.device))
        lstm_outputs = self.output_projection(outputs_transform)


        variational_k_preprobabilities1 = self.subpolicy_output_layer1(
            lstm_outputs)
        variational_k_preprobabilities2 = self.subpolicy_output_layer2(
            lstm_outputs)
        if mask_input is not None:
            variational_k_preprobabilities2 = variational_k_preprobabilities2.masked_fill(mask_input == 0, float('-1e6'))
        presampled_k_index, variational_k_probabilities1 = gumbel_softmax(variational_k_preprobabilities1,
                                                                          temperature=temperature,
                                                                          hard=True,
                                                                          deterministic=greedy, return_prob=True)
        presampled_obj_index, variational_k_probabilities2 = gumbel_softmax(variational_k_preprobabilities2,
                                                                            temperature=temperature,
                                                                            hard=True,
                                                                            deterministic=greedy,
                                                                            return_prob=True)

        sampled_action = torch.cat([presampled_k_index, presampled_obj_index], dim=-1)
        # Predict Gaussian means and variances.

        return sampled_action.squeeze(0).detach()

    def get_embedding(self, state_input, goal_input, mask_input=None, lang_input=None, temperature=1.0, rewards=None, epsilon=0.001, deterministic=False, online_training=False):

        b, seq, _, _, _ = state_input.size()
        state_input = state_input.view(-1, state_input.shape[2], state_input.shape[3], state_input.shape[4])
        state_transform = self.obs_enc(state_input)  # Fully connected layer
        state_transform = state_transform.view(b, seq, -1)
        goal_input = goal_input.view(-1, seq, goal_input.shape[2])

        lang_transform = self.lang_enc(goal_input).detach()
        lang_transform = lang_transform.view(b, seq, -1)


        input = torch.cat([state_transform, lang_transform], dim=-1)

        mask1 = ~((rewards!= -1).any(dim=-1))

        input_transform = self.input_projection(input)
        if not self.causal_mask:
            mask1 = torch.cat([mask1, torch.zeros(b, 1).to(input_transform.device)], dim=1)
            sentinel_token = torch.zeros(1, self.d_model).unsqueeze(0)
            input_transform = torch.cat((input_transform, sentinel_token.repeat(b, 1, 1).to(input_transform.device)), dim=1)
            mask2 = (torch.ones(seq + 1, seq + 1).tril(diagonal=-17).bool() | torch.ones(seq + 1, seq + 1).triu(
                diagonal=1).bool()).to(input_transform.device)

            outputs_transform = self.transformer_encoder(input_transform, mask=mask2, src_key_padding_mask=mask1)[:, :-1, :] #is_causal=False,
        else:
            outputs_transform = self.transformer_encoder(input_transform, mask=torch.ones(seq, seq).triu(diagonal=1).bool().to(input_transform.device), src_key_padding_mask=mask1)  # is_causal=False,
        lstm_outputs = self.output_projection(outputs_transform)
        if online_training:
            lstm_outputs = lstm_outputs.detach()

        variational_k_preprobabilities1 = self.subpolicy_output_layer1(
            lstm_outputs)
        variational_k_preprobabilities2 = self.subpolicy_output_layer2(
            lstm_outputs.detach())
        if mask_input is not None:
            variational_k_preprobabilities2 = variational_k_preprobabilities2.masked_fill(mask_input == 0, float('-1e6'))
            #variational_k_preprobabilities2 = variational_k_preprobabilities2 * mask_input

        presampled_k_index, variational_k_probabilities1 = gumbel_softmax(variational_k_preprobabilities1,
                                                                          temperature=temperature,
                                                                          hard=True,
                                                                          deterministic=True, return_prob=True)
        presampled_obj_index, variational_k_probabilities2 = gumbel_softmax(variational_k_preprobabilities2,
                                                                            temperature=temperature,
                                                                            hard=True,
                                                                            deterministic=True,
                                                                            return_prob=True)



        return presampled_k_index, presampled_obj_index, variational_k_probabilities1, variational_k_probabilities2, lstm_outputs

class LisaDiscretePolicyNetwork(torch.nn.Module):


    def __init__(self, args, obsenc, lanenc, actenc, img_emb_size=256, action_size=89 + 12, act_emb_size=64,
                 hidden_size=128, num_subpolicies=50, num_objects=89, g_dimensions=256, number_layers=3, d_model=512):

        # Ensures inheriting from torch.nn.Module goes nicely and cleanly.
        # super().__init__()
        super(LisaDiscretePolicyNetwork, self).__init__()

        self.args = args
        self.obs_enc = obsenc
        self.lang_enc = lanenc
        self.act_enc = actenc
        self.hidden_size = hidden_size
        self.output_size = num_subpolicies + num_objects
        self.num_layers = number_layers
        self.num_subpolicies = num_subpolicies
        self.num_objects = num_objects
        self.batch_size = 16
        self.k_exploration_bias = 0.
        self.m_exploration_bias = 0.
        self.k_probability_factor = 0.01
        self.m_probability_factor = 0.01
        self.d_model = d_model
        self.input_size = img_emb_size + act_emb_size + g_dimensions

        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.input_projection = torch.nn.Linear(self.input_size, self.d_model)
        self.output_projection = torch.nn.Linear(self.d_model, self.hidden_size)

        # Transform to output space - Latent z and Latent b.
        # THIS OUTPUT LAYER TAKES 2*HIDDEN SIZE as input because it's bidirectional.
        self.termination_output_layer = torch.nn.Linear(self.hidden_size, 2)

        # Softmax activation functions for Bernoulli termination probability and latent z selection .
        self.batch_softmax_layer = torch.nn.Softmax(dim=-1)
        self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=-1)

        self.subpolicy_output_layer = torch.nn.Linear(self.hidden_size, self.num_subpolicies)
        self.k_first_layer = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.activation_layer = torch.nn.Tanh()
        self.hidden_activation = F.relu
        self.variance_activation_layer = torch.nn.Softplus()
        self.variance_activation_bias = 0.

        self.variance_factor = 0.01


    def forward(self, state_input, action_input, goal_input, skill_input, m_input, presampled_m, epsilon, new_z_selection=True, var_epsilon=0., deterministic=False,
                sampled_k=None, sampled_z=None, temperature=1.0):

        b, seq, _, _, _ = state_input.size()
        state_input = state_input.view(-1, state_input.shape[2], state_input.shape[3], state_input.shape[4])
        state_transform = self.obs_enc(state_input).detach()  # Fully connected layer
        state_transform = state_transform.view(b, seq, -1)
        action_input = action_input.view(-1, action_input.shape[2]).to(torch.float32)
        act_transform = self.act_enc(action_input).detach()
        act_transform = act_transform.view(b, seq, -1)
        goal_input = goal_input.view(-1, seq, goal_input.shape[2])
        lang_transform = self.lang_enc(goal_input).detach()
        lang_transform = lang_transform.view(b, seq, -1)
        mask = ~((goal_input != -1).any(dim=-1))
        presampled_m = presampled_m.to(torch.float32)
        input = torch.cat([state_transform, act_transform, lang_transform], dim=-1)

        input_transform = self.input_projection(input)
        outputs_transform = self.transformer_encoder(input_transform, mask=torch.ones(seq, seq).triu(diagonal=1).bool().to(input.device), src_key_padding_mask=mask)
        outputs = self.output_projection(outputs_transform)



        latent_k_preprobabilities = self.subpolicy_output_layer(self.hidden_activation(self.k_first_layer(outputs)))
        latent_m_preprobabilities = self.termination_output_layer(outputs) + self.m_exploration_bias
        # if mask_input is not None:

        latent_ms, latent_m_probabilities = gumbel_softmax(latent_m_preprobabilities, temperature=temperature,
                                                          hard=True, deterministic=deterministic, return_prob=True)

        latent_ks, latent_k_probabilities = gumbel_softmax(latent_k_preprobabilities, temperature=temperature,
                                                          hard=True,
                                                          deterministic=deterministic, return_prob=True)
        #FIXME not used

        latent_ms[:, 0, 0] = 1
        latent_ms[:, 0, 1] = 0
        sampled_m_indexes = latent_ms#combine_tensors(latent_ms, presampled_m)
        sampled_k_indexes = []
        #TODO
        log_ks = []

        for bb in range(latent_ms.shape[0]):
            sampled_k_index = combine_outputs(latent_ks[bb], sampled_m_indexes[bb])
            log_k = (sampled_k_index * latent_k_probabilities).sum(dim=-1, keepdim=True)
            sampled_k_indexes.append(sampled_k_index)
            log_ks.append(log_k)
        sampled_k_indexes = torch.stack(sampled_k_indexes).view(b, seq, -1)
        sampled_k_index_last = torch.cat(
            ((torch.zeros((b, 1, sampled_k_indexes.shape[2]))).to(input.device), sampled_k_indexes[:, :-1, :]),
            dim=1).detach()

        k_prob = sampled_k_index_last * latent_m_probabilities[:, :,
                                        1:] + latent_k_probabilities * latent_m_probabilities[:, :, :1]
        return sampled_k_indexes, sampled_m_indexes.view(b, seq,
                                                         -1), k_prob, latent_m_probabilities  # cross_entropy

    def get_actions(self, state_input, action_input, goal_input, skill_input, m_input, presampled_m, greedy=False, temperature=1.0):

        b, seq, _, _, _ = state_input.size()
        state_input = state_input.view(-1, state_input.shape[2], state_input.shape[3], state_input.shape[4])
        state_transform = self.obs_enc(state_input).detach()  # Fully connected layer
        state_transform = state_transform.view(b, seq, -1)
        action_input = action_input.view(-1, action_input.shape[2]).to(torch.float32)
        act_transform = self.act_enc(action_input).detach()
        act_transform = act_transform.view(b, seq, -1)
        goal_input = goal_input.view(-1, seq, goal_input.shape[2])
        lang_transform = self.lang_enc(goal_input).detach()
        lang_transform = lang_transform.view(b, seq, -1)
        # input = torch.cat([state_transform, skill_input, lang_transform], dim=-1)
        mask = ~((goal_input != -1).any(dim=-1))
        presampled_m = presampled_m.to(torch.float32)
        input = torch.cat([state_transform, act_transform, lang_transform], dim=-1)

        input_transform = self.input_projection(input)
        outputs_transform = self.transformer_encoder(input_transform, mask=torch.ones(seq, seq).triu(diagonal=1).bool().to(input.device), src_key_padding_mask=mask)
        # outputs_transform = self.transformer_encoder(input_transform, is_causal=True,
        #                                              src_key_padding_mask=mask)[:,:1]

        outputs = self.output_projection(outputs_transform)

        latent_k_preprobabilities = self.subpolicy_output_layer(self.hidden_activation(self.k_first_layer(outputs)))
        latent_m_preprobabilities = self.termination_output_layer(outputs) + self.m_exploration_bias

        latent_ms, latent_m_probabilities = gumbel_softmax(latent_m_preprobabilities, temperature=temperature,
                                                           hard=True, deterministic=greedy, return_prob=True)

        latent_ks, latent_k_probabilities = gumbel_softmax(latent_k_preprobabilities, temperature=temperature,
                                                           hard=True,
                                                           deterministic=greedy, return_prob=True)
        # TODO

        latent_ms[:, 0, 0] = 1
        latent_ms[:, 0, 1] = 0
        sampled_m_indexes = latent_ms #combine_tensors(latent_ms, presampled_m) #T0D0
        sampled_k_indexes = []

        #FIXME
        for bb in range(latent_ms.shape[0]):
            sampled_k_index = combine_outputs(latent_ks[bb], sampled_m_indexes[bb])
            sampled_k_indexes.append(sampled_k_index)
        return torch.stack(sampled_k_indexes).view(b, seq, -1).detach(), sampled_m_indexes.view(b, seq, -1).detach()
        #return latent_ks, sampled_m_indexes.view(b, seq, -1).detach()

