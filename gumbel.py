import numpy as np
import math
import pickle
from torch.autograd import Variable
import torch
import torch.nn.functional as F

def sample_gumbel(shape, eps=1e-10, use_cuda=False):
    """Sample from Gumbel(0, 1)"""
    if use_cuda:
        tens_type = torch.cuda.FloatTensor
    else:
        tens_type = torch.FloatTensor
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature, dim=-1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    y = logits + gumbels
    #y = logits + sample_gumbel(logits.shape, use_cuda=logits.is_cuda)

    return F.softmax(y / temperature, dim=dim)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False, deterministic=False, dim=-1, return_prob=False, masked_actions=None):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """

    if not deterministic:
        y = gumbel_softmax_sample(logits, temperature, dim=dim)
    else:
        y = F.softmax(logits, dim=dim)
    if masked_actions is not None:
        y = y * masked_actions
    if hard:
        y_hard = onehot_from_logits(y, dim=dim)
        y_out = (y_hard - y).detach() + y

        if not return_prob:
            return y_out
        else:
            return y_out, y
    else:
        return y

def onehot_from_logits(logits, eps=0.0, dim=-1):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(dim, keepdim=True)[0]).float()
    #print(logits[0],"a")
    #print(len(argmax_acs),argmax_acs[0])
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False).cuda()

    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]).cuda())])
