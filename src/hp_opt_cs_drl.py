import torch
import random
import numpy as np
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import math

class PolicyNet(nn.Module):
    def __init__(self,input_dim):
        super(PolicyNet, self).__init__()

        self.input_dim = input_dim
        in_chanel = input_dim*input_dim + input_dim
        self.fc_layer = nn.Sequential(
            nn.Linear(in_chanel, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, in_chanel),
            nn.Tanh()
        )

    def forward(self, mean, precision_component):
        tmp = torch.cat([mean, precision_component.view(-1)])
        input = torch.unsqueeze(tmp,0)
        x = torch.squeeze(self.fc_layer(input))/100.0
        mean_update = x[:self.input_dim]
        precision_component_update = x[self.input_dim:].reshape([self.input_dim,self.input_dim])

        return mean_update,precision_component_update

class learnable_highd_gaussian_continuous(nn.Module):
    def __init__(self,vals,initial_precision = None,max_precision = None,fixed_precision = True, dirichlet=False, n_clients=0, rl_nettype='mlp'):
        super(learnable_highd_gaussian_continuous,self).__init__()

        self.dim = len(vals)
        if rl_nettype == 'mlp':
            self.PolicyNet = PolicyNet(self.dim)
        else:
            raise NotImplementedError

        self.vals = [np.array(x) for x in vals]

        self.vals_center = np.array([(x[0] + x[-1])/2 for x in self.vals])
        self.vals_scale = np.array([x[-1] - x[0] for x in self.vals])

        self.dirichlet = dirichlet
        self.n_clients = n_clients

        self.mean =torch.zeros(self.dim) + 10e-8

        precision_val = 5.0 if initial_precision is None else initial_precision
        precision_component = torch.sqrt(torch.eye(self.dim) * precision_val) + 10e-8
        if fixed_precision:
            self.register_buffer('precision_component',precision_component)
        else:
            self.precision_component = precision_component
            
        if max_precision is not None:
            self.max_precision_component = math.sqrt(self.max_precision)
        else:
            self.max_precision_component = None


    def forward(self):

        mean_update,precision_component_update = self.PolicyNet(self.mean,self.precision_component)
        self.mean = self.mean + mean_update
        self.precision_component = self.precision_component + precision_component_update
        self.mean.data.copy_(torch.clamp(self.mean.data,-1.0,1.0))

        dist = MultivariateNormal(loc=self.mean,
                                  precision_matrix=self.precision_component)
        sample = dist.sample()
        logprob = dist.log_prob(sample)
        sample = sample * self.vals_scale + self.vals_center

        if self.dirichlet:
            aw_tensor = self.dist_d.rsample()
            logprob_d = self.dist_d.log_prob(aw_tensor)
            weight = [aw_tensor[i].item() for i in range(self.n_clients)]
            return sample,logprob, weight, logprob_d

        return sample, logprob

def mean_and_cut(v_losses,cutoff_round):
    v_losses_means  = np.array([x[x != -np.inf].mean() for x in np.array(v_losses)[1:]])
    v_losses_means_cut = v_losses_means[cutoff_round:]
    return v_losses_means_cut
    
def get_hyper_loss(val_losses,cutoff_round,logprob_history,no_reinforce_baseline,windowed_updates, logprob_d_history=None):
    # refernce https://gitlab.com/anon.iclr2020/robust_federated_learning/-/blob/master/hparam_optimization.py
    val_losses_means_cut = mean_and_cut(val_losses,cutoff_round)
    loss = 0
    improvements = [0]
    if val_losses_means_cut.size > 1:
       improvements = - ((val_losses_means_cut[1:] / val_losses_means_cut[:-1]) - 1)
       mean_improvement = improvements.mean()
       if no_reinforce_baseline:
           mean_improvement = 0.0
       updates_limit  = len(improvements)+1 if windowed_updates else 2
       for j in range(1,updates_limit):
           loss = logprob_history[-j-1] * (improvements[-j] - mean_improvement)

       if logprob_d_history is not None: # optimize the dirichlet distribution
           for j in range(1, updates_limit):
               loss_d = logprob_d_history[-j - 1] * (improvements[-j] - mean_improvement)
           loss += loss_d

    return loss,improvements[-1]


