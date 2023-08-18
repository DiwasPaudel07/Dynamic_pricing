






import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init
from torch.distributions import Normal
import numpy as np


from common.multi_head_attention import MultiHeadAttention

class QNetwork(nn.Module):
    def __init__(self, obs_shape, act_shape, hidden_units=256):
        super(QNetwork, self).__init__()
        in_dim = obs_shape
        out_dim = act_shape
              
        self.fc1 = nn.Linear(in_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, out_dim)
        
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        action_value = self.fc3(x)
        return action_value
    
    
class QNetwork_MHA(nn.Module):
    def __init__(self, obs_shape, act_shape, hidden_units, device):
        super(QNetwork_MHA, self).__init__()
        in_dim = obs_shape
        out_dim = act_shape
              
        self.fc1 = nn.Linear(in_dim, hidden_units)
        self.attention = MultiHeadAttention(hidden_units, hidden_units, device).to(device)
        self.fc2 = nn.Linear(2*hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, out_dim)
        
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        #print('input_dim', x.size())
        x = f.relu(self.fc1(x))
        #print('input_dim', x.size())
        x = x.permute(1, 0, 2)  # Reshape for multihead attention
        #print('permutated dim', x.size())
        attn_output = f.relu(self.attention(x))
        x = torch.cat([x, attn_output], -1)  
        x = f.relu(self.fc2(x))
        x =  self.fc3(x)
        action_value = x.permute(1, 0, 2)  # Reshape back to original shape
        #print('output dim', action_value.size())
        #action_value = self.fc3(x)
        return action_value



class Actor_FF(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, device):
        super(Actor_FF, self).__init__()
        
        self.log_std_min = -20
        self.log_std_max = 2
        
        self.fc1 = nn.Linear(num_inputs, hidden_size).to(device)
        self.fc2 = nn.Linear(hidden_size , hidden_size).to(device)
        self.mean_linear = nn.Linear(hidden_size, num_actions).to(device)
        self.log_std_linear = nn.Linear(hidden_size, num_actions).to(device)
        
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.mean_linear.weight)
        init.xavier_uniform_(self.log_std_linear.weight)
        
        
    def forward(self, state, deterministic = False, with_logprob = True):
        x = f.relu(self.fc1(state))
        x = f.relu(self.fc2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        if deterministic: u = mean
        else: u = dist.rsample()
        a = torch.tanh(u)
        
        if with_logprob:
            # get probability density of logp_pi_a from probability density of u, which is given by the original paper.
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)

            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - f.softplus(-2 * u))).sum(axis=1, keepdim=True)
        else:
            logp_pi_a = None
            

        return a, logp_pi_a
    
    


class Actor_MHA(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, num_heads, device):
        super(Actor_MHA, self).__init__()
        
        self.log_std_min = -20
        self.log_std_max = 2
        
        self.fc1 = nn.Linear(num_inputs, hidden_size).to(device)
        self.attention = MultiHeadAttention(hidden_size, num_heads, device).to(device)
        self.fc2 = nn.Linear(2 * hidden_size , hidden_size).to(device)
        self.mean_linear = nn.Linear(hidden_size, num_actions).to(device)
        self.log_std_linear = nn.Linear(hidden_size, num_actions).to(device)
        
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.mean_linear.weight)
        init.xavier_uniform_(self.log_std_linear.weight)




        
        
    def forward(self, state, deterministic = False, with_logprob = True):
        x = f.relu(self.fc1(state))
        x = x.permute(1, 0, 2)  # Reshape for multihead attention
        attn_output = f.relu(self.attention(x))
        x = torch.cat([x, attn_output], -1)  # Residual connection
        x = f.relu(self.fc2(x))
        x = x.permute(1, 0, 2)  # Reshape back to original shape
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        if deterministic: u = mean
        else: u = dist.rsample()
        a = torch.tanh(u)
        
        if with_logprob:
            # get probability density of logp_pi_a from probability density of u, which is given by the original paper.
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)

            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - f.softplus(-2 * u))).sum(axis=1, keepdim=True)
        else:
            logp_pi_a = None
            

        return a, logp_pi_a
