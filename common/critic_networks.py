




import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init
from common.multi_head_attention import MultiHeadAttention



class Critic_FF(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, device):
        super(Critic_FF, self).__init__()

        self.fc1 = nn.Linear(num_inputs + num_actions, hidden_size).to(device)
        self.fc2 = nn.Linear(hidden_size, hidden_size).to(device)
        self.fc3 = nn.Linear(hidden_size, num_actions).to(device)
        
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Critic_MHA(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, num_heads, device):
        super(Critic_MHA, self).__init__()

        self.fc1 = nn.Linear(num_inputs + num_actions, hidden_size).to(device)
        self.attention = MultiHeadAttention(hidden_size, num_heads, device).to(device)
        self.fc2 = nn.Linear(2 * hidden_size, hidden_size).to(device)
        self.fc3 = nn.Linear(hidden_size, num_actions).to(device)
        
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, state, action):
        x = torch.cat((state, action), 2) # the dim 0 is number of samples
        #x = torch.cat([state, action])
        x = f.relu(self.fc1(x))
        x = x.permute(1, 0, 2)  # Reshape for multihead attentio
        attn_output = f.relu(self.attention(x))
        x = torch.cat([x, attn_output], -1)# Residual connection
       # x = F.relu(self.attention(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.permute(1, 0, 2)  # Reshape back to original shape
        return x
            