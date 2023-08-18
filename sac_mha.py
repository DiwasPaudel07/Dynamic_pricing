
import torch
from common.actor_networks import Actor_MHA
from common.critic_networks import Critic_MHA
import torch.nn.functional as f

import copy
import numpy as np


#%%
class SAC_MHA():
    def __init__(
            self, args):
        
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.hidden_dim = args.hidden_dim
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.alpha = args.alpha
        self.adaptive_alpha = args.adaptive_alpha
        self.lr = args.lr
        self.num_heads = args.num_heads
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor_MHA(self.state_dim, self.action_dim, self.hidden_dim, self.num_heads, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.lr)
        self.q1 = Critic_MHA(self.state_dim, self.action_dim, args.hidden_dim, self.num_heads, self.device)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr = self.lr)
        self.q2 = Critic_MHA(self.state_dim, self.action_dim, self.hidden_dim, self.num_heads, self.device)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr = self.lr)
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)
        
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.target_q1.parameters():
            p.requires_grad = False
            
        for p in self.target_q2.parameters():
            p.requires_grad = False
            
        
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            #self.target_entropy = torch.tensor(-action_dim, dtype=float, requires_grad=True, device=device)
            self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=self.device)
            # We learn log_alpha instead of alpha to ensure exp(log_alpha)=alpha>0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)
            
    def select_action(self, state, deterministic, with_logprob=False):
        # only used when interact with the env
        with torch.no_grad():
            #state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            a, _ = self.actor(state, deterministic, with_logprob)
        return a.cpu().numpy().flatten()
    
    def train(self,replay_buffer):
        reward_scale = 10
        s, a, r, s_prime, dead_mask = replay_buffer.sample(self.batch_size)
        r = reward_scale * (r - r.mean(dim=0)) / (r.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
        s = s.unsqueeze(0)
        a = a.unsqueeze(0)
        r = r.unsqueeze(0)
        s_prime = s_prime.unsqueeze(0)
        dead_mask = dead_mask.unsqueeze(0)

        
        #----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_prime, log_pi_a_prime = self.actor(s_prime)
            target_Q1 = self.target_q1(s_prime, a_prime)
            target_Q2 = self.target_q2(s_prime, a_prime)

            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (1 - dead_mask) * self.gamma * (target_Q - self.alpha * log_pi_a_prime) #Dead or Done is tackled by Randombuffer

        # Get current Q estimates
        current_Q1 = self.q1(s, a)
        current_Q2 = self.q2(s, a)
        q1_loss = f.mse_loss(current_Q1, target_Q) 
        q2_loss = f.mse_loss(current_Q2, target_Q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        #----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for params in self.q1.parameters():
            params.requires_grad =     False
        for params in self.q2.parameters():
            params.requires_grad =     False

        a, log_pi_a = self.actor(s)
        current_Q1 = self.q1(s, a)
        current_Q2 = self.q2(s, a)
        Q = torch.min(current_Q1, current_Q2)

        a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        for params in self.q1.parameters():
            params.requires_grad = True
        for params in self.q2.parameters():
            params.requires_grad = True
        #----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # we optimize log_alpha instead of aplha, which is aimed to force alpha = exp(log_alpha)> 0
            # if we optimize aplpha directly, alpha might be < 0, which will lead to minimun entropy.
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        #----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
           ############################################# 
        return a_loss, q1_loss
        
    def save_model1(self, episode):
            torch.save(self.q1.state_dict(), "./model/agent1_sac_mha_q1_{}.pth".format(episode))
            torch.save(self.q2.state_dict(), "./model/agent1_sac_mha_q2_{}.pth".format(episode))
            torch.save(self.actor.state_dict(), "./model/agent1_sac_mha_policy_{}.pth".format(episode))
            
            
    def save_model2(self, episode):
            torch.save(self.q1.state_dict(), "./model/agent2_sac_mha_q1_{}.pth".format(episode))
            torch.save(self.q2.state_dict(), "./model/agent2_sac_mha_q2_{}.pth".format(episode))
            torch.save(self.actor.state_dict(), "./model/agent2_sac_mha_policy_{}.pth".format(episode))