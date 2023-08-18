import torch
import torch.nn as nn

from common.epsilon_scheduler import EpsilonScheduler
from common.actor_networks import QNetwork_MHA
from common.buffers import transition


import numpy as np
#%%

class DQN_MHA(object):
    def __init__(self, args):
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.hidden_dim = args.hidden_dim
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_update_interval = args.target_update_interval
        

        self.eval_net = QNetwork_MHA(self.state_dim, self.action_dim, self.hidden_dim, self.device).to(self.device)
        self.target_net = QNetwork_MHA(self.state_dim, self.action_dim, self.hidden_dim, self.device).to(self.device)
        self.learn_step_counter = 0                                     # for target updating
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.loss_func = nn.MSELoss()
        self.epsilon_scheduler = EpsilonScheduler(args.eps_start, args.eps_end, args.eps_decay)
        self.updates = 0

    def choose_action(self, x):
        
        epsilon = self.epsilon_scheduler.get_epsilon()
        if np.random.uniform() > epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 2)[1].data.cpu().numpy()[0][0]     # return the argmax
        else:   # random
            action = np.random.randint(0, self.action_dim)
        return action

    def learn(self, sample,):
        batch_samples = transition(*zip(*sample))

        states = torch.cat(batch_samples.state).float().to(self.device)
        next_states = torch.cat(batch_samples.next_state).float().to(self.device)
        actions = torch.cat(batch_samples.action).to(self.device)
        rewards = torch.cat(batch_samples.reward).float().to(self.device)
        is_terminal = torch.cat(batch_samples.is_terminal).to(self.device)
        
        is_terminal = is_terminal.unsqueeze(0)
        rewards = rewards.unsqueeze(0)
        
        Q = self.eval_net(states.unsqueeze(0).to(self.device))        
        Q_s_a = Q.gather(2, actions.view(1, 256, 1))

        Q_s_prime_a_prime = self.target_net(next_states.unsqueeze(0).to(self.device)).detach()

        Q_s_prime_a_prime, _ = torch.max(Q_s_prime_a_prime, dim=2)
        
        Q_s_prime_a_prime = Q_s_prime_a_prime.unsqueeze(2)
        
        # Compute the target
        target = rewards + self.gamma * (1 - is_terminal.int())* Q_s_prime_a_prime

        loss = self.loss_func(target.detach(), Q_s_a)
        
        # Zero gradients, backprop, update the weights of policy_net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.target_update_interval == 0:
            self.update_target()

        return loss.item()

    def save_model1(self, episode):
        torch.save(self.eval_net.state_dict(), "./model/dqn_mha_dqn_mha/agent1_dqn_mha_mha_eps{}.pth".format(episode))
    
    def save_model2(self, episode):
        torch.save(self.eval_net.state_dict(), "./model/dqn_mha_dqn_mha/agent2_dqn_mha_mha_eps{}.pth".format(episode))
        
    def load_model1(self, episode):
        self.eval_net.load_state_dict(torch.load("./model/dqn_mha_dqn_mha/agent1_dqn_mha_mha_eps{}.pth".format(episode), map_location=torch.device('cpu')))
        self.eval_net.eval()
        
    def load_model2(self, episode):
        self.eval_net.load_state_dict(torch.load("./model/dqn_mha_dqn_mha/agent2_dqn_mha_mha_eps{}.pth".format(episode), map_location=torch.device('cpu')))
        self.eval_net.eval()


    def update_target(self, ):
        """
        Update the target model when necessary.
        """
        self.target_net.load_state_dict(self.eval_net.state_dict())