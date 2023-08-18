

import torch
import torch.nn as nn

from common.epsilon_scheduler import EpsilonScheduler
from common.actor_networks import QNetwork
from common.buffers import transition


import numpy as np
#%%

class DQN(object):
    def __init__(self, args):
        
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.hidden_dim = args.hidden_dim
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_update_interval = args.target_update_interval
        

        self.eval_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.learn_step_counter = 0                                     # for target updating
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.loss_func = nn.MSELoss()
        self.epsilon_scheduler = EpsilonScheduler(args.eps_start, args.eps_end, args.eps_decay)
        self.updates = 0

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(self.device)
        # input only one sample
        # if np.random.uniform() > EPSILON:   # greedy
        epsilon = self.epsilon_scheduler.get_epsilon()
        if np.random.uniform() > epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]     # return the argmax
            # print(action)
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
       
        Q = self.eval_net(states) 
        Q_s_a=Q.gather(1, actions)

        
        none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0], dtype=torch.int64, device=self.device)
        none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)

        Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=self.device)
        if len(none_terminal_next_states) != 0:
            Q_s_prime_a_prime[none_terminal_next_state_index] = self.target_net(none_terminal_next_states).detach().max(1)[0].unsqueeze(1)

        Q_s_prime_a_prime = (Q_s_prime_a_prime-Q_s_prime_a_prime.mean())/ (Q_s_prime_a_prime.std() + 1e-5)  # normalization
        
        # Compute the target
        target = rewards + self.gamma * Q_s_prime_a_prime

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
        torch.save(self.eval_net.state_dict(), "./model/dqn_ff_dqn_ff/agent1_dqn_ff_ff_eps{}.pth".format(episode))
    
    def save_model2(self, episode):
        torch.save(self.eval_net.state_dict(), "./model/dqn_ff_dqn_ff/agent2_dqn_ff_ff_eps{}.pth".format(episode))


    def update_target(self, ):
        """
        Update the target model when necessary.
        """
        self.target_net.load_state_dict(self.eval_net.state_dict())