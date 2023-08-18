import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'


import argparse


parser_dqn_mha = argparse.ArgumentParser(description='PyTorch DQN-FF Args')
parser_dqn_mha.add_argument('--eps_start', type = float, default = 0.8, metavar = 'N', 
                    help = 'start value of epsilon')
parser_dqn_mha.add_argument('--eps_end', type = float, default = 0.05, metavar = 'N', 
                    help = 'end value of epsilon')
parser_dqn_mha.add_argument('--eps_decay', type = float, default = 0.8, metavar = 'N', 
                    help = 'decay rate of ep')
parser_dqn_mha.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser_dqn_mha.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser_dqn_mha.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser_dqn_mha.add_argument('--target_update_interval', type=int, default=100, metavar='G',
                    help='target update interval (default: 100)')
parser_dqn_mha.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser_dqn_mha.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123456)')
parser_dqn_mha.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser_dqn_mha.add_argument('--num_episodes', type=int, default=1000001, metavar='N',
                    help='maximum number of training episodes (default: 1000000)')
parser_dqn_mha.add_argument('--update_itr', type=int, default=2, metavar='N',
                    help='neural network updates frequency (default: 1)')
parser_dqn_mha.add_argument('--replay_size', type=int, default=256, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser_dqn_mha.add_argument('--state_dim', type = int, default = 7, metavar = 'N', 
                    help = 'dimension of state')
parser_dqn_mha.add_argument('--action_dim', type = int, default = 101, metavar = 'N', 
                    help = 'dimension of action')
parser_dqn_mha.add_argument('--hidden_dim', type = int, default = 256, metavar = 'N', 
                    help = 'number of units in hidden layer')
parser_dqn_mha.add_argument('--num_heads', type = int, default = 2, metavar = 'N', 
                    help = 'number of heads in multihead attention layer')
parser_dqn_mha.add_argument('--cuda_is_available', type = bool, default = True,
                    help = 'Gpu or cpu')
args_dqn_mha = parser_dqn_mha.parse_args()

#%%
import torch
import numpy as np
import matplotlib.pyplot as plt


from simulator.HubSelection import *
from simulator.AggregatedChargingDemand import *
from simulator.OptimizePowerSourcesPricing import *
from simulator.Environment import *
from simulator.EVCH import *

from common.buffers import replay_buffer_DQN
from common.buffers import ReplayBuffer_SAC

from dqn_mha import *
#%%

import pandas as pd 

df_da_commitments = pd.read_csv("./simulator/df_da_commitments_reduced.csv")


T = 24

df_da_prices = pd.read_csv("./simulator/df_da_prices.csv")
df_rt_prices = pd.read_csv("./simulator/df_rt_prices.csv")

df_evs_seeking_charge = pd.read_csv("./simulator/evs_seeking_charge.csv")

df_evs_seeking_charge = df_evs_seeking_charge

M_bss = 4000
m_bss = 500
M_bss_ch_rate = 2000
M_bss_dch_rate = 2000


n_hubs = 2
probab_price_sensitivity = 1
n_cs = [150, 150]
price_similarity_range = 0.05
# Instantiate objects
hub_selector = HubSelection(n_hubs, probab_price_sensitivity,  n_cs, price_similarity_range)
power_aggregator = AggregatedChargingDemand()
power_source_optimizer = OptimizePowerSourcesPricing(M_bss, m_bss, M_bss_ch_rate, M_bss_dch_rate)
env = Environment(df_evs_seeking_charge, df_da_prices, df_rt_prices)
hub1 = EVCH(df_da_commitments, M_bss, m_bss, M_bss_ch_rate, M_bss_dch_rate)
hub2 = EVCH(df_da_commitments, M_bss, m_bss, M_bss_ch_rate, M_bss_dch_rate)

average_da_price = df_da_prices.mean().tolist()


torch.manual_seed(args_dqn_mha.seed)


num_entries = int((2 - 1) / 0.01) + 1
action_list = [1 + i * 0.01 for i in range(num_entries)]
max_episodes = 2_000_005
rewards1 = []
rewards2 = []
epi_loss1 = []
epi_loss2 = []

test_index = [i for i in range(0, 365, 11)]
total_index = [i for i in range(365)]

train_index = [x for x in total_index if x not in test_index]

dqn_trainer1 = DQN_MHA(args_dqn_mha)
dqn_trainer2 = DQN_MHA(args_dqn_mha)     

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
def plot(rewards1, rewards2):
    plt.figure(figsize=(20,5))
    plt.plot(rewards1, label = 'agent1')
    plt.plot(rewards2, label = 'agent2')
    plt.grid(linestyle = '--')
    plt.legend()
    plt.show() 


if __name__ == '__main__':   
    
    
    r_buffer1 = replay_buffer_DQN(args_dqn_mha.replay_size)
    r_buffer2 = replay_buffer_DQN(args_dqn_mha.replay_size)
    
    for eps in range(max_episodes):
        if eps % 50_000 == 0:
                print('**********************')
                print('episode', eps)
                print('**********************')
        index = random.choice(train_index)
        
        # Reset the environment and hub
        env.reset(index)
        hub1.reset(index)
        hub2.reset(index)
    
        # randomly set the hub's power and price
        hub1.bss_power = random.randrange(500,3500)
        hub1.bss_price = random.randrange(20,80)
        
        hub2.bss_power = random.randrange(500,3500)
        hub2.bss_price = random.randrange(20,80)
        
        ep_r_price = 0
        ep_r_power = 0
            
        episode_state1 = []
        episode_action1 = []
        episode_last_action1 = []
        episode_reward1 = []
        episode_next_state1 = []
        episode_done1 = []
        
        episode_state2 = []
        episode_action2 = []
        episode_last_action2 = []
        episode_reward2 = []
        episode_next_state2 = []
        episode_done2 = []
        
        for step in range(24):
            t = step 
            
            if t == T-1:
                done = True
            else: 
                done = False
                
            # Get the number of EVs seeking charge at the current time step
            num_evs_seeking_charge = env.list_evs_seeking_charge[t]
            
            # Set the hub state price with the current time step, number of EVs seeking charge, DA prices and RT prices
            hub1.state_price = np.array([t, num_evs_seeking_charge, hub1.list_da_commitments[t], env.list_da_prices[t], env.list_rt_prices[t], hub1.bss_power, hub1.bss_price])
            hub2.state_price = np.array([t, num_evs_seeking_charge, hub2.list_da_commitments[t], env.list_da_prices[t], env.list_rt_prices[t], hub2.bss_power, hub2.bss_price])
                 
            action1 = dqn_trainer1.choose_action(torch.FloatTensor(hub1.state_price).unsqueeze(0).unsqueeze(0).to(device))
            #action1 = dqn_trainer1.choose_action(hub1.state_price)

            hub1.retail_price = action_list[action1] * min(env.list_da_prices[t], env.list_rt_prices[t])
          
            action2 = dqn_trainer2.choose_action(torch.FloatTensor(hub2.state_price).unsqueeze(0).unsqueeze(0).to(device))\
            #action2 = dqn_trainer2.choose_action(hub2.state_price)
            hub2.retail_price = action_list[action2] * min(env.list_da_prices[t], env.list_rt_prices[t])
            
            
            num_evs_in_each_hub = hub_selector.calculate_response(num_evs_seeking_charge, [hub1.retail_price, hub2.retail_price])
            
    
            # Check if the number of EVs in the first hub is None and set it to 0 if it is
            if num_evs_in_each_hub[0] == None:
              num_evs_in_each_hub[0] = 0
            
            if num_evs_in_each_hub[1] == None:
              num_evs_in_each_hub[1] = 0
                 
            power_aggregator.calculate_aggregated_ev_charging_demand(num_evs_in_each_hub[0])
            hub1.aggregated_ev_charging_demand = power_aggregator.actual_aggregated_ev_charging_demand
            
            power_aggregator.calculate_aggregated_ev_charging_demand(num_evs_in_each_hub[1])
            hub2.aggregated_ev_charging_demand = power_aggregator.actual_aggregated_ev_charging_demand
            
            
            if hub1.aggregated_ev_charging_demand == None:
              hub1.aggregated_ev_charging_demand = 0
              
              
            if hub2.aggregated_ev_charging_demand == None:
              hub2.aggregated_ev_charging_demand = 0
                
              
           
            hub1.da_ev, hub1.bss_ev, hub1.rt_ev, hub1.da_bss, hub1.da_rt,\
            hub1.profit_with_retail_price = power_source_optimizer.power_management(hub1.aggregated_ev_charging_demand,\
                                                                                   hub1.list_da_commitments[t],\
                                                                                   hub1.retail_price,\
                                                                                   env.list_da_prices[t],\
                                                                                   env.list_rt_prices[t],\
                                                                                   hub1.bss_power,
                                                                                   hub1.bss_price)
                
            
            
            hub2.da_ev, hub2.bss_ev, hub2.rt_ev, hub2.da_bss, hub2.da_rt,\
            hub2.profit_with_retail_price = power_source_optimizer.power_management(hub2.aggregated_ev_charging_demand,\
                                                                                   hub2.list_da_commitments[t],\
                                                                                   hub2.retail_price,\
                                                                                   env.list_da_prices[t],\
                                                                                   env.list_rt_prices[t],\
                                                                                   hub2.bss_power,
                                                                                   hub2.bss_price)
                    
                
            hub1.calculate_pricing_reward()
            
            hub1.update_bss_power_and_price()
            
            hub2.calculate_pricing_reward()
            
            hub2.update_bss_power_and_price()
            
            if t < T-1:
                hub1.next_state_price = np.array([t+1, env.list_evs_seeking_charge[t+1], hub1.list_da_commitments[t+1], env.list_da_prices[t+1], env.list_rt_prices[t+1], hub1.bss_power, hub1.bss_price])
                hub2.next_state_price = np.array([t+1, env.list_evs_seeking_charge[t+1], hub2.list_da_commitments[t+1], env.list_da_prices[t+1], env.list_rt_prices[t+1], hub2.bss_power, hub2.bss_price])
                
            else:
                hub1.next_state_price = np.repeat(1000, 7)
                hub2.next_state_price = np.repeat(1000, 7)
                
            episode_state1.append(hub1.state_price)
            episode_action1.append(action1)
            episode_reward1.append(hub1.reward_pricing)
            episode_next_state1.append(hub1.next_state_price)
            episode_done1.append(done) 
            
            episode_state2.append(hub2.state_price)
            episode_action2.append(action2)
            episode_reward2.append(hub2.reward_pricing)
            episode_next_state2.append(hub2.next_state_price)
            episode_done2.append(done) 
            
            r_buffer1.add([hub1.state_price, hub1.next_state_price, [action1], [hub1.reward_pricing], [done]])
            r_buffer2.add([hub2.state_price, hub2.next_state_price, [action2], [hub2.reward_pricing], [done]])
        
            dqn_trainer1.epsilon_scheduler.step(eps)
            dqn_trainer2.epsilon_scheduler.step(eps)
            
            if eps % 50_000 == 0:
                    print('------------------------------')
                    print('state', hub1.state_price)
                    print('t:', t, 'DRL-output1',action1,'DRL-output2',action2 )
                    print('p_ev1', hub1.retail_price,'pev2', hub2.retail_price)
                    print('da_price', env.list_da_prices[t], 'rt_price', env.list_rt_prices[t])
                    print('total evs', env.list_evs_seeking_charge[t])
                    print('ev in each hub', num_evs_in_each_hub)
                    print('agg ev charging demand', hub1.aggregated_ev_charging_demand, hub2.aggregated_ev_charging_demand)
                    print('rewards1', hub1.reward_pricing, 'rewards2', hub2.reward_pricing)
                    print('da_bss', hub1.da_bss)
                    print('epsilon', dqn_trainer1.epsilon_scheduler.epsilon)
            if done:
                    break
                
        if eps > 15:
            for i in range(args_dqn_mha.update_itr):
                sample1 = r_buffer1.sample(args_dqn_mha.replay_size)
                sample2 = r_buffer2.sample(args_dqn_mha.replay_size)
                loss1 = dqn_trainer1.learn(sample1)
                loss2 = dqn_trainer2.learn(sample2)
            epi_loss1.append(loss1)
            epi_loss2.append(loss2)
                
                
                
        if eps % 50 == 0 and eps>0: # plot and model saving interval
                plot(rewards1, rewards2)
                np.save('./model/dqn_mha_dqn_mha/rewards1', rewards1)
                np.save('./model/dqn_mha_dqn_mha/rewards2', rewards2)
          
               
        if eps % 50_000 == 0:
            print('Episode: ', eps,'|agent1:', np.sum(episode_reward1),'|agent2:', np.sum(episode_reward2))
        rewards1.append(np.sum(episode_reward1))
        rewards2.append(np.sum(episode_reward2))  
        
        if eps % 50 == 0:
            dqn_trainer1.save_model1(eps)
            dqn_trainer2.save_model2(eps)        
    