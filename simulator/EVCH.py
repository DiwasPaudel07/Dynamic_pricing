# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:43:03 2023

@author: diwaspaudel
"""

import numpy as np
import random
import copy


class EVCH():
  def __init__(self, df_da_commitments, M_bss, m_bss, M_bss_ch_rate, M_bss_dch_rate):

    self.df_da_commitments = df_da_commitments

    self.list_da_commitments = None

    self.M_bss, self.m_bss, self.M_bss_ch_rate, self.M_bss_dch_rate = M_bss, m_bss, M_bss_ch_rate, M_bss_dch_rate
    self.bss_power, self.bss_price = None, None

    self.aggregated_ev_charging_demand = None
    self.aggregated_ev_charging_demand_with_baseline = None
    
    self.profit_with_retail_price = None 
    self.profit_with_baseline_price = None
    self.profit_with_retail_price_and_bss = None

    self.state_price = None 
    self.state_power = None 
    
    self.next_state_price = None 
    self.next_state_power = None 

    self.retail_price_tanh = None
    self.retail_price = None 
    self.baseline_price = None
    
    self.reward_pricing = None
    self.reward_power_management = None
    
    self.bss_action = None

    self.da_ev, self.bss_ev, self.rt_ev, self.da_bss, self.rt_bss, self.da_rt, self.bss_rt = None, None, None, None, None, None, None


  def reset(self, index):
    #self.list_da_commitments = self.df_da_commitments.iloc[index].to_numpy()
    self.list_da_commitments = self.df_da_commitments.iloc[0].to_numpy()



  def update_retail_price(self, p_da, p_rt): 
      
      self.retail_price = min(p_da, p_rt) + (np.clip(self.retail_price_tanh, -1, 1) + 1) *\
                                                ((2 * min(p_da, p_rt) -  min(p_da, p_rt))/2)
    

  def update_bss_action(self):
    #self.bss_action = self.power_mgmt_actor.select_action(self.state_power)
    #self.bss_action = random.uniform(-1 , 1)
    if self.bss_action < 0:
        self.bss_ch = - self.bss_action * min(self.M_bss_ch_rate, self.M_bss - self.bss_power)
        self.bss_dch = 0
    elif self.bss_action > 0:
      self.bss_ch = 0 
      self.bss_dch = self.bss_action *  min(self.M_bss_dch_rate, self.bss_power - self.m_bss)
    elif self.bss_action == 0:
      self.bss_ch = 0 
      self.bss_dch = 0


  def calculate_pricing_reward(self):
    self.reward_pricing = self.profit_with_retail_price

        
    
    

  def calcualte_power_mgmt_reward(self):
    if self.bss_action > 0:
        if self.bss_power <= self.m_bss+100:
            self.reward_power_management = -300
        else:
            self.reward_power_management = np.clip(self.profit_with_retail_price_and_bss -  self.profit_with_baseline_price, -500, 500)
    elif self.bss_action < 0:
        if self.bss_power >= self.M_bss-100:
            self.reward_power_management = -300
        else:
            self.reward_power_management = np.clip(self.profit_with_retail_price_and_bss -  self.profit_with_baseline_price, -500, 500)
    elif self.bss_action == 0:
        self.reward_power_management = np.clip(self.profit_with_retail_price_and_bss -  self.profit_with_baseline_price, -500, 500)
        
  


  def update_bss_power_and_price(self):

    self.bss_price = (copy.deepcopy(self.bss_price) * copy.deepcopy(self.bss_power) + copy.deepcopy(self.da_bss) * copy.deepcopy(self.state_price[3]))/(copy.deepcopy(self.bss_power) + copy.deepcopy(self.da_bss)) 

    self.bss_power = (copy.deepcopy(self.bss_power) + copy.deepcopy(self.da_bss) - copy.deepcopy(self.bss_ev))

