# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:41:50 2023

@author: diwaspaudel
"""

class Environment():
  def __init__(self, df_evs_seeking_charge, df_da_prices, df_rt_prices):
    self.df_evs_seeking_charge = df_evs_seeking_charge
    self.df_da_prices = df_da_prices
    self.df_rt_prices = df_rt_prices
    self.list_evs_seeking_charge = None
    self.list_da_prices = None 
    self.list_rt_prices = None


  def reset(self, index):
    self.list_evs_seeking_charge = self.df_evs_seeking_charge.iloc[index].to_numpy()
    self.list_da_prices = self.df_da_prices.iloc[index].to_numpy()
    self.list_rt_prices = self.df_rt_prices.iloc[index].to_numpy()