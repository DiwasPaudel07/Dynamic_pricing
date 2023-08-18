# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:27:18 2023

@author: diwaspaudel
"""

import numpy as np
import math

class HubSelection():
  """
  
  
  ...
  
  Attrubutres 
  -----------
  
  n_hubs : int
      the number of hubs in competition
      
  probab_price_sensitivity: float
      probability that an EV is sensitive to the price
      
  n_cs_in_hubs : list of int of size equal to the n_hubs
      number of charging station in each hub
      
  n_cs_occupied : list of int 
      number of EVs in each hub
      
  price_similarity_range: float
      range up to which the two prices will be considered similar by EVs
            
  p_non_price_responsive_evs_selecting_a_hub : numpy array 
  
  p_balking : list of float
  
  x_balking : range of balking
  
  
  Methods
  ----------------------
  
  calculate_hubs_with_similar_prices(prices, ranked_prices)
      Finds the hubs with prices within price similarity range    
  
  find_range(number)
      Finds the range where the non cheapest hub's price falls
  
  calculate_response(n_evs, prices)
      Distributes the EVs among the hub

      
  """
  def __init__(self, n_hubs, probab_price_sensitivity,  n_cs, price_similarity_range):
      
    """
    Parameters
    --------------------
    
    n_hubs : The number of hubs in competition 
    
    probab_price_sensitivity : 
        
    n_cs :
        
    price_similarity_range
    
    """   
      
    self.n_hubs = n_hubs
    self.probab_price_sensitivity = probab_price_sensitivity
    self.n_cs_in_hubs = n_cs
    self.n_cs_occupied = np.zeros(n_hubs)
    self.p_non_price_responsive_evs_selecting_a_hub = np.array([1/n_hubs for i in range(n_hubs)])
    self.price_similarity_range = price_similarity_range
    self.similar_hubs = None
    self.p_balking = [0.1, 0.2, 0.35, 0.6, 0.8, 1 ]
    self.x_balking = [(0, 1.10), (1.10, 1.20), (1.20, 1.35), (1.35, 1.50), (1.50, 1.750), (1.75, 200.0)]


  def calculate_hubs_with_similar_prices(self, prices, ranked_prices):
    """
    Finds the hubs with similar prices
    
    Parameters
    ----------------------
    
    prices : list of float
        The list of price set by all the competing hubs
        
    ranked prices : list of floats
        The list of ranked prices
    
    
    """  
    
    ref_price = prices[ranked_prices[0]]
    similar_hubs = [ranked_prices[0]]
    for i in ranked_prices[1:]:
      if abs(prices[i] - ref_price)/(ref_price + 1e-7) <= self.price_similarity_range:
        similar_hubs.append(i)

    return similar_hubs

  def find_range(self, number):
    """
    Finds the range where the competing hub's price falls as compared to the cheapest hub
    
    Parameters
    --------------------
    
    numnber: float
        The ratio of competing hubs price with respect to the cheapest hub
    
    """
    starts, ends = np.array(self.x_balking).T
    idx = np.where(np.logical_and(starts <= number, number < ends))[0]
    if len(idx) > 0:
        return idx[0]
    else:
        return -1
    
  def calculate_response(self, n_evs, prices):
    """
    Distributes the EVs across the hubs based on their price
        First, distributes the non-price responsive EVs among the hub 
        Then, finds the hubs similar in prices and fills those
        Remaining EVs are then distributed to the non-similar hubs considering the balking probability 
    
    Parameters
    -------------------
    n_evs : int 
        The number of EVs seeking to charge in the given hour
        
    prices : list of prices by all the competing hubs
    
    """  
    
    n_price_responsive_evs = np.random.binomial(n_evs, self.probab_price_sensitivity)
    self.n_cs_occupied = np.random.multinomial(n_evs - n_price_responsive_evs, self.p_non_price_responsive_evs_selecting_a_hub)

    ranked_prices = [i[0] for i in sorted(enumerate(prices), key = lambda x:x[1])]

    self.similar_hubs = self.calculate_hubs_with_similar_prices(prices, ranked_prices)
    n_remain_evs = n_price_responsive_evs
    for hub in self.similar_hubs:
      self.n_cs_occupied[hub] = min(self.n_cs_in_hubs[hub], self.n_cs_occupied[hub] + math.floor(n_price_responsive_evs/len(self.similar_hubs)))
      n_remain_evs -= self.n_cs_occupied[hub] 

    for hub in ranked_prices:
      if hub not in self.similar_hubs:
        self.n_cs_occupied[hub] = min(self.n_cs_in_hubs[hub], self.n_cs_occupied[hub] + n_remain_evs)
        n_remain_evs -= self.n_cs_occupied[hub]
        idx = self.find_range(prices[hub]/(prices[ranked_prices[0]] + 1e-7))
        self.n_cs_occupied[hub] = self.n_cs_occupied[hub] - np.random.binomial(self.n_cs_occupied[hub], self.p_balking[idx])

    return self.n_cs_occupied