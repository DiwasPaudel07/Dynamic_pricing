# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:39:18 2023

@author: diwaspaudel
"""
from scipy.stats import rv_discrete
import numpy as np

class AggregatedChargingDemand():

  def __init__(self):
    # Initialize instance variables to None or default values
    self.num_evs_in_the_hub = None 
    self.x_battery_size = np.array([50, 75, 100]) # Battery size options for EVs
    self.px_battery_size = np.array([0.3, 0.4, 0.3]) # Probability distribution of battery size options
    self.x_starting_soc = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95]) # Starting state-of-charge options for EVs
    self.px_starting_soc = np.array([0.22, 0.08, 0.095, 0.105, 0.125, 0.105, 0.1, 0.07, 0.07, 0.03]) # Probability distribution of starting state-of-charge options

    self.battery_size_distribution = None 
    self.starting_soc_distribution = None 
    
    self.actual_aggregated_ev_charging_demand = None 

  def calculate_aggregated_ev_charging_demand(self, num_evs_in_the_hub):
    self.num_evs_in_the_hub = num_evs_in_the_hub

    # Generate a random battery size distribution for the EVs
    self.battery_size_distribution = rv_discrete(values = (self.x_battery_size, self.px_battery_size)).rvs(size = self.num_evs_in_the_hub)

    # Generate a random starting state-of-charge distribution for the EVs
    self.starting_soc_distribution = rv_discrete(values = (self.x_starting_soc, self.px_starting_soc)).rvs(size = self.num_evs_in_the_hub)

     # Calculate the potential aggregated EV charging demand based on battery size and starting state-of-charge
    self.actual_aggregated_ev_charging_demand = 0.01 * np.sum(self.battery_size_distribution * (100 - self.starting_soc_distribution))

    
