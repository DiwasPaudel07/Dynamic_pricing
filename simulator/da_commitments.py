# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:56:08 2023

@author: diwaspaudel
"""

import math
from AggregatedChargingDemand import *
import numpy as np
import pandas as pd

power_aggregator = AggregatedChargingDemand()

# Calculate expected EVs in the hub
df_evs_seeking_charge = pd.read_csv("evs_seeking_charge.csv")
expected_evs_in_hub = df_evs_seeking_charge.mean() * (1/2)

# Initialize mean DA commitment list
mean_da_commitment = []

# Calculate mean DA commitment for each hour
for hour in range(24):
    power_aggregator.calculate_aggregated_ev_charging_demand(math.ceil(expected_evs_in_hub.iloc[hour]))
    mean_da_commitment.append(power_aggregator.actual_aggregated_ev_charging_demand)
# Set random seeds for reproducibility
np.random.seed(123)
# Generate DA commitments for each day
da_com = np.random.randint(0.75 * np.array(mean_da_commitment), 1.25 * np.array(mean_da_commitment), size=(365, 24))

np_mean = np.mean(da_com, axis=0)

data_2d = np_mean.reshape(1, -1)

mean_da_commitments = pd.DataFrame(data_2d, columns=["hour_"+str(i) for i in range(1,25)])

# =============================================================================
# np.savetxt('df_mean_da_commitments.csv', data_2d, delimiter=',')
# #%%
# =============================================================================
df_mean_da_commitments = pd.DataFrame(mean_da_commitments, columns=["hour_"+str(i) for i in range(1,25)])
df_mean_da_commitments.to_csv('df_mean_da_commitments.csv', index=False)

import matplotlib.pyplot as plt
plt.plot(np_mean, label = 'mean')
plt.plot(da_com[0], label = 'actual')
plt.legend()
plt.show()


#%%