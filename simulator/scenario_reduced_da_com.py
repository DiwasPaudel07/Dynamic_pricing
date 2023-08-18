# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 07:37:29 2023

@author: diwaspaudel
"""

import math
from AggregatedChargingDemand import *
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

power_aggregator = AggregatedChargingDemand()

# Calculate expected EVs in the hub
df_evs_seeking_charge = pd.read_csv("evs_seeking_charge.csv")
expected_evs_in_hub = df_evs_seeking_charge.mean() * (1/2)

# Initialize mean DA commitment list
mean_ev_charging_demand= []

np.random.seed(123)

# Calculate mean DA commitment for each hour
for hour in range(24):
    power_aggregator.calculate_aggregated_ev_charging_demand(math.ceil(expected_evs_in_hub.iloc[hour]))
    mean_ev_charging_demand.append(power_aggregator.actual_aggregated_ev_charging_demand)
    
ev_charging_demand = np.random.randint(0.5 * np.array(mean_ev_charging_demand), 1.5 * np.array(mean_ev_charging_demand), size=(320, 24))

# Read the CSV file into a DataFrame
df_da = pd.read_csv("C:/Users/diwaspaudel/Desktop/MA-DRL/AAAI/simulator/df_da_prices.csv")
df_rt = pd.read_csv("C:/Users/diwaspaudel/Desktop/MA-DRL/AAAI/simulator/df_rt_prices.csv")
a = df_da.to_numpy()

b = df_rt.to_numpy()


def scenario_reduction(day_ahead_price, real_time_price, ev_demand, k):
    # Reshape the arrays to combine the data for each day
    num_days = day_ahead_price.shape[0]
    day_ahead_price = day_ahead_price.reshape(num_days, -1)
    real_time_price = real_time_price.reshape(num_days, -1)
    ev_demand = ev_demand.reshape(num_days, -1)

    # Combine the arrays into a single dataset
    data = np.column_stack((day_ahead_price, real_time_price, ev_demand))

    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(standardized_data)

    # Retrieve the cluster centroids
    centroids = kmeans.cluster_centers_

    # Assign each data point to the nearest centroid
    labels = kmeans.labels_

    # Calculate the frequencies of each cluster
    frequencies = np.bincount(labels, minlength=k)

    # Calculate the probabilities of each reduced scenario
    probabilities = frequencies / len(labels)
    
    original_centroids = scaler.inverse_transform(centroids)


    return original_centroids, probabilities

# Example usage
num_days = 365
num_hours = 24


k = 10

reduced_scenarios, probabilities = scenario_reduction(a, b, ev_charging_demand, k)



#%%
r_da = reduced_scenarios[:, 0:24]
r_rt = reduced_scenarios[:, 24:48]
r_ev_demand = reduced_scenarios[:, 48:72]

n_scenarios = 10
T = 24
M_ch = 2000
M_phi = 4000
M_dch = 2000
m_phi = 500
m_ch = 200
m_dch = 200


#%%


import gurobipy as gp 
from gurobipy import GRB

model = gp.Model()

da_ev = model.addVars((i, t) for i in range(n_scenarios) for t in range(T))
bss_ev = model.addVars((i, t) for i in range(n_scenarios) for t in range(T))
rt_ev = model.addVars((i, t) for i in range(n_scenarios) for t in range(T))
da_rt = model.addVars((i, t) for i in range(n_scenarios) for t in range(T))
da_bss = model.addVars((i, t) for i in range(n_scenarios) for t in range(T))
x = model.addVars(((i, t) for i in range(n_scenarios) for t in range(T)), vtype = GRB.BINARY)
y = model.addVars(((i, t) for i in range(n_scenarios) for t in range(T)), vtype = GRB.BINARY)
phi = model.addVars((i, t) for i in range(n_scenarios) for t in range(T))



da_com = model.addVars(((t) for t in range(T)), ub = 10000)

model.addConstrs(da_ev[i,t] + da_bss[i,t] + da_rt[i,t] == da_com[t] for i in range(n_scenarios) for t in range(T))

model.addConstrs(da_ev[i,t] >= 0.8 *r_ev_demand[i,t] for i in range(n_scenarios) for t in range(T))

model.addConstrs(da_ev[i,t] + bss_ev[i,t] + rt_ev[i,t] == r_ev_demand[i,t] for i in range(n_scenarios) for t in range(T))

model.addConstrs(x[i,t] + y[i,t] <= 1 for i in range(n_scenarios) for t in  range(T))

model.addConstrs(da_bss[i,t] <= M_ch * x[i,t] for i in range(n_scenarios) for t in range(T))

model.addConstrs(da_bss[i,t] >= m_ch * x[i,t] for i in range(n_scenarios) for t in range(T))

model.addConstrs(bss_ev[i,t] >= m_dch * y[i,t] for i in range(n_scenarios) for t in range(T))

model.addConstrs(bss_ev[i,t] <= M_dch * y[i,t] for i in range(n_scenarios) for t in range(T))

model.addConstrs(phi[i,0] == 700 - bss_ev[i,0] + da_bss[i,0] for i in range(n_scenarios))

model.addConstrs(phi[i,t] == phi[i, t-1] - bss_ev[i,t] + da_bss[i,t] for i in range(n_scenarios) for t in range(1, T))

model.addConstrs(phi[i,t] >= m_phi for i in range(n_scenarios) for t in range(T))

model.addConstrs(phi[i,t] <= M_phi for i in range(n_scenarios) for t in range(T))

model.setObjective(gp.quicksum(0.01 * (probabilities[i]*((min(r_da[i,t], r_rt[i,t]) - r_da[i,t]) * da_ev[i,t] +\
                                             min(r_da[i,t], r_rt[i,t]) * bss_ev[i,t] +\
                                             (min(r_da[i,t], r_rt[i,t]) - r_rt[i,t]) * rt_ev[i,t] +\
                                             (min(r_da[i,t], r_rt[i,t]) - r_da[i,t]) * da_rt[i,t] - \
                                             r_da[i,t] * da_bss[i,t]))\
                               for i in range(n_scenarios) for t in range(T)), GRB.MAXIMIZE)
    
    
model.optimize()



da_com_list = []

for t in range(24):
    da_com_list.append(da_com[t].x)
    

reduced_da_com = da_com_list
df_da_reduced = pd.DataFrame(reduced_da_com)


df_da_reduced.to_csv('df_da_commitments_reduced',index=False)