

import numpy as np
import pandas as pd 



# Mean number of EVs on the road for each hour of the day, based on the historical data collected by FDOT
#https://www.fdot.gov/statistics/trafficinfo/default.shtm
mean_evs_in_road = [1400, 1400, 1400, 1400, 1400, 2212, 5276, 6292, 5750, 5242, 5364, 5324, 5734, 5720, 6116,
                    6856, 6750, 6584, 4908, 3522, 2768, 2372, 1978, 1428]


# Set random seeds for reproducibility
np.random.seed(123)

# Generate the Poisson distribution of EVs on the road for each hour of 365 days
evs_in_each_hr = np.random.poisson(mean_evs_in_road, (365, 24))

# Probability of an EV seeking a charge for each hour of the day
probab_seeking_charge = [0.002086889, 0.001561998, 0.00125, 0.001352776, 0.001980443, 
                         0.002714556, 0.004072666, 0.007941443, 0.012544334, 0.020388334, 
                         0.020072666, 0.020597557, 0.03325, 0.027182553, 0.022792556, 
                         0.025303223, 0.029274776, 0.030111666, 0.031366999, 0.022792556, 
                         0.015785444, 0.012228665, 0.009094001, 0.004806779]


# Simulate the number of EVs seeking a charge for each hour of each day
evs_seeking_charge = np.random.binomial(evs_in_each_hr, probab_seeking_charge)
evs_seeking_charge = evs_seeking_charge.tolist()

df_evs_seeking_charge = pd.DataFrame(evs_seeking_charge, columns=[f'hour_{i+1}' for i in range(24)])
df_evs_seeking_charge.to_csv('evs_seeking_charge.csv', index=False)



# Read the CSV files
df_da = pd.read_csv("da_hrl_lmps_48592.csv")
df_rt = pd.read_csv("rt_hrl_lmps_48592.csv")

# Drop duplicates and reset index
df_rt = df_rt.drop_duplicates(subset='datetime_beginning_ept', keep="first").reset_index()

day_ahead_price = df_da["total_lmp_da"].values.reshape(-1, 24)
real_time_price = df_rt["total_lmp_rt"].values.reshape(-1, 24)

price_profile = np.concatenate((day_ahead_price, real_time_price), axis=1)
elec_price = pd.DataFrame(price_profile, columns=["X"+str(i) for i in range(1, 49)])

# Split the DataFrame into day ahead and real-time DataFrames
df_da_prices = elec_price.iloc[:, :24]
df_rt_prices = elec_price.iloc[:, 24:]

df_da_prices.to_csv("df_da_prices.csv")
df_rt_prices.to_csv("df_rt_prices.csv")
    


