
import numpy as np

class CompetitiveHubs():

  def __init__(self):
    self.retail_price2 = None
    self.retail_price3 = None

  def update_retail_price(self, da_price, rt_price):
    self.retail_price2 = np.random.uniform(1.25 *  min(da_price, rt_price), 1.50 *  min(da_price, rt_price))
    self.retail_price3 = np.random.uniform(1.25 *  min(da_price, rt_price), 1.50 *  min(da_price, rt_price))


