
import gurobipy as gp 
from gurobipy import GRB

class OptimizePowerSourcesPricing():
  def __init__(self, M_bss, m_bss, M_bss_ch_rate, M_bss_dch_rate):
      self.M_bss = M_bss
      self.m_bss = m_bss
      self.M_bss_ch_rate = M_bss_ch_rate
      self.M_bss_dch_rate = M_bss_dch_rate
  
  def power_management(self, agg_ev_demand, da_com, p_ev, p_da, p_rt, bss_power, p_bss):
    model = gp.Model()
    bss_ch = model.addVar()
    bss_dch = model.addVar()
    da_ev = model.addVar()
    da_bss = model.addVar()
    da_rt = model.addVar()
    bss_ev = model.addVar()
    rt_ev = model.addVar() 
    x = model.addVar(vtype = GRB.INTEGER)
    y = model.addVar(vtype = GRB.INTEGER)
    
    if da_com >= agg_ev_demand:
        model.addConstr(da_ev == agg_ev_demand)
    else:
        model.addConstr(da_ev == da_com)

    model.addConstr(da_ev + bss_ev + rt_ev == agg_ev_demand)
    model.addConstr(da_ev + da_bss + da_rt == da_com)
    model.addConstr(da_bss == bss_ch)
    model.addConstr(bss_ev  == bss_dch)
    model.addConstr(bss_ch <= min(self.M_bss_ch_rate, self.M_bss - bss_power)*x)
    model.addConstr(bss_dch <=  min(self.M_bss_dch_rate, bss_power - self.m_bss)*y)
    model.addConstr(x + y <= 1)

    model.setObjective(0.01 * (\
                     (p_ev - p_da) * da_ev +\
                     (p_ev - p_bss) * bss_ev +\
                     (p_ev - p_rt) * rt_ev +\
                     (-0*p_da)* da_bss +\
                     (min(p_rt, p_da) - p_da) * da_rt)\
                     , GRB.MAXIMIZE)

    model.setParam('OutputFlag', 0)

    model.optimize()

    return da_ev.x, bss_ev.x, rt_ev.x, da_bss.x, da_rt.x, model.obj_val

