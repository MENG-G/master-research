"""
Constant: 
    PG_wt_frac: PG weight fraction, e.g. 0.25
    T: Environment Temperature ℃ 
    RH: Relative Humidity %RH
    delta_T: Iteration Interval sec
    theta: contact angle °
    K: Permeability [m^2]
    sigma: surface tension
    alpha: evaporation correction factor
    rho_H2O: 997 [kg/m^3]
    rho_PG: 1036 [kg/m^3]
    R: contact radius [m]
    S: contact area [m^2]
    Time: time span [s]
"""

delta_T = 16.67e-3 
rho_H2O = 997
rho_PG = 1036
alpha = 0.763



# ===========  pure water 0311 ========== #
# PG_wt_frac = 0 
# T = 22.282 
# T_K = T + 273.15
# RH = 28.182 


# ===========  pure water ========== #
PG_wt_frac = 0 
T = 23.600
T_K = T + 273.15
RH = 33.780


# ===========  PG25 ========== #
# PG_wt_frac = 0.25
# T = 21.514
# T_K = T + 273.15
# RH = 19.892


# ===========  PG50 ========== #
# PG_wt_frac = 0.5
# T = 23.972
# T_K = T + 273.15
# RH = 15.407


