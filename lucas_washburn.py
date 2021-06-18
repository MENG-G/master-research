import math
import numpy as np

class Constant:
    sigma_H2O = 72.73614e-3 # 20℃, [N/m]
    mu_H2O = 1.005e3 # 20℃, [Pa·s]


def locus_washburn(radius, sigma, theta, viscosity, time):
    '''
    input:
        sigma: [N/m]
        theta: contact angle in capillary, [°]
        viscosity: [Pa·s]
        time: [s]
    outoput:
        z: penetration distance in one capillary, [m]
    '''
    z = math.sqrt(radius * sigma * math.cos(math.radians(theta)) / (2 * viscosity)) * math.sqrt(time)
    return z


def calculate_sigma_H2O(T):
    '''
    input:
        T: [K]
    output:
        sigma: surface tension of water, [mN/m]
    '''
    T_C = 647.096 # [K]
    sigma = 235.8 * (1 - T / T_C) ** (1.256) * (1 - 0.625 * (1 - T / T_C))
    return sigma


def dynamic_conatact_angle(Ca, theta_s):
    '''
    input:
        theta_s: static contact angle, [°]
        Ca: capillary number, u * mu / sigma, 
    output:
        theta_d
    '''
    theta_s = math.radians(theta_s)
    theta_d = math.degrees(math.acos(hoff_func(Ca + math.radians(math.cos((hoff_func(theta_s)))))))
    return theta_d


def hoff_func(x):
    f = 1 - 2 * math.tanh(5.16 * (x / (1 + 1.31 * x ** (0.99))) ** 0.706)
    return f


def get_mixture_viscosity(mole_frac_PG, ita_H2O, ita_PG, T):
    '''
    Jouyban–Acree model
    input:
        mole_frac_PG: mole fraction of PG 
        T: [K]
        ita_H2O: viscosity of H2O, [Pa s]
        ita_PG: viscosity of PG, [Pa s]
    output: 
        ita_mixture: [Pa s]
    '''
    ita_mixture = math.exp(mole_frac_PG * math.log(ita_PG) + 
                (1 - mole_frac_PG) * math.log(ita_H2O) + 
                926.206 * (mole_frac_PG * (1 - mole_frac_PG) / T) -
                606.410 * (mole_frac_PG * (1 - mole_frac_PG) * (mole_frac_PG - (1 - mole_frac_PG)) / T))
    return ita_mixture


def get_mixture_sigma(mole_frac_PG, sigma_H2O, sigma_PG, T):
    '''
    Jouyban–Acree model
    input:
        mole_frac_PG: mole fraction of PG 
        T: [K]
        sigma_H2O: [N/m]
        sigma_PG: [N/m]
    output:
        sigma_mixture: [N/m]
    '''
    sigma_mixture = math.exp(mole_frac_PG * math.log(sigma_PG) + 
                (1 - mole_frac_PG) * math.log(sigma_H2O) - 
                183.307 * (mole_frac_PG * (1 - mole_frac_PG) / T) +
                197.808 * (mole_frac_PG * (1 - mole_frac_PG) * (mole_frac_PG - (1 - mole_frac_PG)) / T) - 
                456.916 * (mole_frac_PG * (1 - mole_frac_PG) * (mole_frac_PG - (1 - mole_frac_PG)) ** 2 / T))
    return sigma_mixture


def locus_washburn_Liu(radius, sigma, theta, viscosity, time, permeability, porosity):
    '''
    input:
        sigma: [N/m]
        theta: contact angle in capillary, [°]
        viscosity: [Pa·s]
        time: [s]
    outoput:
        z: penetration distance in one capillary, [m]
    '''
    z = math.sqrt(4 * sigma * math.cos(math.radians(theta)) * permeability / (viscosity * porosity * radius)) * math.sqrt(time)
    return z


def locus_washburn_Darcy(radius, sigma, theta, viscosity, time, permeability):
    '''
    input:
        sigma: [N/m]
        theta: contact angle in capillary, [°]
        viscosity: [Pa·s]
        time: [s]
    outoput:
        z: penetration distance in one capillary, [m]
    '''
    z = 2 * math.sqrt(radius * sigma * math.cos(math.radians(theta)) * permeability / (viscosity * (radius ** 2 + 8 * permeability))) * math.sqrt(time)
    return z



if __name__ == '__main__':
    print(get_mixture_viscosity(mole_frac_PG=0.362, ita_H2O=0.721, ita_PG=12.844, T=318))
    print(get_mixture_sigma(mole_frac_PG=0.686, sigma_H2O=62.67, sigma_PG=43.07, T=318))