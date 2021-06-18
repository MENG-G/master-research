import math
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

'''
'basis_weight': mgsm, [g/m^2]
'weight': sample weight, [g]
'surface': sample surface, [m^2]
'thickness': thickness of coated layer, data from NISHIMURA san, [m]
'sigma_v': culumative pores volume of coated layer, experimental data, [cm^3 / g]
'surface': surface area of sample, [m^2]
'coated_layer_v': volume of single coated layer, [m^3]
'''
paper_dict = {
    'LAG90': {
        'basis_weight': 90,
        'weight': 0.484,
        'thickness': 6.9e-6,
        'sigma_v': 3.94e-2,
        'surface': 0.010755555555555555,
        'coated_layer_v': 7.421333333333333e-08,
        'porosity': 0.25695652173913036,
    },
    'LAG130': {
        'basis_weight': 130,
        'weight': 0.452,
        'thickness': 1.4e-5,
        'sigma_v': 3.74e-2,
        'surface': 0.006953846153846154,
        'coated_layer_v': 9.735384615384615e-08,
        'porosity': 0.17364285714285718,
    },
    'LAG200': {
        'basis_weight': 200,
        'weight': 0.454,
        'thickness': 1.6e-5,
        'sigma_v': 3.63e-2,
        'surface': 0.00454,
        'coated_layer_v': 7.264e-08,
        'porosity': 0.22687499999999997,
    },
    'LAG250': {
        'basis_weight': 250,
        'weight': 0.474,
        'thickness': 2.14e-5,
        'sigma_v': 2.56e-2,
        'surface': 0.003792,
        'coated_layer_v': 8.114879999999998e-08,
        'porosity': 0.14953271028037385,
    },
    'OKT73.3': {
        'basis_weight': 73.3,
        'weight': 0.468,
        'thickness': 7.5e-6,
        'sigma_v': 4.10e-2,
        'surface': 0.012769440654843112,
        'coated_layer_v': 9.577080491132334e-08,
        'porosity': 0.20035333333333336,
    },
    'OKT84.9': {
        'basis_weight': 84.9,
        'weight': 0.492,
        'thickness': 7.5e-6,
        'sigma_v': 3.68e-2,
        'surface': 0.011590106007067136,
        'coated_layer_v': 8.692579505300353e-08,
        'porosity': 0.208288,
    },
    'OKT104.7': {
        'basis_weight': 104.7,
        'weight': 0.481,
        'thickness': 1.2e-5,
        'sigma_v': 3.48e-2,
        'surface': 0.00918815663801337,
        'coated_layer_v': 1.1025787965616046e-07,
        'porosity': 0.15181499999999998,
    },
    'OKT128': {
        'basis_weight': 127.9,
        'weight': 0.478,
        'thickness': 1.3e-5,
        'sigma_v': 3.18e-2,
        'surface': 0.007474589523064894,
        'coated_layer_v': 9.716966379984361e-08,
        'porosity': 0.15643153846153848,
    },
    'SWORD1': {
    },
    'SWORD2': {
    },
    'NEXT_ij1': {
    },
    'NEXT_ij2': {
    },
}


def pores_distribution(x, y, paper, num):
    # logD
    log10x = np.log10(x)

    # interpolate
    log10x_new = np.linspace(min(log10x), max(log10x), num=num)
    f = interp1d(log10x, y, kind='linear')
    y_new = f(log10x_new)
    x_new = np.power(10, log10x_new)
    
    # mean, std
    mean = np.sum(np.multiply(log10x_new, y_new)) / np.sum(y_new)
    std = np.sqrt(np.sum(np.multiply(np.power(log10x_new - mean, 2), y_new) / np.sum(y_new)))

    # delta logD
    delt_log10x_new = [log10x_new[i+1] - log10x_new[i] for i in range(len(log10x_new) -1)]
    delt_log10x_new.append(delt_log10x_new[-1] * delt_log10x_new[-1] / delt_log10x_new[-2])

    # dV / dlogD
    exp_item = np.exp(-np.power(log10x_new - mean, 2) / (2 * std ** 2))
    A = (1 / (np.sqrt(2 * np.pi) * std)) * np.sum(np.multiply(y_new, delt_log10x_new))
    dVdlogD = A * exp_item
    
    # V(D)
    VD = dVdlogD * delt_log10x_new

    # porosity
    porosity = (np.sum(VD) * 1e-6  * paper_dict[paper]['weight'] / paper_dict[paper]['coated_layer_v'])
    
    # filling rate
    filling_rate = 1 -  porosity

    # density of CaCO3: 2.71e6 [g/m^3]
    rho = 2.71e6 * filling_rate
    
    # pores distribution
    n = rho * VD / np.pi / ((x_new / 2) ** 2) * 1e12 / 2
    
    return x_new, dVdlogD, VD, n, porosity


def pores_distribution_test(x, y, paper, num):
    # logD
    log10x = np.log10(x)

    # interpolate
    log10x_new = np.linspace(min(log10x), max(log10x), num=num)
    f = interp1d(log10x, y, kind='linear')
    y_new = f(log10x_new)
    x_new = np.power(10, log10x_new)
    
    # mean, std
    mean = np.sum(np.multiply(log10x_new, y_new)) / np.sum(y_new)
    std = np.sqrt(np.sum(np.multiply(np.power(log10x_new - mean, 2), y_new) / np.sum(y_new)))

    # delta logD
    delt_log10x_new = [log10x_new[i+1] - log10x_new[i] for i in range(len(log10x_new) -1)]
    delt_log10x_new.append(delt_log10x_new[-1] * delt_log10x_new[-1] / delt_log10x_new[-2])

    # dV / dlogD
    exp_item = np.exp(-np.power(log10x_new - mean, 2) / (2 * std ** 2))
    A = (1 / (np.sqrt(2 * np.pi) * std)) * np.sum(np.multiply(y_new, delt_log10x_new))
    dVdlogD = A * exp_item
    
    # V(D)
    VD = dVdlogD * delt_log10x_new
    
    # pores distribution
    n = VD * 0.5 * paper_dict[paper]['weight'] / (paper_dict[paper]['surface'] * paper_dict[paper]['thickness']
        * (x_new / 2) ** 2 * np.pi) * 1e12

    
    return x_new, dVdlogD, VD, n 



def gaussian_fitting(x, y, num):
    # logD
    log10x = np.log10(x)

    # interpolate
    log10x_new = np.linspace(min(log10x), max(log10x), num=num)
    f = interp1d(log10x, y, kind='linear')
    y_new = f(log10x_new)
    x_new = np.power(10, log10x_new)
    
    # mean, std
    mean = np.sum(np.multiply(log10x_new, y_new)) / np.sum(y_new)
    std = np.sqrt(np.sum(np.multiply(np.power(log10x_new - mean, 2), y_new) / np.sum(y_new)))

    # delta logD
    delt_log10x_new = [log10x_new[i+1] - log10x_new[i] for i in range(len(log10x_new) -1)]
    delt_log10x_new.append(delt_log10x_new[-1] * delt_log10x_new[-1] / delt_log10x_new[-2])

    # dV / dlogD
    coe = np.sum(np.multiply(y_new, delt_log10x_new))
#     exp_item = np.exp(-np.power(log10x_new - mean, 2) / (2 * std ** 2))
#     A = (1 / (np.sqrt(2 * np.pi) * std)) 
    
    return mean, std, coe



if __name__ == '__main__':
    
    for paper in paper_dict.keys():
        # print(paper_dict)
        if paper_dict[paper]:
            paper_dict[paper]['surface'] = paper_dict[paper]['weight'] / paper_dict[paper]['basis_weight'] * 2
            paper_dict[paper]['coated_layer_v'] = paper_dict[paper]['surface'] * paper_dict[paper]['thickness']
            paper_dict[paper]['porosity'] = paper_dict[paper]['sigma_v'] * 1e-6 * paper_dict[paper]['weight'] / paper_dict[paper]['coated_layer_v']
    print(paper_dict)

