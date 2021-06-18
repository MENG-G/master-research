import os
import sys
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import re

from py import process
from calculate_evaporate import hu_and_larson, get_mixture_density, wt2mole_frac
import math
from lucas_washburn import dynamic_conatact_angle, locus_washburn, locus_washburn_Liu, \
    locus_washburn_Darcy, get_mixture_viscosity, get_mixture_sigma
from configs_30pL import PG_wt_frac, T, T_K, RH, delta_T, rho_H2O, rho_PG, alpha


# Before Runing!!
# 1. config, 2. porocess_dir, 3. exp_condition_csv


papers = ['l90', 'l130', 'l200', 'l250', 'o73', 'o84', 'o104', 'o128']
# papers = ['l130']

# --------- pure water
# solution = 'pure_water'
# exp_condition_csv = "purewater_30.csv"
# process_dir = "C:/Users/kakum/Desktop/02_Research/00_Experimental Data/20210127_purewater/output_results" 
# process_dir = "C:/Users/kakum/Desktop/02_Research/00_Experimental Data/20210311_purewater_l130/output_results"

# --------- pg25
# solution = 'pg25'
# exp_condition_csv = "pg25_30.csv"
# process_dir = "C:/Users/kakum/Desktop/02_Research/00_Experimental Data/20210205_pg25%/output_results"

# --------- pg50
solution = 'pg50'
exp_condition_csv = "pg50_30.csv"
process_dir = "C:/Users/kakum/Desktop/02_Research/00_Experimental Data/20210208_pg50%/output_results/"



def theoretical_cal():    
    paper_tras = {'l90': 'LAG90', 'l130': 'LAG130', 'l200': 'LAG200', 'l250': 'LAG250',
                'o73': 'OKT73.3', 'o84': 'OKT84.9', 'o104': 'OKT104.7', 'o128': 'OKT128'}    
    
    # loop theoretical calculation
    for paper in papers:
        info = pd.read_csv(exp_condition_csv)
        theta = info[paper][2] # contact angle
        R = info[paper][4] * 1e-6 # contact radius
        S = R ** 2 * math.pi # contact area
        K = info[paper][7] # permeability
        porosity = info[paper][6] # porosity
        Time = 1 # time    
    
        mole_frac_PG = wt2mole_frac(wt_frac_PG = PG_wt_frac)
        mu = get_mixture_viscosity(mole_frac_PG=mole_frac_PG, ita_H2O=1.00e-3, ita_PG=5.76e-2, T=T_K)
        sigma = get_mixture_sigma(mole_frac_PG=mole_frac_PG, sigma_H2O=7.29e-2, sigma_PG=3.86e-2, T=T_K)

        pores_df = pd.read_csv(\
            r'C:/Users/kakum/Desktop/02_Research/00_Experimental Data/20200731_Gaussian Fitting/python/processed_data_100/' \
            + paper_tras[paper] + '.csv')
        
        pen_v = []
        for t in np.arange(0, Time, 0.05):
            tem_v = []
            for i in range(len(pores_df['n'])-1):
                r = pores_df['d'][i] / 2 * 1e-9 # [m]
                n = pores_df['n'][i] 
                # z = locus_washburn_Darcy(radius=r, sigma=sigma, theta=theta, viscosity=mu, time=t, permeability=K) 
                z = locus_washburn_Liu(radius=r, sigma=sigma, theta=theta, \
                    viscosity=mu, time=t, permeability=K, porosity=porosity)
                tem_v.append(S * n * r ** 2 * math.pi * z * 1e15) 
            # print(sum(tem_v))
            pen_v.append(sum(tem_v))
            
            df = pd.DataFrame()
            df["th_pen_vol"] = pen_v
            df.to_csv(f"{process_dir}/{paper}_pen_vol_th.csv", index=False)


def reduce_noise():
    for paper in papers:
        for i in range(1, 8):
            paper_label = f"{paper}_{i}"
            paper_csvpath = f"{process_dir}/{paper_label}.csv"
            if not os.path.exists(paper_csvpath):
                continue

            mean_dict = {}
            for s in ["time[ms]", "bottom_radius[um]", "theta[degree]", "height[um]", "area[m^2]", "Mj[pL]"]:
                mean_dict[s] = []
            
            df = pd.read_csv(paper_csvpath)
            mean_dict["time[ms]"] = df["time[ms]"]

            for j in range(len(df)):
                for s in ["bottom_radius[um]", "theta[degree]", "height[um]", "area[m^2]", "Mj[pL]"]: 
                    if j == 0 or j == len(df)-1:
                        mean_dict[s].append(df[s][j])
                    else:
                        mean_dict[s].append(np.mean([df[s][j-1], df[s][j], df[s][j+1]]))        

            mean_df = pd.DataFrame(mean_dict)
            mean_df.to_csv(f"{process_dir}/{paper_label}_mean.csv", index=False)


def evap_rate_cal():
    for paper in papers:
        for i in range(1, 8):
            paper_label = f"{paper}_{i}"
            paper_csvpath = f"{process_dir}/{paper_label}_mean.csv"
            if not os.path.exists(paper_csvpath):
                continue
            
            result_path = f"{process_dir}/{paper_label}_mean_evap.csv"
            hu_and_larson(input_path=paper_csvpath, output_path=result_path, PG_wt_frac=PG_wt_frac, T=T, RH=RH, delta_T=delta_T)


def evap_vol_cal():
    for paper in papers:
        result_dict = {}
        for i in range(1, 8):
            paper_label = f"{paper}_{i}"
            result_dict[paper_label] = []

            eva_path = f"{process_dir}/{paper_label}_mean_evap.csv"
            ori_path = f"{process_dir}/{paper_label}_mean.csv"
            
            if not os.path.exists(eva_path):
                continue

            eva_data = pd.read_csv(eva_path)
            evaporation_rate = eva_data["J"]

            ori_data = pd.read_csv(ori_path)
            ori_area = ori_data["area[m^2]"]      

            for j in range(len(evaporation_rate)):
                if j == 0:
                    sigma_v = 0
                else:
                    evaporation_wt = evaporation_rate[j] * ori_area[j] * delta_T * alpha
                    mix_rho = get_mixture_density(wt2mole_frac(eva_data["wt_frac_PG"][j]), T_K)
                    evaporation_v = evaporation_wt / mix_rho * 1e15
                    sigma_v += evaporation_v
                
                result_dict[paper_label].append(sigma_v)  

            df = pd.DataFrame()
            df["evaporated_vol[pL]"] = result_dict[paper_label]
            df.to_csv(f"{process_dir}/{paper_label}_eva_vol.csv", index=False)                                     


def pene_cal():
    for paper in papers:
        result_dict = {}
        for i in range(1, 8):
            paper_label = f"{paper}_{i}"
            result_dict[paper_label] = []
            
            eva_path = f"{process_dir}/{paper_label}_eva_vol.csv"
            ori_path = f"{process_dir}/{paper_label}_mean.csv"
            if not os.path.exists(eva_path):
                continue

            eva_data = pd.read_csv(eva_path)
            evaporated_vol = eva_data["evaporated_vol[pL]"]
            
            ori_data = pd.read_csv(ori_path)
            ori_vol = ori_data["Mj[pL]"]

            for j in range(len(ori_vol)):
                result_dict[paper_label].append(ori_vol[0] - evaporated_vol[j] - ori_vol[j])

            df = pd.DataFrame()
            df["pentrated_vol[pL]"] = result_dict[paper_label]
            df.to_csv(f"{process_dir}/{paper_label}_pen_vol.csv", index=False) # 渗透的体积


def pene_ratio_cal():
    for paper in papers:
        result_dict = {}
        th_pen_file = f"{process_dir}/{paper}_pen_vol_th.csv"
        th_pen_data = pd.read_csv(th_pen_file)

        for i in range(1, 8):
            paper_label = f"{paper}_{i}"
            result_dict[paper_label] = []
               
            eva_path = f"{process_dir}/{paper_label}_eva_vol.csv"
            ori_path = f"{process_dir}/{paper_label}_mean.csv"

            if not os.path.exists(eva_path):
                continue

            eva_data = pd.read_csv(eva_path)
            evaporated_vol = eva_data["evaporated_vol[pL]"]      
            ori_data = pd.read_csv(ori_path)
            ori_vol = ori_data["Mj[pL]"]

            for j in range(len(ori_vol)):
                result_dict[paper_label].append((ori_vol[0] - evaporated_vol[j] - ori_vol[j])/ ori_vol[0])

            df = pd.DataFrame()
            df["pentrated_vol[pL]"] = result_dict[paper_label]
            df.to_csv(f"{process_dir}/{paper_label}_pen_vol_ratio.csv", index=False) # 渗透的ratio

            th_penetrated_vol = th_pen_data["th_pen_vol"]
            th_pen_data[f"{paper_label}"] = th_penetrated_vol / ori_vol[0]

        th_pen_data.to_csv(f"{process_dir}/{paper}_pen_vol_th.csv", index=False) # 转换为ratio的理论渗透值


def merge_all():
    vol_dict_ratio = {}
    eva_dict_ratio = {}
    pen_dict = {}
    pen_ratio_dict = {}
    th_pen_dict = {}

    for paper in papers:
        for i in range(1, 8):
            paper_label = f"{paper}_{i}"

            eva_file = f"{process_dir}/{paper_label}_eva_vol.csv"
            ori_file = f"{process_dir}/{paper_label}_mean.csv"
            pen_file = f"{process_dir}/{paper_label}_pen_vol.csv"
            th_pen_file = f"{process_dir}/{paper}_pen_vol_th.csv"
            if not os.path.exists(eva_file):
                continue

            vol_dict_ratio[paper_label] = []
            eva_dict_ratio[paper_label] = []
            pen_dict[paper_label] = []
            pen_ratio_dict[paper_label] = []
            th_pen_dict[paper_label] = []

            eva_data = pd.read_csv(eva_file)
            evaporated_vol = eva_data["evaporated_vol[pL]"]
            ori_data = pd.read_csv(ori_file)
            ori_vol = ori_data["Mj[pL]"]
            pen_data = pd.read_csv(pen_file)
            penetrated_vol = pen_data["pentrated_vol[pL]"]

            th_pen_data = pd.read_csv(th_pen_file)
            th_penetrated_vol = th_pen_data[f"{paper_label}"]            

            for j in range(len(ori_vol)):
                vol_dict_ratio[paper_label].append(ori_vol[j] / ori_vol[0])
                eva_dict_ratio[paper_label].append(evaporated_vol[j] / ori_vol[0])
                pen_ratio_dict[paper_label].append(penetrated_vol[j] / ori_vol[0])
                pen_dict[paper_label].append(penetrated_vol[j])
            th_pen_dict[paper_label] = th_penetrated_vol 

    all_vol_ratio = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in vol_dict_ratio.items()]))
    # all_vol["mean"] = all_vol.apply(lambda x:x.mean(),axis=1)
    # all_vol["std"] = all_vol.apply(lambda x:x.std(),axis=1)
    all_vol_ratio["Time"] = [i*delta_T for i in range(len(all_vol_ratio))]

    all_eva_ratio = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in eva_dict_ratio.items()]))
    all_pen_ratio = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in pen_ratio_dict.items()]))
    all_pen = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in pen_dict.items()]))
    all_th_pen_ratio = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in th_pen_dict.items()]))

    all_vol_ratio.to_csv("all_vol_ratio.csv", index=False)
    all_eva_ratio.to_csv("all_eva_ratio.csv", index=False)
    all_pen_ratio.to_csv("all_pen_ratio.csv", index=False)
    all_pen.to_csv("all_pen.csv", index=False) # 渗透体积量
    all_th_pen_ratio.to_csv("all_th_pen_ratio.csv", index=False) 



def plot_all(solution):
    interval_dict = {
        'pure_water': {'l90': 7, 'l130': 6, 'l200': 7, 'l250': 7, 'o73': 5, 'o84': 4, 'o104': 4, 'o128': 6},
        'pg25': {'l90': 5, 'l130': 5, 'l200': 7, 'l250': 7, 'o73': 4, 'o84': 4, 'o104': 4, 'o128': 4},
        'pg50': {'l90': 8, 'l130': 7, 'l200': 8, 'l250': 7, 'o73': 7, 'o84': 7, 'o104': 5, 'o128': 3}
    }
    maxx_dict = {'l90': 0.8, 'l130': 0.8, 'l200': 0.8, 'l250': 0.8, 'o73': 0.5, 'o84': 0.5, 'o104': 0.5, 'o128': 0.5}

    for paper in papers:
        vol_dict_ratio = {}
        eva_dict_ratio = {}
        pen_dict = {}
        pen_ratio_dict = {}
        th_pen_dict = {}
        for i in range(1, 8):
            paper_label = f"{paper}_{i}"

            eva_file = f"{process_dir}/{paper_label}_eva_vol.csv"
            ori_file = f"{process_dir}/{paper_label}_mean.csv"
            pen_file = f"{process_dir}/{paper_label}_pen_vol.csv"
            th_pen_file = f"{process_dir}/{paper}_pen_vol_th.csv"
            if not os.path.exists(eva_file):
                continue

            vol_dict_ratio[paper_label] = []
            eva_dict_ratio[paper_label] = []
            pen_dict[paper_label] = []
            pen_ratio_dict[paper_label] = []
            th_pen_dict[paper_label] = []

            eva_data = pd.read_csv(eva_file)
            evaporated_vol = eva_data["evaporated_vol[pL]"]
            ori_data = pd.read_csv(ori_file)
            ori_vol = ori_data["Mj[pL]"]
            pen_data = pd.read_csv(pen_file)
            penetrated_vol = pen_data["pentrated_vol[pL]"]

            th_pen_data = pd.read_csv(th_pen_file)
            th_penetrated_vol = th_pen_data[f"{paper_label}"]            

            for j in range(len(ori_vol)):
                vol_dict_ratio[paper_label].append(ori_vol[j] / ori_vol[0])
                eva_dict_ratio[paper_label].append(evaporated_vol[j] / ori_vol[0])
                pen_ratio_dict[paper_label].append(penetrated_vol[j] / ori_vol[0])
                pen_dict[paper_label].append(penetrated_vol[j])
            th_pen_dict[paper_label] = th_penetrated_vol 
        
        # all volume
        all_vol_ratio = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in vol_dict_ratio.items()]))
        all_vol_ratio["mean"] = all_vol_ratio.apply(lambda x:x.mean(),axis=1)
        all_vol_ratio["std"] = all_vol_ratio.apply(lambda x:x.std(),axis=1)
        all_vol_ratio["Time"] = [i*delta_T for i in range(len(all_vol_ratio))]
        # evap volume
        all_eva_ratio = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in eva_dict_ratio.items()]))
        all_eva_ratio["mean"] = all_eva_ratio.apply(lambda x:x.mean(),axis=1)
        all_eva_ratio["std"] = all_eva_ratio.apply(lambda x:x.std(),axis=1)
        # pen volume
        all_pen_ratio = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in pen_ratio_dict.items()]))
        all_pen_ratio["mean"] = all_pen_ratio.apply(lambda x:x.mean(),axis=1)
        all_pen_ratio["std"] = all_pen_ratio.apply(lambda x:x.std(),axis=1)
        # theoretical pen volume
        all_th_pen_ratio = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in th_pen_dict.items()]))
        all_th_pen_ratio["mean"] = all_th_pen_ratio.apply(lambda x:x.mean(),axis=1)
        all_th_pen_ratio["std"] = all_th_pen_ratio.apply(lambda x:x.std(),axis=1)
        
        
        fig, ax = plt.subplots(figsize=(5, 4.3))
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12} 
        
        interval = interval_dict[solution][paper]
        ax.errorbar([i*delta_T*interval for i in range(len(all_vol_ratio)//interval)], 
            [all_vol_ratio["mean"][i*interval] for i in range(len(all_vol_ratio)//interval)], 
            yerr=[all_vol_ratio["std"][i*interval] for i in range(len(all_vol_ratio)//interval)], 
            fmt= "^:", ms=8, label = "Exp. Volume", elinewidth=2, capsize=6)
        
        ax.errorbar([i*delta_T*interval for i in range(len(all_eva_ratio)//interval)], 
            [all_eva_ratio["mean"][i*interval] for i in range(len(all_eva_ratio)//interval)], 
            yerr=[all_eva_ratio["std"][i*interval] for i in range(len(all_eva_ratio)//interval)], 
            fmt= "o:", ms=8, label = "Evap. Volume", elinewidth=2, capsize=6)

        ax.errorbar([i*delta_T*interval for i in range(len(all_pen_ratio)//interval)], 
            [all_pen_ratio["mean"][i*interval] for i in range(len(all_pen_ratio)//interval)], 
            yerr=[all_pen_ratio["std"][i*interval] for i in range(len(all_pen_ratio)//interval)], 
            fmt= "s:", ms=8, label = "Pene. Volume", elinewidth=2, capsize=6)
        
        ax.plot([i*0.05 for i in range(len(th_penetrated_vol))], all_th_pen_ratio["mean"], 'forestgreen', label = "Theoretical Cal.", )
        # ax.plot([i*0.05 for i in range(len(th_penetrated_vol))], all_th_pen_ratio["mean"], 'k', label = "Theoretical Cal.", )
        # 设置legend的字体
        legend_labels = ax.legend(loc="best", frameon=False).get_texts()
        [label.set_fontname('Times New Roman') for label in legend_labels]
        # 设置xy轴标签
        ax.set_xlabel("Time [s]", font)
        ax.set_ylabel("Volume Ratio [-]", font)
        plt.tick_params(labelsize=12)
        
        tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in tick_labels]        
        ax.set_xlim(0, maxx_dict[paper])
        ax.set_ylim(0, 1)

        plt.savefig(f"{solution}_{paper}.png", dpi=500)
        # plt.show()
        # break


def plot_permeation_per_area():
    for paper in papers:
        vol_dict_ratio = {}
        eva_dict_ratio = {}
        pen_dict = {}
        pen_ratio_dict = {}
        th_pen_dict = {}
        for i in range(1, 8):
            paper_label = f"{paper}_{i}"

            eva_file = f"{process_dir}/{paper_label}_eva_vol.csv"
            ori_file = f"{process_dir}/{paper_label}_mean.csv"
            pen_file = f"{process_dir}/{paper_label}_pen_vol.csv"
            th_pen_file = f"{process_dir}/{paper}_pen_vol_th.csv"
            if not os.path.exists(eva_file):
                continue

            # vol_dict_ratio[paper_label] = []
            # eva_dict_ratio[paper_label] = []
            # pen_dict[paper_label] = []
            # pen_ratio_dict[paper_label] = []
            # th_pen_dict[paper_label] = []

            # eva_data = pd.read_csv(eva_file)
            # evaporated_vol = eva_data["evaporated_vol[pL]"]
            ori_data = pd.read_csv(ori_file)
            init_r = ori_data["bottom_radius[um]"][0]
            init_s = (init_r * 1e-6) ** 2 * math.pi # m^2
            
            pen_data = pd.read_csv(pen_file)
            penetrated_vol_per_area = pen_data["pentrated_vol[pL]"] * 1e-15 / init_s # m^3 / m^2
            
            print(penetrated_vol_per_area)

            break
            # th_pen_data = pd.read_csv(th_pen_file)
            # th_penetrated_vol = th_pen_data[f"{paper_label}"]            

            for j in range(len(ori_vol)):
                vol_dict_ratio[paper_label].append(ori_vol[j] / ori_vol[0])
                eva_dict_ratio[paper_label].append(evaporated_vol[j] / ori_vol[0])
                pen_ratio_dict[paper_label].append(penetrated_vol[j] / ori_vol[0])
                pen_dict[paper_label].append(penetrated_vol[j])
            th_pen_dict[paper_label] = th_penetrated_vol 
        break


if __name__=='__main__':
    
    # theoretical_cal()
    # print("Theoretical value calculated!")

    # reduce_noise()
    # print("Noise reduced!")

    # evap_rate_cal()
    # evap_vol_cal()
    # print("Evaporation calculated!")

    # pene_cal()
    # pene_ratio_cal()
    # print("Penetration calculated!")

    # merge_all()
    # print("All data saved!")

    # plot_all(solution)
    # print("All figures saved")

    plot_permeation_per_area()

    
