from cProfile import label
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


# Before Runing!!
# 1. config, 2. porocess_dir, 3. exp_condition_csv


paper_label_lst = ["l130_pig3wt", "l130_pig6wt", "l130_pure_water",
                "o104_pig3wt", "o104_pig6wt", "o104_pure_water"]
process_dir = "C:/Users/kakum/Desktop/02_Research/00_Experimental Data/20210704_pigment/output_results"
delta_T = 0.2 # s
rho_H2O = 997
rho_PG = 1036
alpha = 0.763
PG_wt_frac = 0 
T = 26.423 
T_K = T + 273.15
RH = 74.06


def reduce_noise():
    for paper_label in paper_label_lst:
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
    for paper_label in paper_label_lst:
        paper_csvpath = f"{process_dir}/{paper_label}_mean.csv"
        if not os.path.exists(paper_csvpath):
            continue
        
        result_path = f"{process_dir}/{paper_label}_mean_evap.csv"
        hu_and_larson(input_path=paper_csvpath, output_path=result_path, PG_wt_frac=PG_wt_frac, T=T, RH=RH, delta_T=delta_T)


def evap_vol_cal():
    result_dict = {}
    for paper_label in paper_label_lst:
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
    result_dict = {}
    for paper_label in paper_label_lst:
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
    result_dict = {}
    for paper_label in paper_label_lst:

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
            df["pentrated_vol_ratio"] = result_dict[paper_label]
            df.to_csv(f"{process_dir}/{paper_label}_pen_vol_ratio.csv", index=False) # 渗透的ratio



def plot_all():
    data_dict = {'l130': ["l130_pig3wt", "l130_pig6wt", "l130_pure_water"],
                'o104':["o104_pig3wt", "o104_pig6wt", "o104_pure_water"]}
    vol_dict_ratio = {}
    pen_dict_ratio = {}
    eva_dict_ratio = {}
    t = {}
    for paper in ['l130', 'o104']:
        for paper_label in data_dict[paper]:
            fig, ax = plt.subplots(figsize=(5, 4.3))
            font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}  

            ori_file = f"{process_dir}/{paper_label}_mean.csv"
            pen_ratio_file = f"{process_dir}/{paper_label}_pen_vol_ratio.csv"
            eva_vol_file = f"{process_dir}/{paper_label}_eva_vol.csv"

            vol_dict_ratio[paper_label] = []
            pen_dict_ratio[paper_label] = []
            eva_dict_ratio[paper_label] = []
            t[paper_label] = []

            ori_data = pd.read_csv(ori_file)
            ori_vol = ori_data["Mj[pL]"]   

            pen_ratio_data = pd.read_csv(pen_ratio_file)
            pen_vol_ratio = pen_ratio_data["pentrated_vol_ratio"]

            eva_vol_data = pd.read_csv(eva_vol_file)
            eva_vol = eva_vol_data["evaporated_vol[pL]"]

            for j in range(len(ori_vol)):
                vol_dict_ratio[paper_label].append(ori_vol[j] / ori_vol[0])
                pen_dict_ratio[paper_label].append(pen_vol_ratio[j])
                eva_dict_ratio[paper_label].append(eva_vol[j] / ori_vol[0])
                t[paper_label].append(j / len(ori_vol))

            interval = 50
            ax.plot([t[paper_label][i] for i in range(len(t[paper_label])) if i%interval==0], 
                    [vol_dict_ratio[paper_label][i] for i in range(len(t[paper_label])) if i%interval==0], 
                    "^:", markersize=8,
                    label = "Droplete Volume Ratio")
            
            ax.plot([t[paper_label][i] for i in range(len(t[paper_label])) if i%interval==0], 
                    [eva_dict_ratio[paper_label][i] for i in range(len(eva_dict_ratio[paper_label])) if i%interval==0], 
                    "o:", markersize=8,
                    label = "Evaporation Ratio")

            ax.plot([t[paper_label][i] for i in range(len(t[paper_label])) if i%interval==0], 
                    [pen_dict_ratio[paper_label][i] for i in range(len(pen_dict_ratio[paper_label])) if i%interval==0], 
                    "s:", markersize=8,
                    label = "Permeation Ratio") 
    
            legend_labels = ax.legend(loc="best", frameon=False).get_texts()
            [label.set_fontname('Times New Roman') for label in legend_labels]
            # 设置xy轴标签
            ax.set_xlabel("$\mathrm{T/T_d}$ [-]", font)
            ax.set_ylabel("Volume Ratio [-]", font)
            plt.tick_params(labelsize=12)
            
            tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in tick_labels]        
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            plt.savefig(f"{paper_label}.png", dpi=500)
            # plt.show()


def plot_distance():
    data_dict = {'l130': ["l130_pure_water", "l130_pig3wt", "l130_pig6wt", ],
                'o104':["o104_pure_water", "o104_pig3wt", "o104_pig6wt", ]}

    fmt = {"l130_pig3wt": "o", "l130_pig6wt": "s", "l130_pure_water": "^",
        "o104_pig3wt": "o", "o104_pig6wt": "s", "o104_pure_water": "^"}

    label_dict = {"l130_pig3wt": "Pigment Ink 3 wt%", "l130_pig6wt": "Pigment Ink 6 wt%", "l130_pure_water": "Pure Water",
        "o104_pig3wt": "Pigment Ink 3 wt%", "o104_pig6wt": "Pigment Ink 6 wt%", "o104_pure_water": "Pure Water"}

    pen_dict = {}
    t = {}
    for paper in ['l130', 'o104']:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}          
        interval = 50

        for paper_label in data_dict[paper]:
            ori_file = f"{process_dir}/{paper_label}_mean.csv"
            pen_file = f"{process_dir}/{paper_label}_pen_vol.csv"
            
            pen_dict[paper_label] = []
            t[paper_label] = []

            ori_data = pd.read_csv(ori_file)
            init_r = ori_data["bottom_radius[um]"][0]
            init_s = (init_r * 1e-6) ** 2 * math.pi # m^2

            pen_data = pd.read_csv(pen_file)
            penetrated_vol_per_area = pen_data["pentrated_vol[pL]"] * 1e-15 / init_s * 1e6 
            
            ax.plot(np.sqrt([i*delta_T for i in range(len(penetrated_vol_per_area)) if i%interval==0]),
                [penetrated_vol_per_area[i] for i in range(len(penetrated_vol_per_area)) if i%interval==0],
                fmt[paper_label], markersize=8,
                label = label_dict[paper_label])

    
        legend_labels = ax.legend(loc="best", frameon=False).get_texts()
        [label.set_fontname('Times New Roman') for label in legend_labels]
        # 设置xy轴标签
        ax.set_xlabel("Time [$\mathrm{\sqrt{s}}$]", font)
        ax.set_ylabel("Avg. Permeation Distance [$\mathrm{\mu m}$]", font)
        plt.tick_params(labelsize=12)
        
        tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in tick_labels]        
        ax.set_xlim(0, 16)
        ax.set_ylim(0, )

        plt.savefig(f"{paper}.png", dpi=500)
        # plt.show()






if __name__=='__main__':
    

    # reduce_noise()
    # evap_rate_cal()
    # evap_vol_cal()
    # pene_cal()
    # pene_ratio_cal()




    # plot_all()

    plot_distance()


    # plot_permeation_per_area()

    
