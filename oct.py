import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import cv2
import os

from sympy import continued_fraction_reduce


# papers = ['l90', 'l130', 'l200', 'l250', 'o73', 'o84', 'o104', 'o128']
papers = ['l130','o104']
solution = 'pure_water'
process_dir = "C:/Users/kakum/Desktop/02_Research/00_Experimental Data/20210603_purewater_all/"

# solution = 'pg25'
# process_dir = "C:/Users/kakum/Desktop/02_Research/00_Experimental Data/20210629_pg25_all/"

# solution = 'pg50'
# process_dir = "C:/Users/kakum/Desktop/02_Research/00_Experimental Data/20210701_pg50_all/"

threshold = 60
pix_len = 2.816642011726414 #um

paper_label_dict = {'l90':'A1', 'l130':'A2', 'l200':'A3', 'l250':'A4', 
        'o73':'B1', 'o84':'B2', 'o104':'B3', 'o128':'B4',}
markers = [">", "o", "^", "s"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def plot():
    for paper in papers:
        fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(5, 5))
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}

        for i in range(1, 4):

            paper_label = f"{paper_label_dict[paper]}_{i}"
            data_path = f"{process_dir}/{paper}/{i}"
            if not os.path.exists(data_path):
                continue
            csv_all = sorted(glob.glob(f"{data_path}/image*.csv"))
            # print(re.split("[_.]", csv_all[0])[-2])
            # start = int(re.split("[_.]", csv_all[0])[-2])
            # end = int(re.split("[_.]", csv_all[-1])[-2])
            # csv_range = [i for i in range(start, end+1)]
            # csv_list = [f"{data_path}/image_{:0>5d}.csv".format(i) for i in csv_range]

            for j, csv in enumerate(csv_all):
                df = pd.read_csv(csv, header=None)
                df = df.iloc[::-1] # reverse rows
                if j == 0:
                    recorder = df
                else:
                    recorder = pd.concat([recorder, df], axis=1, ignore_index=False)        

            arr = np.array(recorder)
            img = cv2.GaussianBlur(arr, (5,5), 0)
            img[img>=threshold] = 255
            img[img<threshold] = 0
            
            time_data = pd.read_csv(f"{data_path}/time.csv", header=None)
            delta_t = time_data[0][len(time_data)-1] / len(time_data)
            
            res=[]
            for k in range(img.shape[1]):
                highlights = np.where(img[:,k]==255)[0]
                if len(highlights) == 0:
                    res.append(0)
                else:
                    res.append(250 - min(highlights))

            ax[i-1].plot([i*delta_t for i in range(len(res))], 
                (np.array(res) - res[0])*pix_len,
                markers[i], markersize=2,
                color= colors[i],
                label=paper_label)

            legend_labels = ax[i-1].legend(loc="best", frameon=False).get_texts()
            [label.set_fontname('Times New Roman') for label in legend_labels]
            
            tick_labels = ax[i-1].get_xticklabels() + ax[i-1].get_yticklabels()
            [label.set_fontname('Times New Roman') for label in tick_labels]        
            ax[i-1].set_xlim(0, )
            ax[i-1].set_ylim(-75, 25)
        
        fig.text(0.5, 0.008, "Time [$\mathrm{s}$]", ha='center', font=font, fontsize=12)
        fig.text(0, 0.5, "Distance Change [$\mathrm{\mu m}$]", va='center', rotation='vertical', font=font, fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{solution}_{paper}_oct.png", dpi=500)



if __name__ == '__main__':
    plot()