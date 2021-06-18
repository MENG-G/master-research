import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import cv2
import numpy as np


font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14} 
figsize = (5, 4.5)



def initial_droplet_volume():
    csv_lst = ["purewater_30.csv", "pg25_30.csv", "pg50_30.csv"]
    l130_mean = [] 
    l130_err = []
    o104_mean = []
    o104_err = []
    for csv_file in csv_lst:
        df = pd.read_csv(csv_file)
        # print(df)
        l130_mean.append(df["l130"][0])
        l130_err.append(df["l130"][1])
        o104_mean.append(df["o104"][0])
        o104_err.append(df["o104"][1])      

    fig, ax = plt.subplots(figsize=figsize)
    # sample l130
    ax.errorbar(["Pure water", "PG 25wt%", "PG 50wt%"], 
                l130_mean, 
                yerr=l130_err, 
                fmt= "o:", ms=10, label = "Sample 1", elinewidth=2, capsize=8, alpha=0.6)    
    # sample o104
    ax.errorbar(["Pure water", "PG 25wt%", "PG 50wt%"], 
                o104_mean, 
                yerr=o104_err, 
                fmt= "s:", ms=10, label = "Sample 2", elinewidth=2, capsize=8, alpha=0.6)

    ax.set_ylabel("Initial Volume [pL]", font)
    ax.set_xlabel("PG solution", font)
    
    
    plt.tick_params(labelsize=13)
    tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in tick_labels]

    legend_labels = ax.legend(loc="best", frameon=False).get_texts()
    [label.set_fontname('Times New Roman') for label in legend_labels]
    # ax.set_ylim(20, 90)

    if save_fig:
        plt.savefig(f"initial_volume.png", dpi=500)
    if show:
        plt.show()


def plot_mask_on_original_img():
    mask_img_path = "../02_Research/00_Experimental Data/20210208_pg50%/l130_6/22133.333ms_mask.png"
    image_path = "../02_Research/00_Experimental Data/20210208_pg50%/l130_6/22133.333ms.png"
    
    image = cv2.imread(image_path)
    mask_image = cv2.imread(mask_img_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    image = image[:, :, ::-1]
    image[..., 0] = np.where(mask_image == 255, 255, image[..., 0])
    plt.imshow(image)
    plt.axis("off")
    if save_fig:
        plt.savefig(f"mask_on_original_img.png", bbox_inches='tight', pad_inches = 0)
    if show:
        plt.show()    


if __name__ == '__main__':
    save_fig = True
    show = False
    plot_mask_on_original_img()