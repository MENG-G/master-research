import pandas as pd
import matplotlib.pyplot as plt
import os
import re


font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12} 
figsize = (5, 4.5)

def sigmaV():
    data_path = '../02_Research/00_Experimental Data/20200731_Gaussian Fitting/python/all_volume_data'
    
    LAG_lst = ['SWORD2', 'LAG90', 'LAG130', 'LAG200', 'LAG250', 'NEXT-IJ1',]
    OKT_lst =['SWORD2', 'OKT73.3', 'OKT84.9', 'OKT104.7', 'OKT128', 'NEXT-IJ1',]
    marker = {'LAG90':'o', 'LAG130':'o', 'LAG200':'o', 'LAG250':'o', 
            'OKT73.3':'d', 'OKT84.9':'d', 'OKT104.7':'d', 'OKT128':'d',
            'NEXT-IJ1':'x', 'NEXT-IJ2':'x', 'SWORD1':'+', 'SWORD2':'+'}

    paper_lst = ['LAG130', 'OKT104.7']
    paper_label = {'LAG130': 'Sample 1', 'OKT104.7': 'Sample 2'}
    fmt = {'LAG130': 'o:', 'OKT104.7': 's:'}
    
    fig, ax = plt.subplots(figsize=figsize)

    for paper in paper_lst:
        csv_path = os.path.join(data_path, paper + '.csv')
        data = pd.read_csv(csv_path)
        ax.plot(data['dp/nm'], data['Sigma Vp'], fmt[paper], label=paper_label[paper], alpha=0.6)
    
    # legend
    legend_labels = ax.legend(loc="best", frameon=False).get_texts()
    [label.set_fontname('Times New Roman') for label in legend_labels]

    ax.set_xscale('log')
    
    ax.set_xlabel('D [nm]', font)
    ax.set_ylabel('$V$ [ml/g]', font)
    plt.tick_params(labelsize=12)
    tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in tick_labels]    
    
    ax.set_xlim(10)
    ax.set_ylim(0, 0.05)
    
    if save_fig:
        plt.savefig(f"sigmaV.png", dpi=500)
    if show:
        plt.show()



def dVdlogD():
    data_path = '../02_Research/00_Experimental Data/20200731_Gaussian Fitting/python/'
    data = pd.read_csv(data_path + 'all_raw_data_deleted.csv')
    # data = pd.read_csv(data_path + 'all_raw_data.csv')

    LAG_lst = ['SWORD2', 'LAG90', 'LAG130', 'LAG200', 'LAG250', 'NEXT-IJ1']
    OKT_lst =['SWORD2', 'OKT73.3', 'OKT84.9', 'OKT104.7', 'OKT128', 'NEXT-IJ1']
    marker = {'LAG90':'o', 'LAG130':'o', 'LAG200':'o', 'LAG250':'o', 
            'OKT73.3':'d', 'OKT84.9':'d', 'OKT104.7':'d', 'OKT128':'d',
            'NEXT-IJ1':'x', 'NEXT-IJ2':'x', 'SWORD1':'+', 'SWORD2':'+'}
    
    paper_lst = ['LAG130', 'OKT104.7']
    paper_label = {'LAG130': 'Sample 1', 'OKT104.7': 'Sample 2'}
    fmt = {'LAG130': 'o', 'OKT104.7': 's'}
    dvdlogD_path = data_path + 'processed_data_100/'

    fig, ax = plt.subplots(figsize=figsize)
    
    # plot
    for paper in paper_lst:
        # gaussian fitting results
        df = pd.read_csv(dvdlogD_path + f'{paper}.csv')
        ax.plot(df['d'], df['dVdlogD'], label=f'{paper_label[paper]} Gaussian Fitting')
        # experiental results
        ax.scatter(data['dp/nm'], data[paper], marker=fmt[paper], label=f'{paper_label[paper]} Exp.', alpha=0.6)
    
    # legend
    legend_labels = ax.legend(loc="upper left", frameon=False).get_texts()
    [label.set_fontname('Times New Roman') for label in legend_labels]

    ax.set_xscale('log')
    
    ax.set_xlabel('D [nm]', font)
    ax.set_ylabel('$dV / d \log{D}$ [ml/g]', font)
    plt.tick_params(labelsize=12)
    tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in tick_labels]    
    
    ax.set_xlim(10)
    ax.set_ylim(0, 0.05)
    
    if save_fig:
        plt.savefig(f"dVdlogD.png", dpi=500)
    if show:
        plt.show()



def nperArea():
    data_path = '../02_Research/00_Experimental Data/20200731_Gaussian Fitting/python/'
    paper_lst = ['LAG130', 'OKT104.7']
    paper_label = {'LAG130': 'Sample 1', 'OKT104.7': 'Sample 2'}
    fmt = {'LAG130': 'o', 'OKT104.7': 's'}
    dvdlogD_path = data_path + 'processed_data_100/'


    fig, ax = plt.subplots(figsize=figsize)
    for paper in paper_lst:
        # gaussian fitting results
        df = pd.read_csv(dvdlogD_path + f'{paper}.csv')
        ax.plot(df['d'], df['n'], label=f'{paper_label[paper]} Gaussian Fitting')    
    
    
    # legend
    legend_labels = ax.legend(loc="best", frameon=False).get_texts()
    [label.set_fontname('Times New Roman') for label in legend_labels]

    ax.set_xscale('log')
    
    ax.set_xlabel('D [nm]', font)
    ax.set_ylabel('Number of Pores [$m^{-2}$]', font)
    plt.tick_params(labelsize=12)
    tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in tick_labels]    
    
    # ax.set_xlim(10)
    # ax.set_ylim(0, 0.05)
    
    if save_fig:
        plt.savefig(f"n.png", dpi=500)
    if show:
        plt.show()


def dVdlogD_exp():
    data_path = '../02_Research/00_Experimental Data/20200731_Gaussian Fitting/python/'
    data = pd.read_csv(data_path + 'all_raw_data.csv')    

    paper_lst = ['LAG130', 'OKT104.7']
    paper_label = {'LAG130': 'Sample 1', 'OKT104.7': 'Sample 2'}
    fmt = {'LAG130': 'o', 'OKT104.7': 's'}

    fig, ax = plt.subplots(figsize=figsize)
    
    # plot
    for paper in paper_lst:
        # gaussian fitting results
        ax.plot(data['dp/nm'], data[paper], label=f'{paper_label[paper]} Exp.', marker=fmt[paper], alpha=0.6)

    
    # legend
    legend_labels = ax.legend(loc="upper left", frameon=False).get_texts()
    [label.set_fontname('Times New Roman') for label in legend_labels]

    ax.set_xscale('log')
    
    ax.set_xlabel('D [nm]', font)
    ax.set_ylabel('$dV / d \log{D}$ [ml/g]', font)
    plt.tick_params(labelsize=12)
    tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in tick_labels]    
    
    ax.set_xlim(10)
    ax.set_ylim(0, 0.05)
    
    if save_fig:
        plt.savefig(f"dVdlogD_Exp.png", dpi=500)
    if show:
        plt.show()


    
if __name__ == '__main__':
    save_fig = True
    show = True

    # sigmaV()
    # dVdlogD_exp()
    dVdlogD()
    # nperArea()



    