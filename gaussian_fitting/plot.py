from random import sample
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12} 
figsize = (5.3, 4.7)

def sigmaV():
    data_path = '../02_Research/00_Experimental Data/20200731_Gaussian Fitting/python/all_volume_data'
    
    # LAG_lst = ['SWORD2', 'LAG90', 'LAG130', 'LAG200', 'LAG250', 'NEXT-IJ1',]
    # OKT_lst =['SWORD2', 'OKT73.3', 'OKT84.9', 'OKT104.7', 'OKT128', 'NEXT-IJ1',]
    LAG_lst = ['LAG90', 'LAG130', 'LAG200', 'LAG250', 'SWORD1', 'SWORD2', 'NEXT-IJ1', 'NEXT-IJ2']
    OKT_lst =['OKT73.3', 'OKT84.9', 'OKT104.7', 'OKT128', 'SWORD1', 'SWORD2', 'NEXT-IJ1', 'NEXT-IJ2']
    c_lst = ['SWORD1', 'SWORD2']
    d_lst = ['NEXT-IJ1', 'NEXT-IJ2']
     
    marker = {'LAG90':'o', 'LAG130':'o', 'LAG200':'o', 'LAG250':'o', 
            'OKT73.3':'s', 'OKT84.9':'s', 'OKT104.7':'s', 'OKT128':'s',
            'NEXT-IJ1':'^', 'NEXT-IJ2':'^', 'SWORD1':'*', 'SWORD2':'*'}

    # paper_lst = ['LAG130', 'OKT104.7']
    paper_label = {'LAG90':'A-1', 'LAG130':'A-2', 'LAG200':'A-3', 'LAG250':'A-4', 
            'OKT73.3':'B-1', 'OKT84.9':'B-2', 'OKT104.7':'B-3', 'OKT128':'B-4',
            'NEXT-IJ1':'D-Exp.1', 'NEXT-IJ2':'D-Exp.2', 'SWORD1':'C-Exp.1', 'SWORD2':'C-Exp.2'}
    # fmt = {'LAG130': 'o:', 'OKT104.7': 's:'}
    sample_name = ['A', 'B', 'C', 'D']

    for i, paper_lst in enumerate([LAG_lst, OKT_lst]):
        name = sample_name[i]
        fig, ax = plt.subplots(figsize=figsize)
        for paper in paper_lst:
            csv_path = os.path.join(data_path, paper + '.csv')
            data = pd.read_csv(csv_path)
            ax.plot(data['dp/nm'], data['Sigma Vp'], marker[paper], label=paper_label[paper], alpha=0.6)

    
        # legend
        legend_labels = ax.legend(loc="best", frameon=False).get_texts()
        [label.set_fontname('Times New Roman') for label in legend_labels]

        ax.set_xscale('log')
        
        ax.set_xlabel('$d_p$ [nm]', font)
        ax.set_ylabel('$V_a$ [ml/g]', font)
        plt.tick_params(labelsize=12)
        tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in tick_labels]    
        
        ax.set_xlim(10)
        ax.set_ylim(0, 0.05)
        
        if save_fig:
            plt.savefig(f"{name}_sigmaV.png", dpi=500)
        if show:
            plt.show()

def sigmaV2():
    data_path = '../02_Research/00_Experimental Data/20200731_Gaussian Fitting/python/all_volume_data'
    
    # LAG_lst = ['SWORD2', 'LAG90', 'LAG130', 'LAG200', 'LAG250', 'NEXT-IJ1',]
    # OKT_lst =['SWORD2', 'OKT73.3', 'OKT84.9', 'OKT104.7', 'OKT128', 'NEXT-IJ1',]
    LAG_lst = ['LAG90', 'LAG130', 'LAG200', 'LAG250', 'SWORD1', 'SWORD2', 'NEXT-IJ1', 'NEXT-IJ2']
    OKT_lst =['OKT73.3', 'OKT84.9', 'OKT104.7', 'OKT128', 'SWORD1', 'SWORD2', 'NEXT-IJ1', 'NEXT-IJ2']
    c_lst = ['SWORD1', 'SWORD2']
    d_lst = ['NEXT-IJ1', 'NEXT-IJ2']
     
    marker = {'LAG90':'o', 'LAG130':'o', 'LAG200':'o', 'LAG250':'o', 
            'OKT73.3':'s', 'OKT84.9':'s', 'OKT104.7':'s', 'OKT128':'s',
            'NEXT-IJ1':'^', 'NEXT-IJ2':'^', 'SWORD1':'*', 'SWORD2':'*'}
    
    bs_weight = {'LAG90': 90, 'LAG130': 130, 'LAG200': 200, 'LAG250': 250, 
            'OKT73.3': 73.3, 'OKT84.9': 84.9, 'OKT104.7': 104.7, 'OKT128': 127.9,
            'NEXT-IJ1': 81.4, 'NEXT-IJ2': 81.4, 'SWORD1': 128, 'SWORD2': 128}

    # paper_lst = ['LAG130', 'OKT104.7']
    paper_label = {'LAG90':'A-1', 'LAG130':'A-2', 'LAG200':'A-3', 'LAG250':'A-4', 
            'OKT73.3':'B-1', 'OKT84.9':'B-2', 'OKT104.7':'B-3', 'OKT128':'B-4',
            'NEXT-IJ1':'D-Exp.1', 'NEXT-IJ2':'D-Exp.2', 'SWORD1':'C-Exp.1', 'SWORD2':'C-Exp.2'}
    # fmt = {'LAG130': 'o:', 'OKT104.7': 's:'}
    sample_name = ['A', 'B', 'C', 'D']

    for i, paper_lst in enumerate([LAG_lst, OKT_lst]):
        name = sample_name[i]
        fig, ax = plt.subplots(figsize=figsize)
        for paper in paper_lst:
            csv_path = os.path.join(data_path, paper + '.csv')
            data = pd.read_csv(csv_path)
            ax.plot(data['dp/nm'], data['Sigma Vp'] * bs_weight[paper] * 1e-4, marker[paper], label=paper_label[paper], alpha=0.6)

    
        # legend
        legend_labels = ax.legend(loc="best", frameon=False).get_texts()
        [label.set_fontname('Times New Roman') for label in legend_labels]

        ax.set_xscale('log')
        
        ax.set_xlabel('$d_p$ [$\mathrm{nm}$]', font)
        ax.set_ylabel('$V_a$ [$\mathrm{ml/cm^2}$]', font)
        plt.tick_params(labelsize=12)
        tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in tick_labels]    
        plt.tight_layout()
        
        ax.set_xlim(10)
        ax.set_ylim(0, 0.0008)
        
        if save_fig:
            plt.savefig(f"{name}_sigmaV.png", dpi=500)
        if show:
            plt.show()


def dVdlogD():
    data_path = '../02_Research/00_Experimental Data/20200731_Gaussian Fitting/python/'
    data = pd.read_csv(data_path + 'all_raw_data_deleted.csv')
    # data = pd.read_csv(data_path + 'all_raw_data.csv')

    LAG_lst = ['LAG90', 'LAG130', 'LAG200', 'LAG250']
    OKT_lst =['OKT73.3', 'OKT84.9', 'OKT104.7', 'OKT128']    
    c_lst = ['SWORD1', 'SWORD2']
    d_lst = ['NEXT-IJ1', 'NEXT-IJ2']    
    marker = {'LAG90':'o', 'LAG130':'o', 'LAG200':'o', 'LAG250':'o', 
            'OKT73.3':'s', 'OKT84.9':'s', 'OKT104.7':'s', 'OKT128':'s',
            'NEXT-IJ1':'^', 'NEXT-IJ2':'^', 'SWORD1':'*', 'SWORD2':'*'}
    
    paper_label = {'LAG90':'A-1', 'LAG130':'A-2', 'LAG200':'A-3', 'LAG250':'A-4', 
            'OKT73.3':'B-1', 'OKT84.9':'B-2', 'OKT104.7':'B-3', 'OKT128':'B-4',
            'NEXT-IJ1':'D-Exp.1', 'NEXT-IJ2':'D-Exp.2', 'SWORD1':'C-Exp.1', 'SWORD2':'C-Exp.2'}

    dvdlogD_path = data_path + 'processed_data_100/'
    sample_name = ['A', 'B', 'C', 'D']

    for i, paper_lst in enumerate([LAG_lst, OKT_lst, c_lst, d_lst]):
        fig, ax = plt.subplots(figsize=figsize)
        name = sample_name[i]
        for paper in paper_lst:
            # gaussian fitting results
            df = pd.read_csv(dvdlogD_path + f'{paper}.csv')
            ax.plot(df['d'], df['dVdlogD'], label=f'{paper_label[paper]} Gaussian Fitting')
            # experiental results
            ax.scatter(data['dp/nm'], data[paper], marker=marker[paper], label=f'{paper_label[paper]} Exp.', alpha=0.4)
    
    # legend
        legend_labels = ax.legend(loc="upper left", frameon=False).get_texts()
        [label.set_fontname('Times New Roman') for label in legend_labels]

        ax.set_xscale('log')
        
        ax.set_xlabel('$d_p$ [nm]', font)
        ax.set_ylabel('$dV_a / d \log d_p $ [ml/g]', font)
        plt.tick_params(labelsize=12)
        tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in tick_labels]    
        plt.tight_layout()
        ax.set_xlim(1, )
        ax.set_ylim(0, )
        
        if save_fig:
            plt.savefig(f"{name}_dVdlogD.png", dpi=500)
        if show:
            plt.show()



def nperArea():
    data_path = '../02_Research/00_Experimental Data/20200731_Gaussian Fitting/python/'
    
    LAG_lst = ['LAG90', 'LAG130', 'LAG200', 'LAG250']
    OKT_lst =['OKT73.3', 'OKT84.9', 'OKT104.7', 'OKT128']    
    c_lst = ['SWORD1', 'SWORD2']
    d_lst = ['NEXT-IJ1', 'NEXT-IJ2']    
    marker = {'LAG90':'o', 'LAG130':'o', 'LAG200':'o', 'LAG250':'o', 
            'OKT73.3':'s', 'OKT84.9':'s', 'OKT104.7':'s', 'OKT128':'s',
            'NEXT-IJ1':'^', 'NEXT-IJ2':'^', 'SWORD1':'*', 'SWORD2':'*'}
    
    paper_label = {'LAG90':'A-1', 'LAG130':'A-2', 'LAG200':'A-3', 'LAG250':'A-4', 
            'OKT73.3':'B-1', 'OKT84.9':'B-2', 'OKT104.7':'B-3', 'OKT128':'B-4',
            'NEXT-IJ1':'D-Exp.1', 'NEXT-IJ2':'D-Exp.2', 'SWORD1':'C-Exp.1', 'SWORD2':'C-Exp.2'}
    dvdlogD_path = data_path + 'processed_data_100/'
    sample_name = ['A', 'B', 'C', 'D']

    for i, paper_lst in enumerate([LAG_lst, OKT_lst, c_lst]):
        fig, ax = plt.subplots(figsize=figsize)
        name = sample_name[i]
        for paper in paper_lst:
            # gaussian fitting results
            df = pd.read_csv(dvdlogD_path + f'{paper}.csv')
            ax.plot(df['d'], df['n'], label=f'{paper_label[paper]} Gaussian Fitting')    
    
        # legend
        legend_labels = ax.legend(loc="best", frameon=False).get_texts()
        [label.set_fontname('Times New Roman') for label in legend_labels]

        ax.set_xscale('log')
        
        ax.set_xlabel('$d_p$ [nm]', font)
        ax.set_ylabel('Number of Pores [$\mathrm{m^{-2}}$]', font)
        plt.tick_params(labelsize=12)
        tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in tick_labels]    
        plt.tight_layout()
        if save_fig:
            plt.savefig(f"{name}_n.png", dpi=500)
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
    ax.set_ylabel('$dV_a / d \log{D}$ [ml/g]', font)
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
    save_fig = False
    show = True

    # save_fig = True
    # show = False
    # sigmaV2()
    # dVdlogD_exp()
    # dVdlogD()
    nperArea()



    