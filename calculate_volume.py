import math
import argparse
import csv
import os
import re
import numpy as np
import rsa
from calculate_evaporate import hu_and_larson
import pandas as pd

"""
pixel_len1: 0.9748102696798955 # 5x
pixel_len2: 0.2418065541763232 # 20x

parameters:
R: radius of sphere
r: radius of droplet bottom circle
d: 2 * r 
"""
pixel_len1 = 0.9748102696798955
pixel_len2 = 0.2418065541763232


def parse():
    parser = argparse.ArgumentParser(description='calculate the volume of droplet')
    
    parser.add_argument('--data_dir', default='../input_points', help='video directory')
    parser.add_argument('--save_dir', default='../output_results', help='folder where to save extracted frames')
    parser.add_argument('--paper', default='sword_1', help='select papers')
    parser.add_argument('--verbose', default=True, help='whether to print log')
    # sword and next setp should be 1, other papers should be 3
    parser.add_argument('--step', default=1, help='step')
    parser.add_argument('--fps', default=60, help='fps')

    args = parser.parse_args()
    
    return args


def get_sphere(xa, ya, xb, yb, xc, yc):
    A = xa ** 2 - xb ** 2 + ya ** 2 - yb ** 2
    B = xa ** 2 - xc ** 2 + ya ** 2 - yc ** 2
    yac = ya - yc
    yab = ya - yb
    xac = xa - xc
    xab = xa - xb
    if ((yac * xab) - (yab * xac)) != 0:
        center_x = ((A * yac) - (B * yab)) / (2 * ((yac * xab) - (yab * xac)))
        center_y = ((A * xac) - (B * xab)) / (2 * ((yab * xac) - (yac * xab)))
        R = math.sqrt((xa - center_x) * (xa - center_x) + (ya - center_y) * (ya - center_y))
    else:
        center_x = 1
        center_y = 1
        R = 1
    return center_x, center_y, R


def get_volume(r, h):
    v = (math.pi * h / 6) * (3 * r ** 2 + h ** 2)
    return v * 10 ** (-3)


def creat_csv():
    paper_names = ['l90', 'l130', 'l200', 'l250', 'o73', 'o84', 'o104', 'o128', 'next', 'sword']
    for paper_name in paper_names:
        for i in range(1, 4):
            file_path = '../input_points/' + paper_name + '_{0}'.format(i) + '.csv'
            with open(file_path, 'w') as csvfile:
                fieldnames = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()


def get_contact_angle(h, r):
    theta = math.atan2(h, r)
    return math.degrees(2 * theta)


def get_distance(x1, y1, x2, y2):
    d = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return d


def get_surface_area(R, h):
    s = 2 * math.pi * R * h
    return s


def calculate_droplet_parameters(args):

    paper_names = ['l90', 'l130', 'l200', 'l250', 'o73', 'o84', 'o104', 'o128', 'next', 'sword']
    for paper_name in paper_names:
        for num in range(1, 7):
            input_path = os.path.join(args.data_dir, paper_name + f'_{num}' + '.csv')
            data = [['time[s]', 'time[ms]', 'bottom radius', 'Mj(pL)', 'theta', 'area(m^2)']]
            with open(input_path, 'r') as lines:
                for i, line in enumerate(lines):
                    if i == 0:
                        continue

                    if len(re.split(r',', line)) == 8:
                        x1, y1, x2, y2, x3, y3, x4, y4, = re.split(r',', line)
                    else: 
                        x1, y1, x2, y2, x3, y3, x4, y4, _ = re.split(r',', line)
                    x1, y1, x2, y2, x3, y3, x4, y4 = float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)
                    
                    # calculate sphere center and radius through 4 points
                    center_x1, center_y1, R1 = get_sphere(x1, y1, x2, y2, x3, y3)
                    center_x2, center_y2, R2 = get_sphere(x1, y1, x2, y2, x4, y4)
                    _, _, R = (center_x1 + center_x2) / 2, (center_y1 + center_y2) / 2, (R1 + R2) / 2

                    r = (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2) * pixel_len1   # um
                    # calculate the height of droplet
                    R = R * pixel_len1
                    h = R - math.sqrt(R ** 2 - r ** 2)   # um
                    theta = get_contact_angle(h, r)
                    V = get_volume(r, h) # uL
                    S = get_surface_area(R, h) * 10 ** (-12) # m^2
                    
                    time = args.step * (i-1) / args.fps 
                    time_ms = time * 1000
                    info = [time, time_ms, r, V, theta, S]
                    data.append(info)
            
            output_path = os.path.join(args.save_dir, paper_name + f'_{num}' + '.csv')
            # write results to csv
            with open(output_path, 'w', newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)


def calculate_evaporate_volume():
    paper_names = ['l90', 'l130', 'l200', 'l250', 'o73', 'o84', 'o104', 'o128', 'next', 'sword']
    for paper_name in paper_names:
        for num in range(1, 7):
            input_path = '../output_results/' + paper_name + f'_{num}' + '.csv'
            output_path = '../evaporation_results/' + paper_name + f'_{num}' + '.csv'
            hu_and_larson(input_path, output_path, PG_wt_frac=0, T=26.152, RH=51.519, delta_T=0.1)


def calculate_droplet_parameters_oneFile(args):
    # input_path = os.path.join(args.data_dir, paper_name + f'_{num}' + '.csv')
    input_path =r"C:/Users/kakum/Desktop/02_Research/00_Experimental Data/20210205_pg25%/mask/l90_2/points.csv"
    print(os.path.exists(input_path))
    data = [['time[ms]', 'bottom_radius[um]', 'theta[degree]', 'height[um]', 'area[m^2]', 'Mj[pL]']]
    with open(input_path, 'r') as lines:
        print("Successfully opened the file!")

        for i, line in enumerate(lines):
            if i == 0:
                continue

            if len(re.split(r',', line)) == 8:
                x1, y1, x2, y2, x3, y3, x4, y4, = re.split(r',', line)
            else: 
                x1, y1, x2, y2, x3, y3, x4, y4, _ = re.split(r',', line)
            x1, y1, x2, y2, x3, y3, x4, y4 = float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)
            
            # calculate sphere center and radius through 4 points
            center_x1, center_y1, R1 = get_sphere(x1, y1, x2, y2, x3, y3)
            center_x2, center_y2, R2 = get_sphere(x1, y1, x2, y2, x4, y4)
            _, _, R = (center_x1 + center_x2) / 2, (center_y1 + center_y2) / 2, (R1 + R2) / 2

            r = (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2) * pixel_len2   # um
            # calculate the height of droplet
            R = R * pixel_len2
            h = R - math.sqrt(R ** 2 - r ** 2)   # um
            theta = get_contact_angle(h, r)
            V = get_volume(r, h) # pL
            S = get_surface_area(R, h) * 10 ** (-12) # m^2
            
            time = args.step * (i-1) / args.fps 
            time_ms = time * 1000
            info = [time_ms, r, theta, h, S, V]
            data.append(info)
    
    # output_path = os.path.join(args.save_dir, paper_name + f'_{num}' + '.csv')
    output_path = r"C:/Users/kakum/Desktop/02_Research/00_Experimental Data/20210205_pg25%/mask/l90_2/l90_2.csv"

    # write results to csv
    with open(output_path, 'w', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def get_initial_info(data_dir):
    papers = ['l90', 'l130', 'l200', 'l250', 'o73', 'o84', 'o104', 'o128']

    initial_volumelist = {}
    initial_theta = {}
    initial_r = {}
    res = {}

    for paper in papers:
        initial_volumelist[paper] = []
        initial_theta[paper] = []
        initial_r[paper] = []
        res[paper] = []
    
    for paper in papers:
        for i in range(1, 8):
            file = f"{data_dir}/{paper}_{i}.csv"
            if not os.path.exists(file):
                continue

            data = pd.read_csv(file)
            initial_volumelist[paper].append(data['Mj[pL]'][0])
            initial_theta[paper].append(data['theta[degree]'][0])
            initial_r[paper].append(data['bottom_radius[um]'][0])
        
        res[paper].append(np.mean(initial_volumelist[paper])) # initial volume mean
        res[paper].append(np.std(initial_volumelist[paper])) # initial volume std
        res[paper].append(np.mean(initial_theta[paper])) # initial initial_theta mean
        res[paper].append(np.std(initial_theta[paper])) # initial initial_theta std
        res[paper].append(np.mean(initial_r[paper])) # initial r mean
        res[paper].append(np.std(initial_r[paper])) # initial r std

    print(res)
    df = pd.DataFrame(res)
    df.to_csv("initial_informations.csv")




if __name__=='__main__':

    # data_dir = '../02_Research/00_Experimental Data/20210127_purewater/output_results' # pure water
    # data_dir = '../02_Research/00_Experimental Data/20210205_pg25%/output_results/' # pg25
    data_dir = '../02_Research/00_Experimental Data/20210208_pg50%/output_results/' # pg50
    get_initial_info(data_dir)
