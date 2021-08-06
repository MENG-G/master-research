"""
image processing is needed before running this code
image size: width: 640 x height: 480

input: processed images with 4 designeted points
output: a csv file recording coordinates of these 4 points of each image
"""
import cv2 as cv
import numpy as np
import argparse
import os
import csv
import glob
import re

def parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', default='../processed', help='images directory')
    parser.add_argument('--save_dir', default='../input_points', help='input csv')
    parser.add_argument('--paper', default='sword_5', help='select papers')
    parser.add_argument('--verbose', default=True, help='whether to print log')
    parser.add_argument('--step', default=1, help='step')
    parser.add_argument('--fps', default=30, help='fps')

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()

    
    # file_path = os.path.join(args.data_dir, args.paper)
    # file_path = r"./l90_2"
    # images = glob.glob(file_path + "/*ms.png")
    images = ["./3816.667ms.png", "./3833.333ms.png"]
    images = sorted(images, key=lambda x: float(".".join(re.split("[./\\\ms]", x)[-5:-3])))
    
    info = [['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'filename']]
    for image_path in images:
        img = cv.imread(image_path)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        index = np.where(img_gray==0)

        if index[0].size==0:
            continue
        else:
            x, y = zip(*sorted(list(zip(index[1], index[0]))))
        
        info.append([x[0], y[0], x[3], y[3], x[1], y[1], x[2], y[2], image_path])

    csv_path = r"./points.csv"
    with open(csv_path, 'w', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(info)
    
    print('Input points have been saved!') 