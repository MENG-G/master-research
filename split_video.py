import argparse
import cv2
import os
import glob
import re

def parse():
    parser = argparse.ArgumentParser(description='extract frames from video')
    
    parser.add_argument('--data_dir', default='./', help='video directory')
    parser.add_argument('--save_dir', 
        default='C:/Users/kakum/Desktop/02_Research/00_Experimental Data/20210704_pigment/', 
        help='folder where to save extracted frames')
    parser.add_argument('--step', default=2, help='step')
    parser.add_argument('--fps', default=10, help='fps')

    parser.add_argument('--verbose', default=True, help='whether to print log')

    args = parser.parse_args()
    
    return args


def extract_frames(video_path, args):
    video_name = re.split("[\\\./]", video_path)[-2]
    save_path = os.path.join(args.save_dir, video_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    cap =cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("File open failed.")
    
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for f in range(0, total_frame, args.step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        # change to [ms]
        time = f * 1000 / args.fps 
        cv2.imwrite(save_path+'/{:.3f}ms.png'.format(time), frame)
        if args.verbose and (f/args.step)%10==0:
            print('{:>4d} images have been saved.'.format(int(f/args.step)))
    
    print("All frames have been saved.")
    cap.release()
    

if __name__=='__main__':
    args = parse()
    video_lst = glob.glob("../02_Research/00_Experimental Data/20210704_pigment/l130_pig6wt.mp4")
    for video_path in video_lst:
    # print(re.split("[\\\.]", video_lst[0])[-2])
    # print(video_lst)
        # print(video_path)
        extract_frames(video_path, args)
        