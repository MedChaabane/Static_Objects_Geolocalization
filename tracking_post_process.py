import os
import cv2
import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Joint Pose and Association Train')
parser.add_argument("--test_root", type=str, default='/s/red/a/nobackup/vision/nuScenes/Tracking/data9/test/', help="path to test root ")
parser.add_argument("--predictions_root", type=str, default='/s/red/a/nobackup/vision/nuScenes/Tracking/Tracking_TLs/videos/', help="path to the output files ")
parser.add_argument("--save_folder", type=str, default='/s/red/a/nobackup/vision/nuScenes/Tracking/evaluate_tracking/', help="path where to save ground truth files in same MOT format")





    
    
    
    
def post_process():
    args = parser.parse_args()    
    predictions_folder = args.predictions_root
    save_predictions_folder = args.save_folder +  'predictions/'   
    if not os.path.exists(save_predictions_folder):
        os.makedirs(save_predictions_folder)
    separator = ','
    for file in os.listdir(predictions_folder):
        if file.endswith('.txt'):
            fol = predictions_folder+file
            first=True
            with open(save_predictions_folder+file[3:], "w") as file1: 
                with open(fol) as fp:
                   for cnt, line in enumerate(fp):
                       if not first :
                           file1.write('\n')
                       ll = line.split()

                       ll=ll[:6]
                       new_ll=[]
                       for el in ll:
                            new_ll.append(int(float(el)))
                       new_ll[0]+=1
                       ll=[]
                       for el in  new_ll:
                           ll.append(str(el))
                       ll=ll+['-1','-1','-1','-1']
                       file1.write(separator.join(ll)) 
                       first=False
                
                



    test_root = args.test_root
    save_gt_folder = args.save_folder + 'GT/' 
    if not os.path.exists(save_gt_folder):
        os.makedirs(save_gt_folder)

    for file in os.listdir(test_root):
        fol = test_root +file
        if not os.path.exists(save_gt_folder+file):
            os.makedirs(save_gt_folder+file)
            os.makedirs(save_gt_folder+file+'/gt')
        gt_file = fol+'/gt/gt.txt'
        first=True
        with open(save_gt_folder+file+'/gt/'+'gt.txt', "w") as file1: 
            with open(gt_file) as fp:
               for cnt, line in enumerate(fp):
                   if not first :
                        file1.write('\n')
                   ll = line.split(' ')

                   ll=ll[:6]+['1','1','1','1']
                   file1.write(separator.join(ll)) 
                   first=False
    
    
    
if __name__ == '__main__':
    post_process()
