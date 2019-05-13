import cv2
import io
import os
import subprocess
import glob
import numpy as np
import random
#base_dir = "/Users/madhuhegde/Downloads/cholec80/"
#image_dir = base_dir+"images/"
#label_dir = base_dir+"labels/"

class_labels = {"Preparation\n":0, "CalotTriangleDissection\n":1, "ClippingCutting\n":2, 
           "GallbladderDissection\n":3, "GallbladderPackaging\n":4, "CleaningCoagulation\n":5, "GallbladderRetraction\n":6}


def split_cholec_data(image_dir, label_dir, ratio=0.2):

  #image_files = glob.glob(image_dir+"*.jpg")
  label_files = glob.glob(label_dir+"*.txt")

  train_X_list = list()
  test_X_list = list()
  
  train_y_list = list()
  test_y_list = list()
  
  test_ratio = int(10*ratio)


  for label_file in label_files:
    with open(label_file) as handle:
      #Read extra line that says Frames Phases
      handle.readline()
      labels = handle.readlines()
      
    #print(len(labels))
    train_clip_list = np.arange(len(labels))
     
    test_clip_list = []
    for i in range(int(len(labels)/10)):
         test_clip_list.extend(random.sample(range(i*10, (i+1)*10), test_ratio))
        
    test_clip_list.sort()
    test_clip_array = np.array(test_clip_list)
    #print(test_seg_array)
      
    train_clip_list = np.setdiff1d(train_clip_list,test_clip_array)
      
    #test_array.append(test_seg_array.reshape(1, -1))
    #train_array.append(train_seg_array.reshape(1, -1))
    train_clip_list = list(train_clip_list)
    
    label_array = np.array(labels)
    train_clip_y = label_array[train_clip_list]
    test_clip_y = label_array[test_clip_list]
    train_y_list.extend(train_clip_y)
    test_y_list.extend(test_clip_y)
      
      
      
    label_file_name = label_file.split('/')[-1]
      
    image_folder = image_dir+label_file_name.replace('-label.txt', '')
    image_files = glob.glob(image_folder+"/*.jpg")
      
    image_files = np.array(image_files)
    train_clip_X = image_files[train_clip_list]
    test_clip_X = image_files[test_clip_list]
    train_X_list.extend(train_clip_X)
    test_X_list.extend(test_clip_X)
      
      
      
    print (test_X_list[1:20])
 
  return


