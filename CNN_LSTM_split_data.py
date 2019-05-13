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
  label_files = glob.glob(label_dir+"video*.txt")

  train_array = list()
  test_array = list()
  test_ratio = int(10*ratio)


  for label_file in label_files:
    with open(label_file) as handle:
      #Read extra line that says Frames Phases
      handle.readline()
      labels = handle.readlines()
      
      #print(len(labels))
      train_seg_array = np.arange(len(labels))
      #print(train_seg_array)
      test_seg_array = []
      for i in range(int(len(labels)/10)):
         test_seg_array.extend(random.sample(range(i*10, (i+1)*10), test_ratio))
        
      test_seg_array.sort()
      test_seg_array = np.array(test_seg_array)
      #print(test_seg_array)
      
      train_seg_array = np.setdiff1d(train_seg_array,test_seg_array)
      
      test_array.append(test_seg_array.reshape(1, -1))
      train_array.append(train_seg_array.reshape(1, -1))
 
  return(train_array, test_array)


