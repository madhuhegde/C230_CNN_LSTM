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

class_labels = {"Preparation":0, "CalotTriangleDissection":1, "ClippingCutting":2, 
           "GallbladderDissection":3, "GallbladderPackaging":4, "CleaningCoagulation":5, "GallbladderRetraction":6}



def generate_feature_list(image_dir, label_dir):

  
  label_files = glob.glob(label_dir+"video*.txt")

  feature_list = list()
  

  for label_file in label_files:
    with open(label_file) as handle:
      #Read extra line that says Frames Phases
      handle.readline()
      labels = handle.readlines()
      
    print(len(labels))
    label_file_name = label_file.split('/')[-1]
      
    image_folder = image_dir+label_file_name.replace('-label.txt', '')
    print("Image Folder:"+ image_folder)
    image_files = glob.glob(image_folder+"/video*.jpg")
    image_files.sort(key=os.path.getmtime)
    print(len(image_files))
    local_image_files = list()
    for image_file in image_files:
      file_name = image_file.split('/')
      local_image_files.append(file_name[-2]+'/'+file_name[-1])

      
    print(len(local_image_files), len(labels))
    feature_list.extend([local_image_files[i], labels[i]] for i in range(len(labels)))
  
  return(feature_list)
