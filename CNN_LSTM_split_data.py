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



def generate_feature_train_list(image_dir, label_dir):

  
  label_files = glob.glob(label_dir+"video*.txt")

  feature_list = list()
  

  for label_file in label_files:
    #print(label_file)  
    with open(label_file) as handle:
      #Read extra line that says Frames Phases
      handle.readline()
      labels = handle.readlines()
      
    #print(len(labels))
    label_file_name = label_file.split('/')[-1]
      
    image_folder = image_dir+label_file_name.replace('-label.txt', '')
    #print(image_folder)
    image_files = glob.glob(image_folder+"/video*.jpg")
    image_files.sort(key=os.path.getmtime)
    #print(len(image_files))
    local_image_files = list()
    for image_file in image_files:
      file_name = image_file.split('/')
      local_image_files.append(file_name[-2]+'/'+file_name[-1])

    rand_choice = [0,1]
    if(random.choice(rand_choice)):
        data_aug_1 = 0
        data_aug_2 = 1
    else:
        data_aug_1 = 0
        data_aug_2 = 0

    #augmentation disabled
    #print(len(local_image_files), len(labels))
    feature_list.extend([local_image_files[i], labels[i], data_aug_1] for i in range(len(labels)))
    #feature_list.extend([local_image_files[i], labels[i], data_aug_2] for i in range(len(labels)))
  
  return(feature_list)

def generate_feature_test_list(image_dir, label_dir):

  
  label_files = glob.glob(label_dir+"video*.txt")

  feature_list = list()
  

  for label_file in label_files:
    #print(label_file)  
    with open(label_file) as handle:
      #Read extra line that says Frames Phases
      handle.readline()
      labels = handle.readlines()
      
    #print(len(labels))
    label_file_name = label_file.split('/')[-1]
      
    image_folder = image_dir+label_file_name.replace('-label.txt', '')
    #print(image_folder)
    image_files = glob.glob(image_folder+"/video*.jpg")
    image_files.sort(key=os.path.getmtime)
    #print(len(image_files))
    local_image_files = list()
    for image_file in image_files:
      file_name = image_file.split('/')
      local_image_files.append(file_name[-2]+'/'+file_name[-1])

      
    #print(len(local_image_files), len(labels))
    feature_list.extend([local_image_files[i], labels[i]] for i in range(len(labels)))
  
  return(feature_list)
