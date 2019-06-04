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


def transition_clip(transition_samples):
   label_set = []
   for sample in transition_samples:
      label = sample[1].split('\t')[1].strip()
      label_set.append(label)
      
   label_set = set(label_set)
   
   #print(label_set)
   
   if(len(label_set)>1):
      return(True)
   else:
      return(False)     

def remove_transition_samples(samples, frames_per_clip = 25):
  smooth_samples = []
  num_frames = int(len(samples)/frames_per_clip)
    
  
  for frame_count in range(0, num_frames):
    
    sample_start = frame_count*frames_per_clip
    frame_samples = samples[sample_start:sample_start + frames_per_clip]
           
    if(transition_clip(frame_samples) == False):
      smooth_samples.extend(frame_samples)
      
  return(smooth_samples)

def generate_feature_augment_list(image_dir, label_dir, video_files):

  
  #label_files = glob.glob(label_dir+"video*.txt")
  label_files =  video_files
  #print(label_files)
  feature_list = list()
  
  for label_file in label_files:
  
    label_file_name = label_dir+label_file+'-label.txt'  
    with open(label_file_name) as handle:
      #Read extra line that says Frames Phases
      handle.readline()
      labels = handle.readlines()
      
    
    #print(len(image_files))
    aug_labels = []
    local_image_files = list()
    for label in labels:
      label_split = label.split('\t')
      index = str(int(label_split[0])+1)
      phase = label_split[1].strip()
     
      image_file = label_file+'-'+index+'.jpg'
      if(phase != "CalotTriangleDissection") and (phase != "GallbladderDissection"):
         aug_labels.append(label)
         local_image_files.append(label_file+'/'+image_file)
      
    #print(len(local_image_files), len(labels))
    feature_list.extend([local_image_files[i], aug_labels[i]] for i in range(len(aug_labels)))
    
  return(feature_list)
  
def generate_feature_train_list(image_dir, label_dir, video_files):

  
  #label_files = glob.glob(label_dir+"video*.txt")
  label_files =  video_files
  #print(label_files)
  feature_list = list()
  
  for label_file in label_files:
    label_file_name = label_dir+label_file+'-label.txt'
    #print(label_file_name)  
    
    with open(label_file_name) as handle:
      #Read extra line that says Frames Phases
      handle.readline()
      labels = handle.readlines()
   
    #print(len(image_files))
    local_image_files = list()
    for label in labels:
      label_split = label.split('\t')
      index = str(int(label_split[0])+1)
     
      image_file = label_file+'-'+index+'.jpg'

      local_image_files.append(label_file+'/'+image_file)
      
    #print(len(local_image_files), len(labels))
    feature_list.extend([local_image_files[i], labels[i]] for i in range(len(labels)))
    
  return(feature_list)

def generate_feature_test_list(image_dir, label_dir, video_files):

  
  label_files = video_files #glob.glob(label_dir+"video*.txt")

  feature_list = list()
  

  for label_file in label_files:
   
    label_file_name = label_dir+label_file+'-label.txt'
    #print(label_file_name)  
    
    with open(label_file_name) as handle:
      #Read extra line that says Frames Phases
      handle.readline()
      labels = handle.readlines()
   
    #print(len(image_files))
    local_image_files = list()
    for label in labels:
      label_split = label.split('\t')
      index = str(int(label_split[0])+1)
     
      image_file = label_file+'-'+index+'.jpg'

      local_image_files.append(label_file+'/'+image_file)
      
    #print(len(local_image_files), len(labels))
    feature_list.extend([local_image_files[i], labels[i]] for i in range(len(labels)))
  
  return(feature_list)

  
def generate_feature_eval_list(image_dir, label_dir, video_files):
  
  label_files =  video_files
  #print(label_files)
  feature_list = list()
  
  for label_file in label_files:
    label_file_name = label_dir+label_file+'-label.txt'
    #print(label_file_name)  
    
    with open(label_file_name) as handle:
      #Read extra line that says Frames Phases
      handle.readline()
      labels = handle.readlines()
   
    #print(len(image_files))
    local_image_files = list()
    for label in labels:
      label_split = label.split('\t')
      index = str(int(label_split[0])+1)
     
      image_file = label_file+'-'+index+'.jpg'

      local_image_files.append(label_file+'/'+image_file)
      
    #print(len(local_image_files), len(labels))
    feature_list.extend([local_image_files[i], labels[i]] for i in range(len(labels)))
    
  return(feature_list)
