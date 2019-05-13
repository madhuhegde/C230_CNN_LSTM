import cv2
import io
import os
import subprocess
import glob
import numpy as np
#base_dir = "/Users/madhuhegde/Downloads/cholec80/"
#image_dir = base_dir+"images/"
#label_dir = base_dir+"labels/"

class_labels = {"Preparation\n":0, "CalotTriangleDissection\n":1, "ClippingCutting\n":2, 
           "GallbladderDissection\n":3, "GallbladderPackaging\n":4, "CleaningCoagulation\n":5, "GallbladderRetraction\n":6}


def load_cholec_data(image_dir, label_dir, frames_per_clip, array_index):

  #image_files = glob.glob(image_dir+"*.jpg")
  label_files = glob.glob(label_dir+"*.txt")

  classes_array = list()
  data_array = list()

  for file_no, label_file in enumerate(label_files):
    with open(label_file) as handle:
      #Read extra line that says Frames Phases
      handle.readline()
      
      label_file_name = label_file.split('/')[-1]
      
      image_folder = image_dir+label_file_name.replace('-label.txt', '')
      print(image_folder)
      label_array = handle.readlines()
      #print(len(label_array), array_index[file_no].shape)
      array_label_index = array_index[file_no]
      for index in array_label_index[0]:
        
        line = label_array[index]
        class_label = class_labels[line.split('\t')[1]]
        classes_array.append(class_label)
        
      # Free up memory  
      label_array = None  
      #print(len(classes_array))
      
      image_files = glob.glob(image_folder+"/*.jpg")
      print(len(image_files))
      for index in array_label_index[0]:
        image_file = image_files[index]
        image = cv2.imread(image_file)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        #image = (image-128.0)/128.0;
        image = image/255.0
        data_array.append([image]*frames_per_clip)
      
      #Free up memory
      image_files = None
      
      
  #data_array = np.array(data_array)
  classes_one_hot = np.zeros((len(classes_array), len(class_labels)))
  classes_one_hot[np.arange(len(classes_array)), classes_array] = 1      
  #np.squeeze(data_array, axis=0)
  #print(np.max(data_array[:,:,:,1:100]), np.min(data_array[:,:,:,1:100]))
  #print(data_array.shape, classes_one_hot.shape)  
  return(data_array, classes_one_hot)    
  
      
      
      

