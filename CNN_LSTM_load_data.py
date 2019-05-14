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


def load_cholec_data(image_dir, label_dir, frames_per_clip, array_index):

  #image_files = glob.glob(image_dir+"*.jpg")
  label_files = glob.glob(label_dir+"video*.txt")

  classes_array = list()
  data_array = list()

  for file_no, label_file in enumerate(label_files):
    with open(label_file) as handle:
      #Read extra line that says Frames Phases
      handle.readline()
      label_array = handle.readlines()
      
    label_file_name = label_file.split('/')[-1]
      
    image_folder = image_dir+label_file_name.replace('-label.txt', '')
    print(image_folder)
      
    #print(len(label_array), array_index[file_no].shape)
    array_label_index = array_index[file_no]
    
    for index in array_label_index[0]:
      line = label_array[index]
      class_label = class_labels[line.split('\t')[1]]
      classes_array.append(class_label)
        
    # Free up memory  
    label_array = None  
    #print(len(classes_array))
      
    image_files = glob.glob(image_folder+"/video*.jpg")
    image_files.sort(key=os.path.getmtime)
    
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
  
      

def generator(samples, batch_size=32, frames_per_clip=4):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            phases = []
            for batch_sample in batch_samples:
                image_file = batch_sample[0]
                #phase = batch_sample[1]
                phase = class_labels[batch_sample[1].split('\t')[1]]
                
                image = cv2.imread(image_file)
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
                #image = (image-128.0)/128.0;
                image = image/255.0
                
                images.append([image]*frames_per_clip)
                phases.append(phase)

            
            # trim image to only see section with road
            X_batch = np.array(images)
            classes_one_hot = np.zeros((len(phases), len(class_labels)))
            classes_one_hot[np.arange(len(phases)), phases] = 1  
            y_batch = classes_one_hot
            yield (X_batch, y_batch)      

