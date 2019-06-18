import cv2
import io
import os
import subprocess
import glob
import numpy as np
import random
import json
config = json.load(open('config/config.json'))
base_dir = config['base_dir']
base_image_dir = base_dir +"images/"
train_image_dir = base_image_dir+"train/"
test_image_dir = base_image_dir+"test/"
eval_image_dir = base_image_dir+"eval/"
#label_dir = base_dir+"labels/"


#class_labels = {"Preparation":0, "CalotTriangleDissection":1, "ClippingCutting":2, 
#           "GallbladderDissection":3, "GallbladderPackaging":4, "CleaningCoagulation":5, "GallbladderRetraction":6}
class_labels = {"Preparation":0, "CleaningCoagulation":1, "GallbladderRetraction":2}

              
def generator_train(samples, batch_size=32, frames_per_clip=4, shuffle=True):
   
    while 1: # Loop forever so the generator never terminates
        num_frames = int(len(samples)/frames_per_clip)
        shuffle_order = np.arange(num_frames)
        if(shuffle):
          random.shuffle(shuffle_order)
        
        frames_count = 0
        for offset in range(0, num_frames, batch_size):
            #batch_samples = samples[offset:offset+batch_size*frames_per_clip]
            
            print('\t',offset)

            images = []
            phases = []
            for i in range(batch_size):
              # Read only one label for each frames_per_clip
              batch_sample = samples[shuffle_order[frames_count]*frames_per_clip]
              
              phase = class_labels[batch_sample[1].split('\t')[1].strip()]
              phases.append(phase)
              
              consecutive_images = []
              #Read frame_per_clip images for every one label
              for j in range(frames_per_clip):
                
                batch_sample = samples[shuffle_order[frames_count]*frames_per_clip+j]
       
                flip = batch_sample[2]
                image_file = train_image_dir+batch_sample[0]
                
                image = cv2.imread(image_file)
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
                #image = (image-128.0)/128.0;
                image = image/255.0  
                
                #Data augmentation by Horizontal flip
                if(flip):
                  image = cv2.flip(image, 0)
                  
                consecutive_images.append(image)
                     
              images.append(consecutive_images)
              frames_count = frames_count +1
               
        
            X_batch = np.array(images)
            classes_one_hot = np.zeros((len(phases), len(class_labels)))
            classes_one_hot[np.arange(len(phases)), phases] = 1  
            y_batch = classes_one_hot
            yield (X_batch, y_batch)   
            
            
def generator_test(samples, batch_size=32, frames_per_clip=4, shuffle=False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        
        for offset in range(0, num_samples, batch_size*frames_per_clip):
            batch_samples = samples[offset:offset+batch_size*frames_per_clip]
    

            images = []
            phases = []
            for i in range(batch_size):
              # Read only one label for each frames_per_clip
              batch_sample = batch_samples[frames_per_clip*i]
              
              phase = class_labels[batch_sample[1].split('\t')[1].strip()]
              phases.append(phase)
              
              consecutive_images = []
              #Read frame_per_clip images for every one label
              for j in range(frames_per_clip):
                
                batch_sample = batch_samples[j+i*frames_per_clip]
                image_file = test_image_dir+batch_sample[0]
                
                image = cv2.imread(image_file)
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
                #image = (image-128.0)/128.0;
                image = image/255.0
                consecutive_images.append(image)
                    
              images.append(consecutive_images)
 
            X_batch = np.array(images)
            classes_one_hot = np.zeros((len(phases), len(class_labels)))
            classes_one_hot[np.arange(len(phases)), phases] = 1  
            y_batch = classes_one_hot
            yield (X_batch, y_batch)   
            
            
            
            
def generator_CNN_train(samples, batch_size=32, frames_per_clip=1, shuffle=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
    
        if(shuffle):
          random.shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            print('\t', offset, len(batch_samples))

            images = []
            phases = []
            for i in range(batch_size):
              # Read only one label for each frames_per_clip
              batch_sample = batch_samples[i]
              
              phase = class_labels[batch_sample[1].split('\t')[1].strip()]
              phases.append(phase)
              image_file = train_image_dir+batch_sample[0]
              image = cv2.imread(image_file)
              image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
                
              #image = (image-128.0)/128.0;
              image = image/255.0
              images.append(image)
               
        
            X_batch = np.array(images)
            classes_one_hot = np.zeros((len(phases), len(class_labels)))
            classes_one_hot[np.arange(len(phases)), phases] = 1  
            y_batch = classes_one_hot
            yield (X_batch, y_batch)   
            
            
def generator_CNN_test(samples, batch_size=32, frames_per_clip=1, shuffle=False):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        if(shuffle):
          random.shuffle(samples)
          
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
    

            images = []
            phases = []
            for i in range(batch_size):

              batch_sample = batch_samples[i]       
              phase = class_labels[batch_sample[1].split('\t')[1].strip()]
              phases.append(phase)
              image_file = test_image_dir+batch_sample[0]
              image = cv2.imread(image_file)
              image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
                
              #image = (image-128.0)/128.0;
              image = image/255.0
              images.append(image)
               
            X_batch = np.array(images)
            classes_one_hot = np.zeros((len(phases), len(class_labels)))
            classes_one_hot[np.arange(len(phases)), phases] = 1  
            y_batch = classes_one_hot
            yield (X_batch, y_batch)             

def generator_eval(samples, batch_size=32, frames_per_clip=4, shuffle=False):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        
        for offset in range(0, num_samples, batch_size*frames_per_clip):
            batch_samples = samples[offset:offset+batch_size*frames_per_clip]
    

            images = []
            phases = []
            for i in range(batch_size):
              # Read only one label for each frames_per_clip
              batch_sample = batch_samples[frames_per_clip*i]
              
              phase = class_labels[batch_sample[1].split('\t')[1].strip()]
              phases.append(phase)
              
              consecutive_images = []
              #Read frame_per_clip images for every one label
              for j in range(frames_per_clip):
                
                batch_sample = batch_samples[j+i*frames_per_clip]
                image_file = eval_image_dir+batch_sample[0]
                
                image = cv2.imread(image_file)
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
                #image = (image-128.0)/128.0;
                image = image/255.0
                consecutive_images.append(image)
               
                
              images.append(consecutive_images)
               
        
            X_batch = np.array(images)
            classes_one_hot = np.zeros((len(phases), len(class_labels)))
            classes_one_hot[np.arange(len(phases)), phases] = 1  
            y_batch = classes_one_hot
            yield (X_batch, y_batch)             
