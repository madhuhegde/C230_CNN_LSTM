


import cv2
import io
import os
import subprocess
import glob
import numpy as np
import pickle
from CNN_LSTM_split_data import  generate_feature_train_list, generate_feature_augment_list

generate_feature_eval_list = generate_feature_train_list
generate_feature_test_list = generate_feature_train_list

from CNN_LSTM_load_data import  generator_train, generator_test
from CNN_LSTM_load_data import  generator_CNN_train, generator_CNN_test

class_labels = {"Preparation":0, "CalotTriangleDissection":1, "ClippingCutting":2, 
           "GallbladderDissection":3, "GallbladderPackaging":4, "CleaningCoagulation":5, "GallbladderRetraction":6}



#base_dir = "/home/madhu_hegde/cs230/data/cholec_mini_data/"
import json
config = json.load(open('config/config.json'))
base_dir = config['base_dir']
base_image_dir = base_dir+"images/"
base_label_dir = base_dir+"labels/"
test_image_dir = base_image_dir + "test/"
test_label_dir = base_label_dir + "test/"
train_image_dir = base_image_dir + "train/"
train_label_dir = base_label_dir + "train/"


test_videos = ['video01']
aug_videos = ['video05', 'video08', 'video09', 'video10', 'video12', 'video14', 'video21', 'video25',  'video43', 
            'video45', 'video48', 'video57', 'video64', 'video71']
train_videos =  ['video02', 'video04', 'video12', 'video17', 'video21', 'video24', 
                'video36', 'video40', 'video41','video51', 'video60', 'video65']




#save as [video][phase]
def generate_video_histogram(video_list):
   hist_dict = {}
   
   for list_item in video_list:
   
     #print(list_item)
     video_file = list_item[0].split('/')
     video_file = video_file[0]
     label = list_item[1].split('\t')
     label = label[1].strip()
     if video_file not in hist_dict:
       hist_dict[video_file] = {}
       
       
     video_dict = hist_dict[video_file]
     #print(video_dict)
     if label not in video_dict:
        video_dict[label] = 1
     else:
        video_dict[label] = video_dict[label] + 1
        
   return hist_dict
  
# save as [label][video]  
def generate_video_histogram_v2(video_list):
   hist_dict = {}
   
   for list_item in video_list:
   
     #print(list_item)
     video_file = list_item[0].split('/')
     video_file = video_file[0]
     label = list_item[1].split('\t')
     label = label[1].strip()
     if label not in hist_dict:
       hist_dict[label] = {}
       
       
     label_dict = hist_dict[label]
     #print(label_dict)
     if video_file not in label_dict:
        label_dict[video_file] = 1
     else:
        label_dict[video_file] = label_dict[video_file] + 1
        
   return hist_dict   


def verify_test_vector():
  test_list = generate_feature_test_list(test_image_dir, test_label_dir, test_vidoes)
  print(len(test_list))
  print(test_list[0])
  
  test_generator = generator_test(test_list, 4, 4, True)
  for i in range(int(len(test_list[0:128])/32)):
    [X, y] = test_generator.__next__()
    
    
  print (np.max(X[0]))
  return(test_list)
  
def verify_train_vector():
 
  train_list = generate_feature_train_list(train_image_dir, train_label_dir, train_videos)
  print(len(train_list))
  print(train_list[0])
  
  train_generator = generator_train(train_list, 4, 4, True)
  for i in range(int(len(train_list[0:128])/32)):
    [X, y] = train_generator.__next__()
    
  print (np.max(X[0]))

  print(X.shape, y.shape)
  return(train_list)
  

def verify_CNN_generator():
  train_list = generate_feature_train_list(train_image_dir, train_label_dir)
  train_generator = generator_CNN_train(train_list, 4, 1, shuffle=True)


  for i in range(int(len(train_list[0:128])/16)):
    [X, y] = train_generator.__next__()

  print(X.shape, y.shape)
  return  


#hist_dict = generate_video_histogram(train_list)

#with open('logs/video_histogram', 'wb') as file_pi:
#  pickle.dump(hist_dict,file_pi)
__name__ = "__main__"

train_list = generate_feature_train_list(train_image_dir, train_label_dir, train_videos)
print(len(train_list))
print(train_list[176:188])

#train_list = verify_train_vector()
#train_list = train_list[0:128]  
#train_generator = generator_train(train_list, 4, 4, True)


#hist_dict = generate_video_histogram(train_list)

#with open('logs/video_histogram', 'wb') as file_pi:
  #pickle.dump(hist_dict,file_pi)

#with open('logs/video_histogram', 'rb') as file_pi:
#  hist_dict= pickle.load(file_pi)
  
  
#for i in range(int(len(train_list)/16)):
#  [X, y] = train_generator.__next__()


#test_generator = generator_test(test_list, 4, 4)







