import numpy as np
import os
import time
import random
from CNN_LSTM_load_data import  generator_CNN_train, generator_CNN_test
from CNN_LSTM_split_data import generate_feature_train_list, generate_feature_test_list
import tensorflow
import matplotlib
import json, pickle
#matplotlib.use("TkAgg")
import pdb
from matplotlib import pyplot as plt

from LossHistory import LossHistory
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib

os.environ['KMP_DUPLICATE_LIB_OK']='True'

config = json.load(open('config/config.json'))
base_dir = config['base_dir']
model_save_dir = config["model_save_dir"]
history_dir = config["history_dir"]

base_image_dir = base_dir+"images/"
base_label_dir = base_dir+"labels/"
test_image_dir = base_image_dir + "test/"
test_label_dir = base_label_dir + "test/"
train_image_dir = base_image_dir + "train/"
train_label_dir = base_label_dir + "train/"

train_videos = ['video02', 'video04', 'video05', 'video10', 'video11','video12', 'video13', 'video14', 
                'video15', video17']
				
aug_videos = ['video36', 'video37', 'video41', 'video43', 'video48','video49', 'video50', 'video51', 
                'video53', video60', 'video61', 'video65']		
				
test_videos = ['video06', 'video16', 'video20', 'video23', 'video27', 'video31', 'video33', 'video35', 
               'video44', 'video45', 'video47', 'video55', 'video57']



# 7 phases for surgical operation
class_labels = {"Preparation":0, "CalotTriangleDissection":1, "ClippingCutting":2, 
           "GallbladderDissection":3, "GallbladderPackaging":4, "CleaningCoagulation":5, "GallbladderRetraction":6}


num_classes = 7

# just rename some varialbes
frames = 1 #redundant variable. Never change it other than 1
channels = 3
rows = 224
columns = 224 
BATCH_SIZE = 8
nb_epochs = 2

# Define callback function if detailed log required
class History(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_loss = []
        self.train_acc = []
        self.val_acc = []
        self.val_loss = []

    def on_batch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.train_acc.append(logs.get('categorical_accuracy'))
        
    def on_epoch_end(self, batch, logs={}):    
        self.val_acc.append(logs.get('val_categorical_accuracy'))
        self.val_loss.append(logs.get('val_loss'))
        
# Implement ModelCheckPoint callback function to save CNN model
class CNN_LSTM_ModelCheckpoint(tensorflow.keras.callbacks.Callback):

    def __init__(self,cnn_model, cnn_filename, lstm_model=None, lstm_filename=None):
        self.cnn_filename = cnn_filename
        self.cnn_model = cnn_model
        self.lstm_filename = lstm_filename
        self.lstm_model = lstm_model

    def on_train_begin(self, logs={}):
        self.max_val_acc = 0
        
 
    def on_epoch_end(self, batch, logs={}):    
        val_acc = logs.get('val_categorical_accuracy')
        if(val_acc > self.max_val_acc):
           self.max_val_acc = val_acc
           self.cnn_model.save(self.cnn_filename) 
           if(self.lstm_model):
              self.lstm_model.save(self.lstm_filename)


def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_VGG16_only_model():

  #Use pretrained VGG16 
  cnn_base = VGG16(input_shape=(rows,columns,channels), weights='imagenet', include_top=False)

  #Add the fully-connected layers 
  x = cnn_base.output
  x = Flatten(name='flatten')(x)
  x = Dense(4096, activation='relu', name='fc1')(x)
  x = Dropout(0.5)(x)
  x = Dense(1024, activation='relu', name='fc2')(x)
  x = Dropout(0.2)(x)
  x = Dense(num_classes, activation='softmax', name='predictions')(x)

  #Create your own model 
  cnn_model = Model(inputs=cnn_base.input, outputs=x)

  for layer in cnn_base.layers: #[:-13]:
    layer.trainable = True #False
    
  for layer in cnn_base.layers:  
    print(layer.trainable)
    
  return(cnn_model) 


if __name__ == "__main__":

  cnn_model = get_VGG16_only_model()
  
  num_gpus = get_available_gpus()
  num_gpus = []
  
  #GPU Optimization
  if(len(num_gpus)>0):
    num_gpus = len(num_gpus)
    gpu_model = multi_gpu_model(cnn_model, 
                             gpus=num_gpus,
                             cpu_merge=True,
                             cpu_relocation=True)
  else:
    gpu_model = cnn_model
    
  cnn_model.summary()

  #Similar to Adam
  optimizer = Nadam(lr=0.00001,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)

  #softmax crossentropy
  gpu_model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["categorical_accuracy"]) 




  train_samples  = generate_feature_train_list(train_image_dir, train_label_dir, train_videos)
  aug_samples  = generate_feature_augment_list(train_image_dir, train_label_dir, aug_videos)
  print(len(train_samples), len(aug_samples))
  train_samples.extend(aug_samples)
  print(len(train_samples))
  validation_samples = generate_feature_test_list(test_image_dir, test_label_dir, test_videos)
  #validation_samples = validation_samples[0:60*32*5]
  train_len = int(len(train_samples)/(BATCH_SIZE*frames))
  train_len = (train_len)*BATCH_SIZE*frames
  train_samples = train_samples[0:train_len]
  validation_len = int(len(validation_samples)/(BATCH_SIZE*frames))
  validation_len = (validation_len-2)*BATCH_SIZE*frames
  validation_samples = validation_samples[0:validation_len]
  print (train_len, validation_len)

  saveCNN_Model = CNN_LSTM_ModelCheckpoint(cnn_model, model_save_dir+"vgg16_model.h5")


  #define callback functions
  history = History()
  callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=2),
             #ModelCheckpoint(filepath=model_save_dir+'best_model.h5', monitor='val_loss',
             #save_best_only=True),
             history,
             saveCNN_Model]
 #            TensorBoard(log_dir='./logs/Graph', histogram_freq=0, write_graph=True, write_images=True)]

  # load training data
  train_generator = generator_CNN_train(train_samples, batch_size=BATCH_SIZE, frames_per_clip=1, shuffle=True)
  validation_generator = generator_CNN_test(validation_samples, batch_size=BATCH_SIZE, frames_per_clip=1, shuffle=False)

  gpu_model.fit_generator(train_generator, 
            steps_per_epoch=int(len(train_samples)/(BATCH_SIZE*frames)), 
            validation_data=validation_generator, 
            validation_steps=int(len(validation_samples)/(BATCH_SIZE*frames)), 
            #callbacks = [history],
            callbacks = callbacks,
            epochs=nb_epochs, verbose=1)

  #plot_model(model, to_file='./logs/model.png', show_shapes=True)
  logfile = open('./logs/losses.txt', 'wt')
  logfile.write('\n'.join(str(l) for l in history.val_loss))
  logfile.close()
                        
  #history.key() = ['loss', 'categorical_accuracy', 'val_loss', 'val_categorical_accuracy'])
  #print(history.history['loss'])
  history_dict = {}
  history_dict['val_loss'] = history.val_loss
  history_dict['train_loss'] = history.train_loss
  history_dict['train_acc'] = history.train_acc
  history_dict['val_acc'] = history.val_acc

  #json.dump(history.history, open(history_dir+'model_history', 'w'))
  with open(history_dir+'vgg16_model_history', 'wb') as file_pi:
        pickle.dump(history_dict, file_pi)
        

  #delete model and clear session
  del gpu_model
  del cnn_model
  tensorflow.keras.backend.clear_session()



