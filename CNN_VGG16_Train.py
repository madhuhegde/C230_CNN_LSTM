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
nb_epochs = 10

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

    def __init__(self,cnn_model, cnn_filename, lstm_model, lstm_filename):
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
           self.lstm_model.save(self.lstm_filename)


def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

#Use pretrained VGG16 
cnn_base = VGG16(input_shape=(rows,columns,channels), weights='imagenet', include_top=False)
#cnn_base.summary()

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

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
num_gpus = get_available_gpus()
if(len(num_gpus)>0):
    num_gpus = len(num_gpus)
    gpu_model = multi_gpu_model(cnn_model, 
                             gpus=num_gpus,
                             cpu_merge=True,
                             cpu_relocation=True)
else:
    gpu_model = cnn_model
cnn_model.summary()


optimizer = Nadam(lr=0.00001,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)


gpu_model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["categorical_accuracy"]) 




train_samples  = generate_feature_train_list(train_image_dir, train_label_dir)
validation_samples = generate_feature_test_list(test_image_dir, test_label_dir)
#validation_samples = validation_samples[0:60*32*5]
train_len = int(len(train_samples)/(BATCH_SIZE*frames))
train_len = (train_len)*BATCH_SIZE*frames
train_samples = train_samples[0:train_len]
validation_len = int(len(validation_samples)/(BATCH_SIZE*frames))
validation_len = (validation_len-2)*BATCH_SIZE*frames
validation_samples = validation_samples[0:validation_len]
print ("Loading train data")
# load training data
train_generator = generator_CNN_train(train_samples, batch_size=BATCH_SIZE, frames_per_clip=1, shuffle=True)
validation_generator = generator_CNN_test(validation_samples, batch_size=BATCH_SIZE, frames_per_clip=1, shuffle=False)

gpu_model.fit_generator(train_generator, 
            steps_per_epoch=int(len(train_samples)/(BATCH_SIZE*frames)), 
            validation_data=validation_generator, 
            validation_steps=int(len(validation_samples)/(BATCH_SIZE*frames)), 
            epochs=nb_epochs, verbose=1)





