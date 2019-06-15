import numpy as np
import os
import time
import tensorflow
import matplotlib
import json, pickle
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pdb

from LossHistory import LossHistory
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Reshape
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model

from CNN_LSTM_load_data import  generator_train, generator_test
from CNN_LSTM_split_data import generate_feature_train_list, generate_feature_test_list
from CNN_LSTM_split_data import generate_feature_augment_list, remove_transition_samples


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


				
		   
test_videos = ['video04',  'video12', 'video16', 'video17', 'video24', 'video27', 'video36', 'video40', 'video44','video49']
aug_videos = ['video11', 'video15',  'video18', 'video21', 'video22', 'video23', 'video26',
               'video25', 'video28', 'video30', 'video31',  'video34', 'video35', 'video37', 'video39',
               'video42', 'video43',  'video45', 'video48', 'video50', 'video51', 'video52', 'video57',  'video60', 'video66',  
	           'video67', 'video72']

train_videos =  ['video01', 'video02', 'video05', 'video08', 'video09', 'video12','video14', 'video41','video46', 'video47', 'video61', 'video64'] 			   

train_videos = ['video41', 'video64', 'video43', 'video04', 'video57', 'video08', 'video58', 'video10', 'video45', 'video05']
aug_videos =   ['video27', 'video18', 'video30', 'video02', 'video13', 'video51', 'video59', 'video34', 'video33', 'video14', 
                'video11', 'video15', 'video48', 'video09', 'video17', 'video46', 'video07', 'video49', 'video06', 'video67', 
				'video23', 'video01', 'video66', 'video61', 'video03', 'video32', 'video16', 'video22', 'video72', 'video69']

train_videos =  ['video01', 'video02', 'video05', 'video08', 'video09', 'video12','video14', 'video41','video46', 'video47', 'video61', 'video64']
 
train_videos = ['video41', 'video64', 'video43', 'video04', 'video57', 'video08', 'video58', 'video10', 'video45', 'video05']
aug_videos =   ['video27', 'video18', 'video30', 'video02', 'video13', 'video51', 'video59', 'video34', 'video33', 'video14',
                'video11', 'video15', 'video48', 'video09', 'video17', 'video46', 'video07', 'video49', 'video06', 'video67',
                'video23', 'video01', 'video66', 'video61', 'video03', 'video32', 'video16', 'video22', 'video72', 'video69']
# 7 phases for surgical operation
class_labels = {"Preparation":0, "CalotTriangleDissection":1, "ClippingCutting":2, 
           "GallbladderDissection":3, "GallbladderPackaging":4, "CleaningCoagulation":5, "GallbladderRetraction":6}

class_labels = {"Preparation":0,  "CleaningCoagulation":1, "GallbladderRetraction":2}           


num_classes = len(class_labels)

# Dimensions of input feature 
frames = 25    #Number of frames over which LSTM prediction happens
channels = 3  #RGB
rows = 224    
columns = 224 

#training parameters
BATCH_SIZE = 8# Need GPU with 32 GB RAM for BATCH_SIZE > 16
nb_epochs = 3 # 

lstm_model = None
class_weights = [2,2,1,2,2,1,1]
# Compute class_weights for imbalanced train set
def compute_class_weight(input_list):

  label_list = []
  for label in input_list:
    label = label[1].split('\t')[1].strip()
    label_list.append(label)
  
  class_weights = class_weight.compute_class_weight('balanced', 
                                                   np.unique(label_list),  
                                                   label_list)
                                                   
  return(class_weights)                                                 

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
        
    #def on_epoch_end(self, batch, logs={}):    
        self.val_acc.append(logs.get('val_categorical_accuracy'))
        self.val_loss.append(logs.get('val_loss'))
        
# Implement ModelCheckPoint callback function to save CNN model
class CNN_LSTM_ModelCheckpoint(tensorflow.keras.callbacks.Callback):

    def __init__(self,cnn_model, cnn_filename, lstm_model, lstm_filename):
        #self.cnn_filename = cnn_filename
        #self.cnn_model = cnn_model
        self.lstm_filename = lstm_filename
        self.lstm_model = lstm_model

    def on_train_begin(self, logs={}):
        self.max_val_acc = 0
        
 
    def on_epoch_end(self, batch, logs={}):    
        val_acc = logs.get('val_categorical_accuracy')
        if(val_acc > self.max_val_acc):
           self.max_val_acc = val_acc
           #self.cnn_model.save(self.cnn_filename) 
           self.lstm_model.save(self.lstm_filename)
          
 
 
def train_stateful_lstm(train_videos, test_videos, callbacks):

  print(train_videos)
  train_samples  = generate_feature_augment_list(train_image_dir, train_label_dir, train_videos)
  
  train_samples = remove_transition_samples(train_samples, frames)
  print(len(train_samples))
  #class_weights = compute_class_weight(train_samples)
  

  validation_samples = generate_feature_augment_list(test_image_dir, test_label_dir, test_videos)
  
  validation_samples = remove_transition_samples(validation_samples, frames)
  
  train_len = int(len(train_samples)/(BATCH_SIZE*frames))
  train_len = (train_len)*BATCH_SIZE*frames
  train_samples = train_samples[0:train_len]
  validation_len = int(len(validation_samples)/(BATCH_SIZE*frames))
  validation_len = (validation_len)*BATCH_SIZE*frames
  validation_samples = validation_samples[0:validation_len]
  print (train_len, validation_len)




  # load training data
  train_generator = generator_train(train_samples, batch_size=BATCH_SIZE, frames_per_clip=frames,shuffle=False)
  validation_generator = generator_test(validation_samples, batch_size=BATCH_SIZE, frames_per_clip=frames, shuffle=False)

  lstm_model.fit_generator(train_generator, 
            steps_per_epoch=int(len(train_samples)/(BATCH_SIZE*frames)), 
            validation_data=validation_generator, 
            validation_steps=int(len(validation_samples)/(BATCH_SIZE*frames)), 
            #class_weight = class_weights,
            callbacks = callbacks,
            epochs=nb_epochs, verbose=1)
  
  
  return    


if __name__=="__main__":
  
  #Define Input with batch_shape to train stateful LSTM  
  video = Input(shape=(frames,rows,columns,channels))

  #load lstm_model with shuffled data
  prev_lstm_model = load_model(model_save_dir+'lstm_model.h5')
  lstm_weights = list()

  #load pretrained weights
  for layer in prev_lstm_model.layers:
     weights = layer.get_weights()
     lstm_weights.append(weights)

  # print summary and clear model    
  prev_lstm_model.summary
  del prev_lstm_model

  #load pre-trained cnn model
  cnn_model = load_model(model_save_dir+'cnn_model.h5')

  #freeze cnn weights for stateful LSTM
  for layer in cnn_model.layers:
    layer.trainable = False

  #LSTM model must match stateless LSTM
  encoded_frames = TimeDistributed(cnn_model)(video)
  encoded_sequence = LSTM(1024, stateful=False, name='lstm1')(encoded_frames)

  # RELU or tanh?
  hidden_layer = Dense(units=1024, activation="relu")(encoded_sequence)


  dropout_layer = Dropout(rate=0.5)(hidden_layer)
  outputs = Dense(units=num_classes, activation="softmax")(dropout_layer)
  l_model = Model(video, outputs)

  for i in range(len(l_model.layers)):
    l_model.layers[i].set_weights(lstm_weights[i])

  lstm_model = l_model
  #del l_model

  #Similar to Adam 
  optimizer = Nadam(lr=0.0001,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)

  #softmax crossentropy
  lstm_model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["categorical_accuracy"]) 

#define callback functions
  
  saveCNN_Model = CNN_LSTM_ModelCheckpoint(cnn_model, model_save_dir+"cnn_model_notsaved.h5",
                                    lstm_model, model_save_dir+"lstm_stateless_model.h5")
  history = History()
  callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=2),
               #ModelCheckpoint(filepath=model_save_dir+'best_model.h5', monitor='val_loss',
               #save_best_only=True),
               history,
               saveCNN_Model]
 #             TensorBoard(log_dir='./logs/Graph', histogram_freq=0, write_graph=True, write_images=True)]
 
  
  for video in aug_videos:
    train_video = [video]
    train_stateful_lstm(train_video, test_videos, callbacks)
    lstm_model.reset_states()

#history.key() = ['loss', 'categorical_accuracy', 'val_loss', 'val_categorical_accuracy'])

  history_dict = {}
  history_dict['val_loss'] = history.val_loss
  history_dict['train_loss'] = history.train_loss
  history_dict['train_acc'] = history.train_acc
  history_dict['val_acc'] = history.val_acc

  with open(history_dir+'sl_lstm_model_history', 'wb') as file_pi:
        pickle.dump(history_dict, file_pi)
        
#save model and clear session
  del lstm_model, cnn_model
  tensorflow.keras.backend.clear_session()

  
