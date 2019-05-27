import numpy as np
import os
import time
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
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib

from CNN_LSTM_load_data import  generator_train, generator_test
from CNN_LSTM_split_data import generate_feature_train_list, generate_feature_test_list

os.environ['KMP_DUPLICATE_LIB_OK']='True'

config = json.load(open('config/config.json'))
base_dir = config['base_dir']
model_save_dir = config["model_save_dir"]
history_dir = config["history_dir"]
model_type = config["model_type"]

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

# Dimensions of input feature 
frames = 25    #Number of frames over which LSTM prediction happens
channels = 3  #RGB
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
        
        
def get_VGG16_base():

  #Use pretrained VGG16 
  cnn_base = VGG16(input_shape=(rows,columns,channels),
                 weights="imagenet",
                 #weights = None, 
                 include_top=False)
  cnn_out = GlobalAveragePooling2D()(cnn_base.output)

  cnn_model = Model(inputs=cnn_base.input, outputs=cnn_out)

  #Use Transfer learning and train last 15 layers                 
  for layer in cnn_model.layers[:-15]:
    layer.trainable = False

  for layer in cnn_model.layers:
    print(layer.trainable)
   
  return(cnn_model)         
  
def get_LSTM_model(input, base_model):

  #Build LSTM network
  encoded_frames = TimeDistributed(base_model)(input)
  encoded_sequence = LSTM(2048, name='lstm1')(encoded_frames)

  # RELU or tanh?
  hidden_layer = Dense(units=2048, activation="relu")(encoded_sequence)
  dropout_layer = Dropout(rate=0.5)(hidden_layer)
  outputs = Dense(units=num_classes, activation="softmax")(dropout_layer)
  
  #create model CNN+LSTM
  l_model = Model(input, outputs) 
  
  return(l_model)
  
  
  
def get_stacked_LSTM_model(input, base_model):

  #Build LSTM network
  encoded_frames = TimeDistributed(base_model)(input)
  first_LSTM_layer = LSTM(1024, return_sequences=True, name='lstm1')(encoded_frames)
  dropout_layer_1 = Dropout(rate=0.3) (first_LSTM_layer)
  # RELU or tanh?
  #hidden_layer = Dense(units=1024, activation="relu")(encoded_sequence)
  second_LSTM_layer = LSTM(1024, name='lstm2')(dropout_layer_1)

  dropout_layer_2 = Dropout(rate=0.3)(second_LSTM_layer)
  outputs = Dense(units=num_classes, activation="softmax")(dropout_layer_2)
  
  #create model CNN+LSTM
  l_model = Model(input, outputs) 
  
  return(l_model)  
  
  
# Function pointers for models
  
cnn_func_ptr = {
 'VGG16_NORM_LSTM' : get_VGG16_base,
 'VGG16_STACKED_LSTM' : get_VGG16_base
 
}     
  
lstm_func_ptr = {
 'VGG16_NORM_LSTM' : get_LSTM_model,
 'VGG16_STACKED_LSTM' : get_stacked_LSTM_model
 
}   
  
# main function   
        
if __name__ == "__main__":        

  # Define Input type
  video = Input(shape=(frames,rows,columns,channels))
  
  cnn_model = cnn_func_ptr[model_type]()
  
  l_model = lstm_func_ptr[model_type](video, cnn_model)
  
  
  
  #get number of GPUs
  num_gpus = get_available_gpus()
  
  print(num_gpus)
  #GPU Optimization
  #disable GPU Optimization until accuracy issues are resolved
  num_gpus = []
  if(len(num_gpus)>0):
    num_gpus = len(num_gpus)
    lstm_model = multi_gpu_model(l_model, 
                             gpus=num_gpus,
                             cpu_merge=True,
                             cpu_relocation=True)
  else:
    lstm_model = l_model
                                
  lstm_model.summary()

  #Similar to Adam
  optimizer = Nadam(lr=0.00001,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)

  #softmax crossentropy
  lstm_model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["categorical_accuracy"]) 

  train_samples  = generate_feature_train_list(train_image_dir, train_label_dir)
  validation_samples = generate_feature_test_list(test_image_dir, test_label_dir)
  train_len = int(len(train_samples)/(BATCH_SIZE*frames))
  train_len = (train_len)*BATCH_SIZE*frames
  train_samples = train_samples[0:train_len]
  validation_len = int(len(validation_samples)/(BATCH_SIZE*frames))
  validation_len = (validation_len)*BATCH_SIZE*frames
  validation_samples = validation_samples[0:validation_len]
  print (train_len, validation_len)

  saveCNN_Model = CNN_LSTM_ModelCheckpoint(cnn_model, model_save_dir+"cnn_model.h5",
                                    l_model, model_save_dir+"lstm_model.h5")

  #define callback functions
  history = History()
  callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=2),
               #ModelCheckpoint(filepath=model_save_dir+'best_model.h5', monitor='val_loss',
               #save_best_only=True),
               history,
               saveCNN_Model]
 #             TensorBoard(log_dir='./logs/Graph', histogram_freq=0, write_graph=True, write_images=True)]

  # load training data
  train_generator = generator_train(train_samples, batch_size=BATCH_SIZE, frames_per_clip=frames,shuffle=True)
  validation_generator = generator_test(validation_samples, batch_size=BATCH_SIZE, frames_per_clip=frames, shuffle=False)

  lstm_model.fit_generator(train_generator, 
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
  history_dict = {}
  history_dict['val_loss'] = history.val_loss
  history_dict['train_loss'] = history.train_loss
  history_dict['train_acc'] = history.train_acc
  history_dict['val_acc'] = history.val_acc

  #json.dump(history.history, open(history_dir+'model_history', 'w'))
  with open(history_dir+'model_history', 'wb') as file_pi:
        pickle.dump(history_dict, file_pi)
        
#print(history.val_acc)
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.show()

#save model and clear session
  del cnn_model
  del lstm_model
  tensorflow.keras.backend.clear_session()



