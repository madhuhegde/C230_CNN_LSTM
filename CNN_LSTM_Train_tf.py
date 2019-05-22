import numpy as np
import os
import time
import tensorflow
import matplotlib
import json, pickle
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from LossHistory import LossHistory
from keras.utils import plot_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


from CNN_LSTM_load_data import  generator_train, generator_test
from CNN_LSTM_split_data import generate_feature_train_list, generate_feature_test_list

os.environ['KMP_DUPLICATE_LIB_OK']='True'

base_dir = "data/"
model_save_dir = "data/model"
log_dir="logs/"
history_dir = "data/history"
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
frames = 2    #Number of frames over which LSTM prediction happens
channels = 3  #RGB
rows = 224    
columns = 224 

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

#Use pretrained VGG16 
video = Input(shape=(frames,rows,columns,channels))
cnn_base = VGG16(input_shape=(rows,columns,channels),
                 weights="imagenet",
                 #weights = None, 
                 include_top=False)
                             

cnn_out = GlobalAveragePooling2D()(cnn_base.output)

cnn = Model(inputs=cnn_base.input, outputs=cnn_out)

#cnn.trainable = True

#Use Transfer learning and train only last 4 layers                 
for layer in cnn.layers[:-10]:
    layer.trainable = False


cnn.summary()

for layer in cnn.layers:
   print(layer.trainable)

#Build LSTM network
encoded_frames = TimeDistributed(cnn)(video)
encoded_sequence = LSTM(512)(encoded_frames)

# RELU or tanh?
hidden_layer = Dense(units=512, activation="relu")(encoded_sequence)
#hidden_layer = Dense(units=512, activation="tanh")(encoded_sequence)

dropout_layer = Dropout(rate=0.5)(hidden_layer)
outputs = Dense(units=num_classes, activation="softmax")(dropout_layer)
model = Model([video], outputs)

model.summary()

#Similar to Adam
optimizer = Nadam(lr=0.00001,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)

#softmax crossentropy
model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["categorical_accuracy"]) 

#training parameters
BATCH_SIZE = 8 # Need GPU with 32 GB RAM for BATCH_SIZE > 16
nb_epochs = 10 # 


#generate indices for train_array an test_array with train_test_split_ratio = 0.


train_samples  = generate_feature_train_list(train_image_dir, train_label_dir)
validation_samples = generate_feature_test_list(test_image_dir, test_label_dir)
#validation_samples = validation_samples[0:60*32*5]
train_len = int(len(train_samples)/(BATCH_SIZE*frames))
train_len = (train_len)*BATCH_SIZE*frames
train_samples = train_samples[0:train_len]
validation_len = int(len(validation_samples)/(BATCH_SIZE*frames))
validation_len = (validation_len-2)*BATCH_SIZE*frames
validation_samples = validation_samples[0:validation_len]
#print (train_len, validation_len)

#define callback functions
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=2),
             ModelCheckpoint(filepath=model_save_dir+'best_model.h5', monitor='val_loss',
             save_best_only=True),
             TensorBoard(log_dir='./logs/Graph', histogram_freq=0, write_graph=True, write_images=True)]

# load training data
train_generator = generator_train(train_samples, batch_size=BATCH_SIZE, frames_per_clip=frames,shuffle=True)
validation_generator = generator_test(validation_samples, batch_size=BATCH_SIZE, frames_per_clip=frames, shuffle=False)

history = model.fit_generator(train_generator, 
            steps_per_epoch=int(len(train_samples)/(BATCH_SIZE*frames)), 
            validation_data=validation_generator, 
            validation_steps=int(len(validation_samples)/(BATCH_SIZE*frames)), 
            #callbacks = [history],
            callbacks = callbacks,
            epochs=nb_epochs, verbose=1)

plot_model(model, to_file='./logs/model.png', show_shapes=True)
logfile = open('./logs/losses.txt', 'wt')
logfile.write('\n'.join(str(l) for l in history.history['loss']))
logfile.close()
                        
#history.key() = ['loss', 'categorical_accuracy', 'val_loss', 'val_categorical_accuracy'])
print(history.history['loss'])

#dump history
#json.dump(history.history, open(history_dir+'model_history', 'w'))
with open(history_dir+'model_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
#print(history.val_acc)
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.show()

#save model and clear session
del model
tensorflow.keras.backend.clear_session()



