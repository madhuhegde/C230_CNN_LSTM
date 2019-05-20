import numpy as np
import os
import time
import random
from CNN_LSTM_load_data import  generator_CNN_train, generator_CNN_test
from CNN_LSTM_split_data import generate_feature_train_list, generate_feature_test_list
import tensorflow

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Nadam, Adam

os.environ['KMP_DUPLICATE_LIB_OK']='True'

base_dir = "/home/madhu_hegde/cs230/data/cholec_mini_data/"
base_image_dir = base_dir+"images/"
base_label_dir = base_dir+"labels/"
test_image_dir = base_image_dir + "test/"
test_label_dir = base_label_dir + "test/"
train_image_dir = base_image_dir + "train/"
train_label_dir = base_label_dir + "train/"

class_labels = {"Preparation":0, "CalotTriangleDissection":1, "ClippingCutting":2, 
           "GallbladderDissection":3, "GallbladderPackaging":4, "CleaningCoagulation":5, "GallbladderRetraction":6}


num_classes = 7

# just rename some varialbes
channels = 3
rows = 224
columns = 224 

#Get back the convolutional part of a VGG network trained on ImageNet
cnn_base = VGG16(input_shape=(rows,columns,channels), weights='imagenet', include_top=False)
#cnn_base.summary()

#Create your own input format (here 3x200x200)
input_image = Input(shape=(rows, columns, channels),name = 'image_input')

#Use the generated model 
cnn = cnn_base(input_image)

#Add the fully-connected layers 
x = Flatten(name='flatten')(cnn)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dropout(0.2)(x)
x = Dense(num_classes, activation='softmax', name='predictions')(x)

#Create your own model 
model = Model(inputs=input_image, outputs=x)
for layer in model.layers[:-11]:
    layer.trainable = False

for layer in model.layers:
    print(layer.trainable)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
model.summary()


optimizer = Nadam(lr=0.002,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)


model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["categorical_accuracy"]) 


#training parameters
BATCH_SIZE = 4 # increase if your system can cope with more data
nb_epochs = 3 # 
frames = 1


train_samples  = generate_feature_train_list(train_image_dir, train_label_dir)
validation_samples = generate_feature_test_list(test_image_dir, test_label_dir)
random.shuffle(train_samples)
train_len = int(len(train_samples)/(BATCH_SIZE*frames))
train_len = (train_len-2)*BATCH_SIZE*frames
train_samples = train_samples[0:train_len]
validation_len = int(len(validation_samples)/(BATCH_SIZE*frames))
validation_len = (validation_len-2)*BATCH_SIZE*frames
validation_samples = validation_samples[0:validation_len]

print ("Start Train and Test data generators")
# load training data
train_generator = generator_CNN_train(train_samples, batch_size=BATCH_SIZE, frames_per_clip=1)
validation_generator = generator_CNN_test(validation_samples, batch_size=BATCH_SIZE, frames_per_clip=1)

model.fit_generator(train_generator, 
            steps_per_epoch=int(len(train_samples)/(BATCH_SIZE*frames)), 
            validation_data=validation_generator, 
            validation_steps=int(len(validation_samples)/(BATCH_SIZE*frames)), 
            epochs=nb_epochs, verbose=1)





