import numpy as np
import os
import time
from CH_load_data import load_cholec_data

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam


base_dir = "/Users/madhuhegde/Downloads/cholec80/"
image_dir = base_dir+"images/"
label_dir = base_dir+"labels/"


num_classes = 7

# just rename some varialbes
frames = 25
channels = 3
rows = 224
columns = 224 



video = Input(shape=(frames,rows,columns,channels))

cnn_base = VGG16(input_shape=(rows,columns,channels),
                 weights="imagenet",
                 include_top=False)

cnn_out = GlobalAveragePooling2D()(cnn_base.output)

cnn = Model(input=cnn_base.input, output=cnn_out)

cnn.trainable = False
#cnn.trainable = True


encoded_frames = TimeDistributed(cnn)(video)

#encoded_sequence = LSTM(256)(encoded_frames)
encoded_sequence = LSTM(80)(encoded_frames)

#hidden_layer = Dense(output_dim=1024, activation="relu")(encoded_sequence)
hidden_layer = Dense(units=80, activation="relu")(encoded_sequence)

outputs = Dense(units=num_classes, activation="softmax")(hidden_layer)

model = Model([video], outputs)

model.summary()

optimizer = Nadam(lr=0.002,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)


model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["categorical_accuracy"]) 





#%%

#training parameters
batch_size = 4 # increase if your system can cope with more data
nb_epochs = 2 # I once achieved 50% accuracy with 400 epochs. Feel free to change

print ("Loading data")
# load training data

X_train, y_train = load_cholec_data(image_dir, label_dir, frames)


# NOTE: if you can't fit all data in memory, load a few users at a time and
# use multiple epochs. I don't recommend using one user at a time, since
# it prevents good shuffling.


#%%
# perform training
print("Training")
model.fit(np.array(np.array(X_train)), y_train, batch_size=batch_size, epochs=nb_epochs, shuffle=False, verbose=1)




