import numpy as np
import os
import time
from CNN_LSTM_load_data import load_cholec_data
from CNN_LSTM_split_data import split_cholec_data


from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam

os.environ['KMP_DUPLICATE_LIB_OK']='True'

base_dir = "/Users/madhuhegde/Downloads/cholec80/"
image_dir = base_dir+"images/"
label_dir = base_dir+"labels/"

class_labels = {"Preparation":0, "CalotTriangleDissection":1, "ClippingCutting":2, 
           "GallbladderDissection":3, "GallbladderPackaging":4, "CleaningCoagulation":5, "GallbladderRetraction":6}


num_classes = 7

# just rename some varialbes
frames = 4
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


#generate indices for train_array an test_array with train_test_split_ratio = 0.
train_test_split_ratio = 0.2
[train_array, test_array] = split_cholec_data(image_dir, label_dir, train_test_split_ratio)

print ("Loading train data")
# load training data

X_train, y_train = load_cholec_data(image_dir, label_dir, frames, train_array)


# NOTE: if you can't fit all data in memory, load a few users at a time and
# use multiple epochs. I don't recommend using one user at a time, since
# it prevents good shuffling.


#%%
# perform training
print("Training")
model.fit(np.array(np.array(X_train)), y_train, batch_size=batch_size, epochs=nb_epochs, shuffle=False, verbose=1)


# clean up the memory
X_train       = None
y_train       = None

print("Testing")

X_test, y_test = load_cholec_data(image_dir, label_dir, frames, test_array)

preds = model.predict(np.array(X_test))

confusion_matrix = np.zeros(shape=(y_test.shape[1],y_test.shape[1]))
accurate_count = 0.0
for i in range(0,len(preds)):
    # updating confusion matrix
    confusion_matrix[np.argmax(preds[i])][np.argmax(np.array(y_test[i]))] += 1

    # if you are not sure of the axes of the confusion matrix, try the following line
    #print ('Predicted: ', np.argmax(preds[i]), ', actual: ', np.argmax(np.array(y_val_one_hot[i])))

    # calculating overall accuracy
    if np.argmax(preds[i])==np.argmax(np.array(y_test[i])):
        accurate_count += 1

print('Validation accuracy: ', 100*accurate_count/len(preds)),' %'
print('Confusion matrix:')
print(class_labels)
print(confusion_matrix)





