import json
import numpy as np
import tensorflow as tf 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
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
from keras.models import load_model
from CNN_LSTM_load_data import  generator_train, generator_test, generator_eval
from CNN_LSTM_split_data import generate_feature_train_list, generate_feature_test_list, generate_feature_eval_list

config = json.load(open('config/config.json'))
base_dir = config['base_dir']
history_dir = config["history_dir"]

base_image_dir = base_dir+"images/"
base_label_dir = base_dir+"labels/"
test_image_dir = base_image_dir + "test/"
test_label_dir = base_label_dir + "test/"
train_image_dir = base_image_dir + "train/"
train_label_dir = base_label_dir + "train/"
eval_image_dir = base_image_dir + "eval/"
eval_label_dir = base_label_dir + "eval/"


print(train_image_dir, train_label_dir) # Senthil Remove
print(eval_image_dir, eval_label_dir) # Senthil Remove

# 7 phases for surgical operation
class_labels = ["Preparation", "CalotTriangleDissection", "ClippingCutting", 
           "GallbladderDissection", "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"]


num_classes = 7

# Dimensions of input feature 
frames = 25    #Number of frames over which LSTM prediction happens
channels = 3  #RGB
rows = 224    
columns = 224 
BATCH_SIZE = 8
nb_epochs = 1

# load weights into new model
lstm_model = tf.keras.models.load_model("model/models/final_model.h5")
lstm_model.summary()

optimizer = Nadam(lr=0.00001,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)

#softmax crossentropy
lstm_model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["categorical_accuracy"]) 

validation_samples = generate_feature_eval_list(eval_image_dir, eval_label_dir)
validation_len = int(len(validation_samples)/(BATCH_SIZE*frames))
validation_len = (validation_len-2)*BATCH_SIZE*frames
validation_samples = validation_samples[0:validation_len]
print ("Validatation Length:", validation_len)

#define callback functions
#OuputImageCallback = CustomImageCallback( validation_generator, log_dir = './logs/Graph')

model_callbacks = [TensorBoard(log_dir='./logs/Graph', histogram_freq=5, write_graph=True, write_images=True, write_grads=True)]
lstm_model.summary()

# load data and predict
y = None
yhat = None
num_batches = int(len(validation_samples)/(BATCH_SIZE*frames))
num_batches = 2 # test code delete later
validation_generator = generator_eval(validation_samples, batch_size=BATCH_SIZE, frames_per_clip=frames, shuffle=False)
steps = 0
for x_batch, y_batch in validation_generator:
    steps+=1
    print("Input shape {0}, X-Length {1}, gt shape {2}, Y-Length {3}".format(np.shape(x_batch),len(x_batch[0]), np.shape(y_batch), len(y_batch)))
    if y is None:
        y = y_batch
    else: 
        y = np.concatenate((y, y_batch))

    # loss = lstm_model.evaluate_generator(validation_generator, steps=int(len(validation_samples)/(BATCH_SIZE*frames)), max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
    pred_batch = lstm_model.predict(x = x_batch, verbose=1)
    print(np.shape(pred_batch)) 
    if yhat is None:
        yhat = pred_batch
    else:
        yhat = np.concatenate((yhat, pred_batch))
    #yhat.append(pred_batch)
    if steps >= num_batches: break

print("yhat before",np.shape(yhat), yhat)
yhat = np.argmax(yhat, axis=1)
print("yhat after", np.shape(yhat), yhat)

y =  np.argmax(y, axis=1)

predfile = open('./logs/predictions_Yhat.txt', 'wt')
predfile.write('\r\n'.join(str(p).strip() for p in yhat))
predfile.close()

validationfile = open('./logs/input_X.txt', 'wt')
validationfile.write('\r\n'.join(str(s) for s in validation_samples))
validationfile.close()

gtfile = open('./logs/gt_Y.txt', 'wt')
gtfile.write('\r\n'.join(str(s) for s in y))
gtfile.close()
print("ground truth", [class_labels[i] for i in y])
print("predictions", [class_labels[i] for i in yhat])

cm = confusion_matrix(y, yhat, labels = [0, 1, 2, 3, 4, 5, 6])
print(cm)

cr = classification_report(y, yhat, [0, 1, 2, 3, 4, 5, 6], class_labels)
print(cr)

