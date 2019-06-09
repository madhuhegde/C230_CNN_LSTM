import json
import numpy as np
import tensorflow as tf 
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='model/models/final_model.h5',
                            help="Path to the model file with weights. e.g. model/models/final_model.h5")
parser.add_argument('--frames', default=25,
                            help="Number of frames per sequence. e.g. 15. for a 5 sec clip of 3fps video.")
parser.add_argument('--batch_size', default=8,
                            help="batch size")

eval_videos =  ['video73', 'video77', 'video78', 'video79','video80'] 

args = parser.parse_args()

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


# 7 phases for surgical operation
class_labels = ["Preparation", "CalotTriangleDissection", "ClippingCutting", 
           "GallbladderDissection", "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"]
           
class_labels_dict = {"Preparation":0, "CalotTriangleDissection":1, "ClippingCutting":2, 
           "GallbladderDissection":3, "GallbladderPackaging":4, "CleaningCoagulation":5, "GallbladderRetraction":6}           

           

# Dimensions of input feature 
frames = args.frames    #Number of frames over which LSTM prediction happens
channels = 3  #RGB
rows = 224    
columns = 224 
BATCH_SIZE = args.batch_size

# load weights into new model
print("Loading model from {0}".format(args.model))
lstm_model = tf.keras.models.load_model(args.model)
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

#define callback functions
#OuputImageCallback = CustomImageCallback( validation_generator, log_dir = './logs/Graph')

model_callbacks = [TensorBoard(log_dir='./logs/Graph', histogram_freq=5, write_graph=True, write_images=False, write_grads=True)]
lstm_model.summary()

def predict_next_label(new_labels):
    out_labels = list()
    prev_label = "Preparation"
    label_history = "Preparation"
    for label in new_labels:
        if (label_history == label): # and (history[1]==new_label)): # and (history[2]==new_label)):
            if(prev_label != label):
                prev_label = label
        label_history = label
        out_labels.append(prev_label)
        
    return out_labels

def join_results(a, b):
    if a is None:
        a = b
    else:
        a = np.concatenate((a, b))
    return a

def evaluate_model(model, validation_samples):
    # load data and predict
    y = None
    yhat = None
    num_batches = int(len(validation_samples)/(BATCH_SIZE*frames))
    print("Input count: {0}, Batch count: {1}".format(len(validation_samples), num_batches))
    
    validation_generator = generator_eval(validation_samples, batch_size=BATCH_SIZE, frames_per_clip=frames, shuffle=False)
    steps = 0
    for x_batch, y_batch in validation_generator:
        steps+=1
        print("Batch# {0}, Input shape {1}, X-Length {2}, gt shape {3}, Y-Length {4}".format(steps, np.shape(x_batch),len(x_batch[0]), np.shape(y_batch), len(y_batch)))
        y = join_results(y, y_batch)
      # loss = lstm_model.evaluate_generator(validation_generator, steps=int(len(validation_samples)/(BATCH_SIZE*frames)), max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
        pred_batch = lstm_model.predict(x = x_batch, verbose=1)
        print(np.shape(pred_batch)) 
        yhat = join_results(yhat, pred_batch)
        #yhat.append(pred_batch)
        if steps >= num_batches: break
    return y, yhat

y = None
yhat = None
for video in eval_videos:
    validation_samples = generate_feature_eval_list(eval_image_dir, eval_label_dir, [video])
    validation_len = int(len(validation_samples)/(BATCH_SIZE*frames))
    validation_len = (validation_len-2)*BATCH_SIZE*frames
    validation_samples = validation_samples[0:validation_len]
    print ("Validatation Length:{0}".format(validation_len))
    yi, yhati = evaluate_model(lstm_model, validation_samples)
    
    yi = np.argmax(yi, axis=1)
    yhati = np.argmax(yhati, axis=1)
    
    y2 = [class_labels[j] for j in yhati]
    y2 = predict_next_label(y2)
    
    yhati = [class_labels_dict[j] for j in y2]
    
    y = join_results(y, yi)
    yhat = join_results(yhat, yhati)

#yhat = np.argmax(yhat, axis=1)
#y =  np.argmax(y, axis=1)

predfile = open('./logs/predictions_Yhat.txt', 'wt')
predfile.write('\r\n'.join(str(p).strip() for p in yhat))
predfile.close()

validationfile = open('./logs/input_X.txt', 'wt')
validationfile.write('\r\n'.join(str(s) for s in validation_samples))
validationfile.close()

gtfile = open('./logs/gt_Y.txt', 'wt')
gtfile.write('\r\n'.join(str(s) for s in y))
gtfile.close()

ytrue_labels = [class_labels[i] for i in y]
yhat_labels = [class_labels[i] for i in yhat]
print("ytrue_labels", ytrue_labels)
print("ypred_labels", yhat_labels)
#print("ground truth", [class_labels[i] for i in y])
#print("predictions", [class_labels[i] for i in yhat])

cm = confusion_matrix(y, yhat, labels = [0, 1, 2, 3, 4, 5, 6])
print(cm)

cr = classification_report(y, yhat, [0, 1, 2, 3, 4, 5, 6], class_labels)
print(cr)

