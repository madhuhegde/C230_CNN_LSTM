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
from CNN_LSTM_split_data import remove_transition_samples
from surgical_flow_model import initialize_trans_matrix, predict_next_label
import pickle

eval_videos = [['video04'],  ['video12'], ['video17'], ['video24'], ['video36'], ['video40'], ['video53']]

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
frames = 25 #args.frames    #Number of frames over which LSTM prediction happens
channels = 3  #RGB
rows = 224    
columns = 224 
BATCH_SIZE = 8 #args.batch_size



# return y, yhat
def evaluate_model(lstm_model, eval_videos, callbacks):
  
  validation_samples = generate_feature_eval_list(eval_image_dir, eval_label_dir, eval_videos)
  #validation_samples = remove_transition_samples(validation_samples, frames)
  validation_len = 4 #int(len(validation_samples)/(BATCH_SIZE*frames))
  validation_len = (validation_len-2)*BATCH_SIZE*frames
  validation_samples = validation_samples[0:validation_len]
  print ("Validatation Length:{0}".format(validation_len))

  y = [1]
  yhat = [1]
  return(y,yhat)
  

#model_callbacks = [TensorBoard(log_dir='./logs/Graph', histogram_freq=5, write_graph=True, write_images=True, write_grads=True)]
#lstm_model.summary()

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

  yhat = np.argmax(yhat, axis=1)
  y =  np.argmax(y, axis=1)
  
  return(y, yhat)
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default='model/models/final_model.h5',
                            help="Path to the model file with weights. e.g. model/models/final_model.h5")
  parser.add_argument('--frames', default=25,
                            help="Number of frames per sequence. e.g. 15. for a 5 sec clip of 3fps video.")
  parser.add_argument('--batch_size', default=8,
                            help="batch size")


  args = parser.parse_args()
        

# Dimensions of input feature 
  frames = args.frames    #Number of frames over which LSTM prediction happens

  BATCH_SIZE = args.batch_size 

  #initialize_trans_matrix()



# load weights into new model
  print("Loading model from {0}".format(args.model))
  lstm_model = load_model(args.model)
  lstm_model.summary()
  callbacks = []
  y = []
  yhat = []
  
  
  for i in range(len(eval_videos)):
     eval_video = eval_videos[i]
     #eval_video must be a list
     [y_video, yhat_video] = evaluate_model(lstm_model, eval_video, callbacks)
     y.extend(y_video)
     yhat.extend(yhat_video)

 # predfile = open('./logs/predictions_Yhat.txt', 'wt')
 # predfile.write('\r\n'.join(str(p).strip() for p in yhat))
  #predfile.close()

#  validationfile = open('./logs/input_X.txt', 'wt')
#  validationfile.write('\r\n'.join(str(s) for s in validation_samples))
#  validationfile.close()

  #gtfile = open('./logs/gt_Y.txt', 'wt')
  #gtfile.write('\r\n'.join(str(s) for s in y))
  #gtfile.close()

  ytrue_labels = [class_labels[i] for i in y]
  yhat_labels = [class_labels[i] for i in yhat]

  #yhat_pred = predict_next_label(yhat_labels)
  #yhat = [class_labels_dict[i] for i in yhat_pred]
  y_save = [ytrue_labels, yhat_labels]
  with open('label_history', 'wb') as file_pi:
        pickle.dump(y_save, file_pi)

#print("ytrue_labels", ytrue_labels)
#print("ypred_labels", yhat_labels)
#print("ground truth", [class_labels[i] for i in y])
#print("predictions", [class_labels[i] for i in yhat])

  cm = confusion_matrix(y, yhat, labels = [0, 1, 2, 3, 4, 5, 6])
  print(cm)

  cr = classification_report(y, yhat, [0, 1, 2, 3, 4, 5, 6], class_labels)
  print(cr)

