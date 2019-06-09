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
from tensorflow.keras.layers import  *
from CNN_LSTM_Train_tf import get_VGG16_base, get_LSTM_model
import test_frcnn as tf_cnn
'''
parser = argparse.ArgumentParser()

parser.add_argument('--model', default='lstm_model.h5',
                            help="Path to the model file with weights. e.g. model/models/final_model.h5")
parser.add_argument('--frames', default=25,
                            help="Number of frames per sequence. e.g. 15. for a 5 sec clip of 3fps video.")
parser.add_argument('--batch_size', default=1,
                            help="batch size")
args = parser.parse_args()
'''
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


cnn_model_path = './cnn_model.h5'
lstm_model_path = './lstm_model.h5'

# 7 phases for surgical operation
class_labels = ["Preparation", "CalotTriangleDissection", "ClippingCutting", 
           "GallbladderDissection", "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"]

# Dimensions of input feature 
frames = 25    #Number of frames over which LSTM prediction happens
channels = 3  #RGB
rows = 224    
columns = 224 
BATCH_SIZE = 1

num_classes = len(class_labels)
video = Input(shape=(frames,rows,columns,channels))
cnn_model = get_VGG16_base()
cnn_model = tf.keras.models.load_model(cnn_model_path)
cnn_model.load_weights(cnn_model_path)
l_model = get_LSTM_model(video, cnn_model)
l_model.load_weights(lstm_model_path)
l_model = tf.keras.models.load_model(lstm_model_path)
l_model.summary()
# load weights into new model
#print("Loading model from {0}".format(args.model))
#lstm_model = tf.keras.models.load_model(args.model)
input1 = Input(shape=(7,))
x1 =Flatten()(input1)
input2 = Input(shape=(7,))
x2 = Flatten()(input2)
# equivalent to added = keras.layers.add([x1, x2])
added = Add()([x1, x2])

#out = Dense(7)(added)
newModel = Model(inputs=[input1, input2], outputs=added)
newModel.summary()

'''
#lstm_model.name = 'test1'
x=Flatten()(lstm_model.output)
inputx = Input(shape=(7,))
x2 = Dense(7, activation='relu')(inputx)
mergedOut = Add()([x, x2])
#newModel = Model(lstm_model.input, x)
newModel = Model([inputx,lstm_model.input], mergedOut)
newModel.summary()
'''
'''
inputk = Input(shape=(7,))
input3 = Input(shape=(None,25,224,224,3))
x1 = lstm_model.output
x2 = Dense(7, activation='relu')(inputk)
mergedOut = Add()([x1, x2])
newModel = Model(inputs=[input3,inputk], outputs=mergedOut)
newModel.summary()
'''
'''
#input_3 = Input(shape=(lstm_model.output).shape())
x2 = Dense(7, activation='relu')(inputk)
#x1 = lstm_model.output
x1 = Dense(7, activation='relu')(lstm_model)
mergedOut = Add()([x1, x2])
newModel = Model(inputs=[input3,inputk], outputs=mergedOut)

newModel.summary()
'''
#lstm_model.summary()

optimizer = Nadam(lr=0.00001,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)
newModel.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["categorical_accuracy"])
#softmax crossentropy
l_model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["categorical_accuracy"]) 

validation_samples = generate_feature_eval_list(eval_image_dir, eval_label_dir)
validation_len = int(len(validation_samples)/(BATCH_SIZE*frames))
validation_len = (validation_len-2)*BATCH_SIZE*frames
validation_samples = validation_samples[0:validation_len]
print ("Validatation Length:{0}".format(validation_len))

#define callback functions
#OuputImageCallback = CustomImageCallback( validation_generator, log_dir = './logs/Graph')

model_callbacks = [TensorBoard(log_dir='./logs/Graph', histogram_freq=5, write_graph=True, write_images=True, write_grads=True)]
#lstm_model.summary()

# load data and predict
y = None
yhat = None
#num_batches = int(len(validation_samples)/(BATCH_SIZE*frames))
num_batches = 600
print("Input count: {0}, Batch count: {1}".format(len(validation_samples), num_batches))

validation_generator = generator_eval(validation_samples, batch_size=BATCH_SIZE, frames_per_clip=frames, shuffle=False)
steps = 0
for x_batch, y_batch, image_tool in validation_generator:
    steps+=1
    print("Batch# {0}, Input shape {1}, X-Length {2}, gt shape {3}, Y-Length {4}".format(steps, np.shape(x_batch),len(x_batch[0]), np.shape(y_batch), len(y_batch)))
    if y is None:
        y = y_batch
    else: 
        y = np.concatenate((y, y_batch))

    # loss = lstm_model.evaluate_generator(validation_generator, steps=int(len(validation_samples)/(BATCH_SIZE*frames)), max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
    pred_batch = l_model.predict(x = x_batch, verbose=1)
    print ("pred_batch:",pred_batch)
    '''
    list_tool=tf_cnn.tool_predict(image_tool, 1)
    print("Tool list:",list_tool)
    inputA = np.array(pred_batch)
    inputB = np.array([list_tool])
    #inputA  = inputA.reshape((,7,1))
    #inputB  = inputB.reshape((,7,1))
    #pred_fake = newModel.predict(x=[inputA,inputB], verbose=1)
    print ("pred_batch:",pred_batch)
    #print("pred_fake:",pred_fake)
    #print("Shape:",np.shape(pred_batch))
    Mult_fake = np.multiply(inputA,inputB)
    print ("Mult Fake:",Mult_fake)
    #print("Mult Shape:",np.shape(Mult_fake))
    pred_fake = newModel.predict(x=[inputA,inputB], verbose=1)
    print("pred_fake:",pred_fake)
    '''
    if yhat is None:
        yhat = pred_batch
        #yhat = pred_fake
    else:
        #yhat = np.concatenate((yhat, pred_fake))
        yhat = np.concatenate((yhat, pred_batch))
    #yhat.append(pred_batch)
    if steps >= num_batches: break

yhat = np.argmax(yhat, axis=1)
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

validationfile = open('./logs/Cla_report.txt', 'wt')
validationfile.write(str(cr))
validationfile.close()

validationfile = open('./logs/Conf_report.txt', 'wt')
validationfile.write(str(cm))
validationfile.close()
