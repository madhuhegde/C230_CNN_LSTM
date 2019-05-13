


import cv2
import io
import os
import subprocess
import glob
import numpy as np
from CNN_LSTM_split_data import split_cholec_data
from CNN_LSTM_load_data import load_cholec_data

base_dir = "/Users/madhuhegde/Downloads/cholec80/"
image_dir = base_dir+"images/"
label_dir = base_dir+"labels/"

[train_array, test_array] = split_cholec_data(image_dir, label_dir, 0.2)


[X_train, y_train] = load_cholec_data(image_dir, label_dir, 10, train_array)
print(np.array(X_train).shape, y_train.shape)


