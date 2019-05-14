


import cv2
import io
import os
import subprocess
import glob
import numpy as np
from CNN_LSTM_split_data import split_cholec_data, train_test_data_split
from CNN_LSTM_load_data import load_cholec_data, generator

base_dir = "/Users/madhuhegde/Downloads/cholec80/"
image_dir = base_dir+"images/"
label_dir = base_dir+"labels/"

[train_list, test_list]  = train_test_data_split(image_dir, label_dir, 0.2)


train_generator = generator(train_list, 8)


[X, y] = train_generator.__next__()

print(X.shape, y.shape)




