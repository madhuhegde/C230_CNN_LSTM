


import cv2
import io
import os
import subprocess
import glob
import numpy as np
from CNN_LSTM_split_data import  generate_feature_list
from CNN_LSTM_load_data import  generator_train, generator_test

base_dir = "/Users/madhuhegde/Downloads/cholec80/"
base_image_dir = base_dir+"images/"
base_label_dir = base_dir+"labels/"
test_image_dir = base_image_dir + "test/"
test_label_dir = base_label_dir + "test/"
train_image_dir = base_image_dir + "train/"
train_label_dir = base_label_dir + "train/"


#[train_list, test_list]  = train_test_data_split(image_dir, label_dir, 0.2)
train_list = generate_feature_list(train_image_dir, train_label_dir)
print(len(train_list))

test_list = generate_feature_list(test_image_dir, test_label_dir)
print(len(test_list))


train_generator = generator_train(train_list, 4, 4)

#for i in range(int(len(train_list)/16)):
#  [X, y] = train_generator.__next__()


test_generator = generator_test(test_list, 4, 4)

for i in range(int(len(test_list)/16)):
 [X, y] = test_generator.__next__()

print(X.shape, y.shape)





