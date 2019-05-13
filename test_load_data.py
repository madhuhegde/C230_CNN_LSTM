


import cv2
import io
import os
import subprocess
import glob
import numpy as np
from CH_load_data import load_cholec_data

base_dir = "/Users/madhuhegde/Downloads/cholec80/"
image_dir = base_dir+"images/"
label_dir = base_dir+"labels/"

[data, label] = load_cholec_data(image_dir, label_dir, 25)
