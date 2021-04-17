import time
import os
import numpy as np 
import cv2
import random
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.models import model_from_json

from keras.models import load_model


from tqdm import tqdm
import matplotlib.pyplot as plt

os.system('cls')


modelPath =r'C:\Users\INKOM06\Pictures\roadDataset\pusbin1130segment\_json4' 
jsonFile = 'unetarchi.json'
jsonPath = os.path.join(modelPath, jsonFile) 
json_file = open(jsonPath, 'r') 
loaded_model_json = json_file.read()
json_file.close()
modelRd = load_model(loaded_model_json)
modelRd.summary()






hdfFile = 'unetarchi.h5'
hdfPath = os.path.join(modelPath, hdfFile) 
modelRd.load_weights(hdfPath)

print('Model and weights have been loaded')
