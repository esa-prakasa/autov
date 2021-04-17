
import time
import os
import numpy as np 
import cv2
import random
#from tensorflow import keras
#from keras.models import Model
#from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda


from tqdm import tqdm
#import matplotlib.pyplot as plt


start = time.time()

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
n_classes = 2

TRAIN_PATH = r'C:\Users\INKOM06\Pictures\roadDataset\pusbin1130segment\train'
TEST_PATH = r'C:\Users\INKOM06\Pictures\roadDataset\pusbin1130segment\test'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

idx = 0
for name in train_ids:
    idx = idx + 1
    print(str(idx)+" "+name)


X_train = np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
Y_train = np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH, 1), dtype = np.bool)
'''
print('Resizing training image and masks')
idx = 0
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    #path = TRAIN_PATH +'\\'+id_
    path = os.path.join(TRAIN_PATH, id_)
    imgFile = id_+'.png'
    imgPath = os.path.join(path,'images',imgFile)
    #print(imgPath)
    img = cv2.imread(imgPath)
    img = cv2.resize(img,(IMG_WIDTH, IMG_HEIGHT) , interpolation = cv2.INTER_AREA)
    X_train[n] = img 
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)
    idx = idx + 1
    
    idy = 0
    for mask_file in next(os.walk(path+'\\mask'))[2]:
        maskPath = os.path.join(path,'mask',mask_file)
        mask_ = cv2.imread(maskPath)
        mask_ = cv2.resize(mask_,(IMG_WIDTH, IMG_HEIGHT) , interpolation = cv2.INTER_AREA)
        mask_ = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
        #os.system('cls')
        #print(mask_.shape)

        mask_ = np.expand_dims(mask_, axis=-1)
        print("mask_.shape ",mask_.shape)
        mask = np.maximum(mask, mask_)
        idy = idy + 1
        print("idx %d   idy %d"%(idx,idy))

        Y_train[n] = mask
        #print(maskPath)

'''

# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images') 
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    path = os.path.join(TEST_PATH, id_)
    imgFile = id_+'.png'
    imgPath = os.path.join(path,'images',imgFile)
    #print(imgPath)

    img = cv2.imread(imgPath)
    sizes_test.append([img.shape[0],img.shape[1]])
    img = cv2.resize(img,(IMG_WIDTH, IMG_HEIGHT) , interpolation = cv2.INTER_AREA)
    X_test[n] = img 


print(type(X_test))
print(len(X_test))

X_test2 = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
print("Len X_test2:  %d "%(len(X_test2)))

print(X_test2[1])
print('Done!')

#print(sizes_test)