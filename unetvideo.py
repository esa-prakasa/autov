import numpy as np
import cv2
import os
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda


os.system("clear")


#####  MODEL LOADED

modelPath =r'/home/lgr0220157/unetcsc/model' 
jsonFile = 'unetarchi.json'
jsonPath = os.path.join(modelPath, jsonFile) 
json_file = open(jsonPath, 'r') 
loaded_model_json = json_file.read()
json_file.close()
modelRd = model_from_json(loaded_model_json)
modelRd.summary()


hdfFile = 'unetarchi.h5'
hdfPath = os.path.join(modelPath, hdfFile) 
modelRd.load_weights(hdfPath)

print('Model and weights have been loaded')


videoPath = '/home/lgr0220157/unetcsc/video'
videoFile = 'NO20201130-151922-000167.mp4'
fullVideoPath = os.path.join(videoPath, videoFile)
cap = cv2.VideoCapture(fullVideoPath)


totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("Total frame is %d "%(totalFrames))

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
n_classes = 2


frameIdx = 0
#ratio = 0.2
ret, frame = cap.read()
M = frame.shape[0]
N = frame.shape[1]

heightOri = M
widthOri = N

print("M value is %d "%(M))
print("N value is %d "%(N))

saveVideo = True

savedVideoFile = 'result.avi'
videoPathToSave = os.path.join(videoPath, savedVideoFile)

if saveVideo == True:
    out = cv2.VideoWriter(videoPathToSave,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (N,M))


#fileNameToSave ="output.jpg"
#filePathToSave = os.path.join(videoPath, fileNameToSave)
#cv2.imwrite(filePathToSave, frame)



#>> frame2 = cv2.resize(frame,(IMG_WIDTH, IMG_HEIGHT) , interpolation = cv2.INTER_AREA)


#X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
#X_test[0] = frame2

#preds  = modelRd.predict(X_test, verbose=1)

#print(preds)

#preds_t  = (preds > 0.5).astype(np.uint8)

#imr = np.zeros((IMG_WIDTH,IMG_HEIGHT),dtype=np.uint8)

#for i in range(IMG_HEIGHT):
#    for j in range(IMG_WIDTH):
#        imr[i,j] = int(preds_t[0,i,j]*255)

#imr2 = cv2.resize(imr,(widthOri, heightOri) , interpolation = cv2.INTER_AREA)

#cv2.imwrite(filePathToSave, imr2)



#while(True) and (frameIdx<(totalFrames-1)):
while(True) and (frameIdx<10):

    ret, frame = cap.read()

    frame = cv2.resize(frame, (Nr,Mr),interpolation = cv2.INTER_AREA)
    #frame2 = np.zeros((Mr,Nr,3), dtype = np.uint8)
    M = frame.shape[0]
    N = frame.shape[1]

    heightOri = M
    widthOri = N

    frame2 = cv2.resize(frame,(IMG_WIDTH, IMG_HEIGHT) , interpolation = cv2.INTER_AREA)

    X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    X_test[0] = frame2

    preds  = modelRd.predict(X_test, verbose=1)

    #print(preds)

    preds_t  = (preds > 0.5).astype(np.uint8)

    imr = np.zeros((IMG_WIDTH,IMG_HEIGHT),dtype=np.uint8)

    for i in range(IMG_HEIGHT):
        for j in range(IMG_WIDTH):
            imr[i,j] = int(preds_t[0,i,j]*255)

    imr2 = cv2.resize(imr,(widthOri, heightOri) , interpolation = cv2.INTER_AREA)


    frame3 = imr2 #cv2.resize(frame2, (Nv,Mv))
    print("Frame index: %d  of Total Frames: %d" %(frameIdx, totalFrames))

    if saveVideo == True:
        out.write(frame3)

    frameIdx = frameIdx + 1
    


cap.release()
out.release()
