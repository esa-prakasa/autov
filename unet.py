## codes: https://github.com/bnsreenu/python_for_microscopists/blob/master/076-077-078-Unet_nuclei_tutorial.py

import os
import numpy as np 
import cv2
import random
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda


from tqdm import tqdm
import matplotlib.pyplot as plt


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
n_classes = 2

TRAIN_PATH = r'C:\Users\INKOM06\Pictures\_DATASET\Unet\stage0_train'
TEST_PATH = r'C:\Users\INKOM06\Pictures\_DATASET\Unet\stage1_test'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

idx = 0
for name in train_ids:
    idx = idx + 1
    print(str(idx)+" "+name)


X_train = np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
Y_train = np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH, 1), dtype = np.bool)

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
    for mask_file in next(os.walk(path+'\\masks'))[2]:
        maskPath = os.path.join(path,'masks',mask_file)
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


print('Done!')



#image_x = random.randint(0, len(train_ids))
#cv2.imshow("X_train",X_train[image_x])
#plt.show()
#cv2.imshow("Y_train",np.squeeze(Y_train[image_x]))
#plt.show()






################################################################
#def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
def multi_unet_model(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    #NOTE: Compile the model in the main program to make it easy to test with various loss functions
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    #model.summary()
    
    return model

os.system('cls')

modelRd = multi_unet_model(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)



modelRd = keras.Model(inputs=[inputs], outputs=[outputs])
modelRd.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modelRd.summary()


############################

#checkpointer = keras.callbacks.ModelCheckpoint('model.h5', verbose=1, save_best_only=True)

#callbacks = [
#    keras.callbacks.EarlyStopping(patience=2, monitor ='val_loss'),
#    keras.callbacks.TensorBoard(log_dir='logs')]


################################
#Modelcheckpoint
checkpointer = keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

callbacks = [
        keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        keras.callbacks.TensorBoard(log_dir='logs')]

results = modelRd.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=3, callbacks=callbacks)

####################################

idx = random.randint(0, len(X_train))


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)




cv2.waitKey(0)
cv2.destroyAllWindows()