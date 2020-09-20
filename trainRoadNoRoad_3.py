import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

os.system('cls')


csvPath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\roads_annotated\ds\_csv'

csvFiles = os.listdir(csvPath)
csvIdx = 0
for csvFileNm in csvFiles:
	print(str(csvIdx)+"  "+csvFiles[csvIdx])
	csvIdx = csvIdx + 1

selectedCsvIdx = int(input("Which csv file that will be used? "))


#print(csvFiles)
filePath = os.path.join(csvPath,csvFiles[selectedCsvIdx])
#filePath = os.path.join(csvPath,"10_samp_1.csv")

#print(csvFiles[selectedCsvIdx][:-3])

#a = input()




modelFileNm = csvFiles[selectedCsvIdx][:-3]
modelJsonFileNm = modelFileNm+"json"
modelH5FileNm = modelFileNm+"hdf5"
modelBestNm = "norm_theBestModelOf_"+modelFileNm+"hdf5"

modelBestPath = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\oregon_us\\roads_annotated\\ds\\_json\\"+modelBestNm

dataset = pd.read_csv(filePath)
print(dataset.head(10))
#print(dataset.describe(include='all'))


Nf = 6
X= dataset.iloc[:,1:Nf]
Y= dataset.iloc[:,Nf]

#print("Input")
#print(X)
#print("Target")
#print(Y)

Xori = X

# print(X.iloc[0,0])

# X.iloc[0,0] = 100/255

# print(X.iloc[0,0])

# print(X)
Ncsv = len(X)


for i in range(Ncsv):
 	X.iloc[i,0] = X.iloc[i,0] - 0.5
 	X.iloc[i,1] = X.iloc[i,1] - 0.5

 	X.iloc[i,2] = X.iloc[i,2] - 0.5
 	X.iloc[i,3] = X.iloc[i,3] - 0.5
 	X.iloc[i,4] = X.iloc[i,4] - 0.5

# 	print("%d  %2.3f  %2.3f  %2.3f  %2.3f  %2.3f"%(i, X.iloc[i,0], X.iloc[i,1], X.iloc[i,2], X.iloc[i,3], X.iloc[i,4]))

print("================================================")
print(X)
#=========================
#standardizing the input feature
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#X = sc.fit_transform(X)
#print("Use Standard Scaler")
#print(X)

Xori = sc.fit_transform(Xori)
print("Use Standard Scaler")
print(Xori)
# ===========================

X2 = X.to_numpy()

print(type(X))
print(type(X2))
print(type(Xori))


print(X2)



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X2, Y, test_size=0.2)

os.system('cls')



from keras import optimizers
inputSz = Nf-1

classifier = Sequential()

classifier.add(Dense(30, activation='relu', kernel_initializer='random_normal', input_dim=inputSz)) 
classifier.add(Dense(15, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(7, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

classifier.summary()
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])




import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

#checkpoint = ModelCheckpoint("best_model_2ndArc.hdf5", monitor='loss', verbose=1,
#checkpoint = ModelCheckpoint(modelBestNm, monitor='loss', verbose=1,
checkpoint = ModelCheckpoint(modelBestPath, monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

batch_size=30 

epochs= int(input("What is the epochs value: "))

#Fitting the data to the training dataset
history = classifier.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_split=0.2,
          callbacks=[checkpoint])

# serialize model to JSON

model_json = classifier.to_json()
with open("C:\\Users\\INKOM06\\Pictures\\roadDataset\\oregon_us\\roads_annotated\\ds\\_json\\norm_"+modelJsonFileNm, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("C:\\Users\\INKOM06\\Pictures\\roadDataset\\oregon_us\\roads_annotated\\ds\\_json\\last_weight_norm_"+modelH5FileNm)
print("Saved model to disk")

print(history.history.keys())
# summarize history for accuracy
#plt.plot(history.history['accuracy'])
#plt.figure()
plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
#plt.show(block=False)

# summarize history for loss
#plt.figure()
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


plt.show()

#plt.show(block=False)



cv2.waitKey()
cv2.destroyAllWindows()