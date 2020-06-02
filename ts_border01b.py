import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint

from keras.models import load_model
from keras.preprocessing import image


kfold = "_fold3"


rootPath  = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"+kfold+"\\train\\"
modelPath = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"+kfold+"\\xmodel\\"
modelName = "best_model_"+kfold+".h5"
model = load_model(modelPath+modelName)

testFolder =[]
testFolder.append("C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"+kfold+"\\test\\B\\")
testFolder.append("C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"+kfold+"\\test\\NB\\")

dimSz = 20
os.system("cls")
accuracy = []

y_test = []
y_pred = []

for clsSelected in range (2):
	testedFolder = testFolder[clsSelected]

	files = os.listdir(testedFolder)
	NF = len(files)

	target_names = [item for item in os.listdir(rootPath) if os.path.isdir(os.path.join(rootPath, item))]

	correctCount = 0

	for i in range(NF): #range(Nb):
		test_image = image.load_img((testedFolder+files[i]), target_size =(dimSz, dimSz))
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis = 0)
		result = int(model.predict(test_image))
		#print("%2.2f"%(result[0]))
		#classIdx = np.argmax(result)
		print(str(i)+" --- "+files[i]+"--"+str(clsSelected)+" >>> "+str(result))
		y_test.append(clsSelected)
		y_pred.append(result)
		if (clsSelected==result):
			correctCount = correctCount + 1

	accuracy.append(correctCount/NF)

print("  ")
for clsSelected in range(2):	
	print("Class of "+target_names[clsSelected]+(" %3.2f"%(accuracy[clsSelected])))


cm = []
cm.append([accuracy[0], (1-accuracy[0])])
cm.append([(1-accuracy[1]), accuracy[1]])

print(cm)


### =============================
#Another statistical parameters:
### =============================

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt




acc   = accuracy_score(y_test,y_pred)
prec  = precision_score(y_test,y_pred, average =None)
recl  = recall_score(y_test,y_pred, average =None)
f1_sc = f1_score(y_test,y_pred, average =None)
cohkp = cohen_kappa_score(y_test,y_pred)
#roc   = roc_auc_score(y_test,y_pred)

print("Accuracy          : %3.4f "%(acc))
print("Average precision : %3.4f "%(np.average(prec)))
print("Aveage recall    : %3.4f "%(np.average(recl)))
print("Average F1-score  : %3.4f "%(np.average(f1_sc)))
print("Cohen-Kappa score : %3.4f "%(cohkp))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

#className = []
#for i in range(10):
#    className.append(str(i))

className = ["Border", "Non Border"]


cmap=plt.cm.Reds
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)

ax.figure.set_size_inches(8,6,True)

ax.set(xticks=np.arange(cm.shape[1]),
  yticks=np.arange(cm.shape[0]),
  xticklabels=className, yticklabels=className,
  title='',
  ylabel='Input Area',
  xlabel='Predicted Area')

# Rotate the tick labels and set their alignment.
#plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
#  rotation_mode="anchor")

fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
  for j in range(cm.shape[1]):
    ax.text(j, i, format(cm[i, j], fmt),
    ha="center", va="center",
    color="white" if cm[i, j] > thresh else "black")

#fig.tight_layout()

plt.show()



