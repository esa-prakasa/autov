import cv2
import os
import numpy as np

# confusion matrix in sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

os.system('cls')


### ================= Fold 0 ================
#rootPath = r'C:\Users\Esa\Pictures\_DATASET\unetpusbin\topCamSegmented\kFoldFolders\D3'   ## fold0 Train: D0-D1-D2  Test: D3
#resPath = r'C:\Users\Esa\Pictures\_DATASET\unetpusbin\result_fold0'
#resPath = r'C:\Users\Esa\Pictures\_DATASET\unetpusbin\result_fold0B'

### ================= Fold 1 ================
#rootPath = r'C:\Users\Esa\Pictures\_DATASET\unetpusbin\topCamSegmented\kFoldFolders\D0'   ## fold1 Train: D1-D2-D3  Test: D0
#resPath = r'C:\Users\Esa\Pictures\_DATASET\unetpusbin\result_fold1'


### ================= Fold 2 ================
#rootPath = r'C:\Users\Esa\Pictures\_DATASET\unetpusbin\topCamSegmented\kFoldFolders\D1'   ## fold1 Train: D0-D2-D3  Test: D1
#resPath = r'C:\Users\Esa\Pictures\_DATASET\unetpusbin\result_fold2'

### ================= Fold 2 ================
rootPath = r'C:\Users\Esa\Pictures\_DATASET\unetpusbin\topCamSegmented\kFoldFolders\D2'   ## fold1 Train: D0-D1-D3  Test: D2
resPath = r'C:\Users\Esa\Pictures\_DATASET\unetpusbin\result_fold3'



folderNames = os.listdir(rootPath)
resNames = os.listdir(resPath)

NTest = len(os.listdir(resPath))


print("No FileName        tp       fn   fp    tn      acc    pre   rec   f1s")
#f = open(r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\cmatrec\fold0.txt", "w")   #fold0
#f = open(r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\cmatrec\fold1.txt", "w")   #fold1
#f = open(r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\cmatrec\fold2.txt", "w")   #fold2
f = open(r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\cmatrec\fold3.txt", "w")   #fold3

for idx in range(NTest):
	fullPath0 = rootPath+"\\"+folderNames[idx]+"\\mask\\" #_mask.png"
	maskFiles = os.listdir(fullPath0)
	fullPath0 = fullPath0+maskFiles[0]
	#print(fullPath0)

	fullPath1 = resPath+"\\"+ folderNames[idx]+".png" #resNames[idx]
	#print(fullPath1)

	img0 = cv2.imread(fullPath0)
	M = int(0.5*img0.shape[0])
	N = int(0.5*img0.shape[1])

	img0 = cv2.resize(img0, (N,M), interpolation = cv2.INTER_AREA)
	img1 = cv2.imread(fullPath1)
	img1 = cv2.resize(img1, (N,M), interpolation = cv2.INTER_AREA)


	actual = []
	predicted = []

	for i in range(M):
		for j in range(N):
			if img0[i,j,0] >= 200:
				val = 1
			if img0[i,j,0] < 200:
				val = 0
			actual.append(val)
			if img1[i,j,0] >= 200:
				val = 1
			if img1[i,j,0] < 200:
				val = 0
			predicted.append(val)

	matrix = confusion_matrix(actual,predicted, labels=[1,0])
#print('Confusion matrix : \n',matrix)

	tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)

	acc = (tp+tn)/(tp+tn+fp+fn)
	pre = tp/(tp+fp)
	rec = tp/(tp+fn)
	f1s = 2*(pre*rec)/(pre+rec)
	


#	print("No FileName        tp       fn   fp    tn      acc    pre   rec   f1s")
	resultText = ("%d %s %d %d %d %d %3.3f %3.3f %3.3f %3.3f \n"%(idx, folderNames[idx],tp, fn, fp, tn, acc, pre, rec, f1s))
	print(resultText)
	f.write(resultText)

# classification report for precision, recall f1-score and accuracy
	#matrix = classification_report(actual,predicted,labels=[1,0])
	#print('Classification report : \n',matrix)

f.close()
	
















cv2.waitKey(0)
cv2.destroyAllWindows()

