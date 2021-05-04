from numpy import random
import numpy as np
import os
import shutil

os.system('cls')

path = r'C:\Users\Esa\Pictures\_DATASET\unetpusbin\expdata\track001frames'  ##LIPI
#path = r'C:\Users\INKOM06\Pictures\_DATASET\expdata\track001frames'         ##Home


targetPath = r'C:\Users\Esa\Pictures\_DATASET\unetpusbin\expdata\s200frames'

fileList = os.listdir(path)

g = []

for i in range(len(fileList)):
	g.append(i)

gNew = random.permutation(g)

gFin = []
sampSz = 200 
for i in range(sampSz):
	gFin.append(gNew[i])

gFin = sorted(gFin)

#print(gFin) 

for i in range(sampSz):
	#print("%d:  %s    %s"%(i,gFin[i], fileList[gFin[i]]))
	print("%d:  %s    %s"%(i,gFin[i], fileList[gFin[i]]))
	srcPath = os.path.join(path,fileList[gFin[i]])
	savePath = os.path.join(targetPath,fileList[gFin[i]])
	shutil.copyfile(srcPath, savePath)
	print(srcPath)
	print(savePath)
	print(" ")


