from numpy import random
import numpy as np
import os

os.system('cls')

#path = r'C:\Users\INKOM06\Pictures\roadDataset\Unetpusbin\track001frames'
path = r'C:\Users\INKOM06\Pictures\_DATASET\expdata\track001frames'

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
	print("%d:  %s    %s"%(i,gFin[i], fileList[gFin[i]]))
