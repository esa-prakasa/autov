import os
rootFolder = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"
rootFolder = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\Purbaleunyi_Highway_1\\ds\\"


for idxFold in range(3):
	foldNo = "_fold"+str(idxFold+1)+"\\"
	subDataSet =  ["train", "valid", "test"]
	classFolder = ["_B", "_NB"]


	for subDt in subDataSet:
		for clsNm in classFolder:
			print(rootFolder+foldNo+subDt+"\\"+clsNm+"\\")
			os.makedirs(rootFolder+foldNo+subDt+"\\"+clsNm+"\\")

	os.makedirs(rootFolder+foldNo+"\\"+"xmodel"+"\\")		