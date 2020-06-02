import os
rootFolder = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"

foldNo = "_fold3\\"
subDataSet =  ["train", "valid", "test"]
classFolder = ["B", "NB"]


for subDt in subDataSet:
	for clsNm in classFolder:
		print(rootFolder+foldNo+subDt+"\\"+clsNm+"\\")
		os.makedirs(rootFolder+foldNo+subDt+"\\"+clsNm+"\\")		