import keras
import os
os.system("cls")


srcFolderB = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\B\\"
srcFolderNB = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\NB\\"



files = [1,2,3,4,5,6,7,8,9,10]
training = files[:int(len(files)*0.6)] #[1, 2, 3, 4, 5, 6, 7, 8]
validation = files[-int(len(files)*0.1):] #[10]
testing = files[-int(len(files)*0.1):] #[10]


print(files)
print(training)
print(validation)
print(testing)

print("Keras has been imported")