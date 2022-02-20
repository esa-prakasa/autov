import cv2
import os
import random

os.system("cls")

path = r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\edges"
files = os.listdir(path)

idx = random.randint(0,len(files))

img = cv2.imread(os.path.join(path,files[idx]))
M = img.shape[0]
N = img.shape[1]

Md2 = M//2
Nd2 = N//2

for i in range(-3,3):
    for j in range(-3,3):
        img[Md2+i, Nd2+j] = 255


# Find the right border 
j = Nd2
while (j<(N-2)):
    j = j + 1
    sum = 0
    for k in range(-1,1):
        sum = sum + img[Md2, j+k,0]
    
    if (sum > 300) and (j >(Nd2+10)):
        for l in range(-10,10):
            img[Md2, j+l,0] = 0
            img[Md2, j+l,1] = 255
            img[Md2, j+l,2] = 255

    print(">> %d %d  %d"%(j, img[Md2, j,0], sum))


# Find the left border 
j = Nd2
while (j>2):
    j = j - 1
    sum = 0
    for k in range(-1,1):
        sum = sum + img[Md2, j+k,0]
    
    if (sum > 250) and (j <(Nd2-10)):
        for l in range(-10,10):
            img[Md2, j+l,0] = 0
            img[Md2, j+l,1] = 255
            img[Md2, j+l,2] = 0

    print(">> %d %d  %d"%(j, img[Md2, j,0], sum))





cv2.imshow(str(idx),img)


cv2.waitKey(0)
cv2.destroyAllWindows