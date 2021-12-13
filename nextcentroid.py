import os
import cv2
os.system("cls")



path = r"C:\Users\Esa\Pictures\zoomAPELs\grid.png"

img = cv2.imread(path)
img0 = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)


i0 = 80
j0 = 90
for i in range(-1,1,1):
    for j in range(-1,1,1):
        img0[i0+i,j0+j,0] = 0
        img0[i0+i,j0+j,1] = 0
        img0[i0+i,j0+j,2] = 255

d = []
d.append(img[i0+1, j0+1])  #0
d.append(img[i0, j0+1])    #1
d.append(img[i0-1, j0+1])  #2
d.append(img[i0-1, j0])    #3
d.append(img[i0-1, j0-1])  #4
d.append(img[i0, j0-1])    #5
d.append(img[i0+1, j0-1])  #6
d.append(img[i0+1, j0])    #7

print(d)

N = len(d)
for i in range(1,(N-1),1):
    delta = d[i+1] - d[i]
    if delta>0:
        idxNext = i + 1 
        print(idxNext)


if idxNext == 0:
    di = 1
    dj = 1
if idxNext == 1:
    di = 0
    dj = 1
if idxNext == 2:
    di = -1
    dj = 1
if idxNext == 3:
    di = -1
    dj = 0
if idxNext == 4:
    di = -1
    dj = -1
if idxNext == 5:
    di = 0
    dj = -1
if idxNext == 6:
    di = 1
    dj = -1
if idxNext == 7:
    di = 1
    dj = 0
    
i0 = 80 + di
j0 = 90 + dj
for i in range(-1,1,1):
    for j in range(-1,1,1):
        img0[i0+i,j0+j,0] = 0
        img0[i0+i,j0+j,1] = 255
        img0[i0+i,j0+j,2] = 0



cv2.imshow("Image", img0)

cv2.waitKey(0)
cv2.destroyAllWindows()
