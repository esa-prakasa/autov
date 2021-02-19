#def mask_from_poly(xys,shape=(480,640)):
import cv2
import numpy as np

shape =[400, 400]


xys = [[  100, 100],
 [100, 300],
 [260, 300],
 [260, 90]]


img = np.zeros(shape,dtype='uint8')
xys = np.array(xys).astype('int32')
cv2.fillConvexPoly(img,xys,(255,0,0))
print(xys)


cv2.imshow("aa", img)

#return img.copy()

cv2.waitKey(0)
cv2.destroyAllWindows()
