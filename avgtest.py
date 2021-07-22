# average test code
import numpy as np 
import cv2

I = np.zeros([200,200],dtype = np.uint8)

I1 = I
I2 = I + 50
I3 = I + 100
I4 = I + 150
I5 = I + 200

J = np.array([I1,I2])

cv2.imshow("J",J[1])

cv2.waitKey(0)
