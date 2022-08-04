import cv2
import numpy as np
import os


os.system("cls")

path = r"C:\Users\Esa\Videos\gopro"
pathToSave = r"C:\Users\Esa\Pictures\_DATASET\pusbin\img"

files = os.listdir(path)
fileIdx = 0

cap = cv2.VideoCapture(os.path.join(path,files[fileIdx]))

if (cap.isOpened()== False):
    print("Error opening video file")
ratio = 0.2
idx = 0

while(cap.isOpened()):
    ret, frame = cap.read()

    if (ret == True):

        M = frame.shape[0]
        N = frame.shape[1]
        frame = cv2.resize(frame, (int(ratio*N),int(ratio*M)), interpolation = cv2.INTER_AREA)

        frame = frame[120:M,:,:]

        red = frame[:,:,2]
        gre = frame[:,:,1]
        blu = frame[:,:,0]

        hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        Hframe = frame[:,:,0]
        Sframe = frame[:,:,1]
        Vframe = frame[:,:,2]

        LABimg = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        Lfrm = frame[:,:,0]
        Afrm = frame[:,:,1]
        Bfrm = frame[:,:,2]


        totalFrame = np.hstack((red, gre))
        totalFrame = np.hstack((totalFrame, blu))

        totalFrame2 = np.hstack((Hframe, Sframe))
        totalFrame2 = np.hstack((totalFrame2, Vframe))

        totalFrame3 = np.hstack((Lfrm, Afrm))
        totalFrame3 = np.hstack((totalFrame3, Bfrm))

        finalFrame = np.vstack((totalFrame, totalFrame2))
        finalFrame = np.vstack((finalFrame, totalFrame3))



        # ret2,th1 = cv2.threshold(red,60,255,cv2.THRESH_BINARY_INV)

        # cv2.imshow('Frame', frame)
        # cv2.imshow('Frame', totalFrame)
        # cv2.imshow('ThrRED', th1)

        cv2.imshow('Frame', finalFrame)

	# Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
	        break


cap.release()

cv2.destroyAllWindows()
