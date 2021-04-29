import numpy as np
import cv2
import os


os.system("cls")

path = r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\expdata\input"

pathToSave = r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\expdata\track001frames"

files = os.listdir(path)
fileIdx = 0

cap = cv2.VideoCapture(os.path.join(path,files[fileIdx]))

totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(totalFrames)
ratio = 1

frameIdx = 0
while(True):
    ret, frame = cap.read()

    M = frame.shape[0]
    N = frame.shape[1]

    frame = cv2.resize(frame, (int(ratio*N),int(ratio*M)))

    if ((frameIdx % 2)==0):
    	idxS = str(100000 + frameIdx)
    	idxS = idxS[1:]
    	fileName = idxS+".png" 
    	print(fileName)
    	cv2.imwrite(os.path.join(pathToSave,fileName), frame)


    cv2.imshow("frame", frame)
    frameIdx = frameIdx + 1


    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break
    
cap.release()
cv2.destroyAllWindows()
