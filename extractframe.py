import numpy as np
import cv2
import os


os.system("cls")

#path = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\"

#path =r"C:\Users\INKOM06\Pictures\_DATASET\roadcibi"
path = r"C:\Users\INKOM06\Pictures\_DATASET\_CIBISurvey2\1202_start"
pathToSave = r"C:\Users\INKOM06\Pictures\_DATASET\_CIBISurvey2\_1202Tsp30frm"

files = os.listdir(path)
fileIdx = 4

cap = cv2.VideoCapture(os.path.join(path,files[fileIdx]))

totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(totalFrames)
ratio = 0.5

frameIdx = 0
while(True):
    ret, frame = cap.read()

    M = frame.shape[0]
    N = frame.shape[1]

    frame = cv2.resize(frame, (int(ratio*N),int(ratio*M)))

    if ((frameIdx % 60)==0):
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
