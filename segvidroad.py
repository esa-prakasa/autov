import numpy as np
import cv2
import os

os.system("cls")

#path = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\"

#path =r"C:\Users\INKOM06\Pictures\_DATASET\roadcibi"
path = r"C:\Users\INKOM06\Pictures\_DATASET\_CIBISurvey2\1202_start"

files = os.listdir(path)
fileIdx = 4



cap = cv2.VideoCapture(os.path.join(path,files[fileIdx]))

totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(totalFrames)

ratio = 0.1


frameIdx = 0
while(True):
    # Capture frame-by-frame
    #ret, frame = cap.read()
    #pct = (frameIdx/totalFrames)*100

    ret, frame = cap.read()

    M = frame.shape[0]
    N = frame.shape[1]

    frame = cv2.resize(frame, (int(ratio*N),int(ratio*M)))
    filterSz = 5
    frameBl = cv2.blur(frame,(filterSz,filterSz))
    labImg = cv2.cvtColor(frameBl, cv2.COLOR_BGR2LAB)
    hsvImg = cv2.cvtColor(frameBl, cv2.COLOR_BGR2HSV)

    [Blue,G,R] = cv2.split(frame)
    [L,A,B] = cv2.split(labImg)
    [H,S,V] = cv2.split(hsvImg)

    rgbMerge = np.hstack((R,G,Blue))
    labMerge = np.hstack((L,A,B))
    hsvMerge = np.hstack((H,S,V))

    finMerge = np.vstack((rgbMerge,labMerge, hsvMerge))





    cv2.imshow('frame-Binary',finMerge)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






'''
frame = cv2.imread(os.path.join(path,files[fileIdx]))

M = frame.shape[0]
N = frame.shape[1]
ratio = 0.2

frame = cv2.resize(frame,(int(ratio*N), int(ratio*M)) , interpolation = cv2.INTER_AREA)   
filterSz = 5
frame = cv2.blur(frame,(filterSz,filterSz))







labImg = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



[Blue,G,R] = cv2.split(frame)
[L,A,B] = cv2.split(labImg)
[H,S,V] = cv2.split(hsvImg)



clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#H = clahe.apply(H)




oriMerge = np.hstack((frame,frame,frame))
rgbMerge = np.hstack((R,G,Blue))
labMerge = np.hstack((L,A,B))
hsvMerge = np.hstack((H,S,V))

finMerge = np.vstack((rgbMerge,labMerge, hsvMerge))


cv2.imshow("finMerge", finMerge)


ret,thH = cv2.threshold(H,50,255,cv2.THRESH_BINARY_INV)

#cv2.imshow("Hue", H)

#cv2.imshow("Binary image", thH)
'''

cv2.waitKey(0)
cv2.destroyAllWindows()

