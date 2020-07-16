import numpy as np
import cv2
import os


os.system("cls")
pathToSave = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\vanish\\"


path = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\oregon_us\\"
videoFile = "oreclip.mp4"


path = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\california\\"
videoFile = "california1.mp4"


#path = "C:\\Users\\INKOM06\\Pictures\\NormalSmall\\"
#videoFile = "NO20200216-150246-000500_s.mp4"

cap = cv2.VideoCapture(path+videoFile)

totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(totalFrames)

frameIdx = 0
ratio = 0.3


ret, frame0 = cap.read()
M0 = frame0.shape[0]
N0 = frame0.shape[1]
frame0 = cv2.resize(frame0, (int(ratio*N0),int(ratio*M0)))
mm0 = frame0.shape[0]
nn0 = frame0.shape[1]

oldFrame = np.zeros([mm0,nn0], dtype = np.uint8)
oldFrame2 = oldFrame.copy()
diffFrame = oldFrame.copy()
oldiesFrame = oldFrame.copy()
#oldFrame3 = oldFrame.copy()

totalFrames = 100

while(True) and (frameIdx<(totalFrames-1)):
    ret, frame = cap.read()
    pct = (frameIdx/totalFrames)*100

    M = frame.shape[0]
    N = frame.shape[1]

    frame = cv2.resize(frame, (int(ratio*N),int(ratio*M)))
    mm = frame.shape[0]
    nn = frame.shape[1]

    labImg = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



    [Blue,G,R] = cv2.split(frame)
    [L,A,B] = cv2.split(labImg)
    [H,S,V] = cv2.split(hsvImg)

    newFrame = V
    for k in range(mm):
        for l in range(nn):
  #          oldFrame3[k,l] = (newFrame[k,l] + oldFrame3[k,l])//(frameIdx+1)
            oldFrame[k,l] = max(newFrame[k,l],oldFrame[k,l])
            oldFrame2[k,l] = min(newFrame[k,l],oldFrame2[k,l])
            diffFrame[k,l] = abs(oldiesFrame[k,l] - newFrame[k,l])
            if diffFrame[k,l] <10:
                diffFrame[k,l] = 0
            if diffFrame[k,l] >=10:
                diffFrame[k,l] = 255
    avgFrame = oldFrame
    avgFrame2 = oldFrame2
   # avgFrame3 = oldFrame3
    oldiesFrame = newFrame.copy()


    cv2.imshow("MAX Images", avgFrame)
    cv2.imshow("MIN Images", avgFrame2)
    #cv2.imshow("Avg Images", avgFrame3)
    cv2.imshow("Different images", diffFrame)
   



    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frameIdx = frameIdx + 1

    oriMerge = np.hstack((frame,frame,frame))
    rgbMerge = np.hstack((R,G,Blue))
    labMerge = np.hstack((L,A,B))
    hsvMerge = np.hstack((H,S,V))

    finMerge = np.vstack((rgbMerge,labMerge, hsvMerge))


    b1 = int(0.65*mm)
    b2 = int(0.95*mm)
    B_Near = Blue[b1:b2,:]
    S_Near = S[b1:b2,:]
    finMerge2 = np.vstack((B_Near, S_Near))
    
    b1 = int(0.6*mm)
    b2 = int(0.7*mm)
    B_Far = Blue[b1:b2,:]
    S_Far = S[b1:b2,:]
    finMerge2b = np.vstack((B_Far, S_Far))



    #cv2.imshow('Original images',oriMerge[int(0.5*mm):int(0.7*mm),:,:])
    cv2.imshow('RGB and LAB images',finMerge)
    #cv2.imshow('Near sides',finMerge2)
    #cv2.imshow('Far sides',finMerge2b)


    #ret, Bbw = cv2.threshold(B_Near,150,255,cv2.THRESH_BINARY)
    #ret, Sbw = cv2.threshold(S_Near,230,255,cv2.THRESH_BINARY)

    #Smask = cv2.inRange(S_Near, 80, 150)




    #finMerge3 = np.vstack((Bbw, Sbw, Smask))
    #cv2.imshow('Binary Both sides',finMerge3)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cv2.imwrite(pathToSave+"vanish.jpg", avgFrame)

cap.release()
cv2.destroyAllWindows()
