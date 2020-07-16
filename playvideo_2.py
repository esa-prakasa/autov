import numpy as np
import cv2
import os


os.system("cls")

path = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\oregon_us\\"
videoFile = "oreclip.mp4"

cap = cv2.VideoCapture(path+videoFile)

totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(totalFrames)

frameIdx = 0
ratio = 0.1

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


    SHist = cv2.equalizeHist(S)

    cv2.imshow("S histeq", SHist)


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


    ret, Bbw = cv2.threshold(B_Near,150,255,cv2.THRESH_BINARY)
    ret, Sbw = cv2.threshold(S_Near,230,255,cv2.THRESH_BINARY)

    Smask = cv2.inRange(S_Near, 80, 150)




    finMerge3 = np.vstack((Bbw, Sbw, Smask))
    #cv2.imshow('Binary Both sides',finMerge3)


    mS = S.shape[0]
    nS = S.shape[1]

    r2 = 0.2

    mS_r = int(r2*mS)
    nS_r = int(r2*nS)

    S2 = cv2.resize(S, (nS_r,mS_r))

    cv2.imshow("S value", S)

    


    
    ret1,th1 = cv2.threshold(S,120,255,cv2.THRESH_BINARY)
    cv2.imshow("S2 value", th1)



    #cv2.imshow("S2 value", S2)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
