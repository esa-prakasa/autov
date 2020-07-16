import numpy as np
import cv2
import os


#os.system("cls")

path = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\"


videoFile = "oregon_us\\oreclip.mp4"
#videoFile = "california\\california1.mp4"

cap = cv2.VideoCapture(path+videoFile)

totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

frameIdx = 0
ratio = 0.1

def getCentroid(image, refPix):
    M = image.shape[0]
    N = image.shape[1]

    nPix = 0
    iAcc = 0
    jAcc = 0
    for i in range(M):
        for j in range(N):
            if image[i,j] == refPix:
                iAcc = iAcc + i
                jAcc = jAcc + j
                nPix = nPix + 1

    iCent = round(iAcc/nPix)
    jCent = round(jAcc/nPix)

    return iCent, jCent


def getDiffFrame(imageA, imageB):
    M = imageA.shape[0]
    N = imageB.shape[1]


    imageC = np.zeros((M,N),np.uint8)

    for i in range(M):
        for j in range(N):
            imageC[i,j] = abs(imageA[i,j]-imageB[i,j])
    return imageC






ret, frame = cap.read()
M = frame.shape[0]
N = frame.shape[1]

frame = cv2.resize(frame, (int(ratio*N),int(ratio*M)))
mm = frame.shape[0]
nn = frame.shape[1]


oldFrame = np.zeros((mm,nn),np.uint8)
deltaFrame = oldFrame.copy() 

print(frameIdx)

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

    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frameIdx = frameIdx + 1

    imgC = G.copy()

    imgC = cv2.blur(imgC,(3,3))
    mS = imgC.shape[0]
    nS = imgC.shape[1]


    
    cv2.imshow("Channel image", imgC)
    deltaH = round(0.1*nS)
    deltaV = round(0.2*mS)
    deltaVb = 10

    nS_D2 = nS // 2

    centArea = imgC[(mS-deltaV):(mS-deltaVb),(nS_D2-deltaH):(nS_D2+deltaH)]

    cv2.imshow("Cent area",centArea)

    centAVG = round(np.average(centArea))
    centSTD = round(np.std(centArea))
    print(str(frameIdx)+"   "+str(centAVG)+"   "+str(centSTD))

    start_point = ((nS_D2-deltaH), (mS-deltaV)) 
    end_point = ((nS_D2+deltaH), (mS-deltaVb)) 
    color = (0, 255, 0)   
    thickness = 1
    frame = cv2.rectangle(frame, start_point, end_point, color, thickness) 


   
    ret1,th1 = cv2.threshold(imgC,centAVG,255,cv2.THRESH_BINARY)
    cv2.imshow("TH 1",th1)


    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(th1,kernel,iterations = 1)
    [iCent, jCent] = getCentroid(dilation,0)

    #rgb = cv2.cvtColor(dilation,cv2.COLOR_GRAY2RGB)

    #for iii in range((iCent-10),(iCent+10),1):
    #    for jjj in range((jCent-10),(jCent+10),1):
    #        rgb[iii,jjj,0] = 0
    #        rgb[iii,jjj,1] = 255
    #        rgb[iii,jjj,2] = 255


    cv2.imshow("S2 value", th1)
    cv2.imshow("opening", dilation)



    #cv2.imshow("RGB", rgb)


    deltaFrame = getDiffFrame(dilation, oldFrame)
    oldFrame = dilation

    cv2.imshow("Delta Frame", deltaFrame)




    start_point = ((jCent-10), (iCent-10)) 
    end_point = ((jCent+10), (iCent+10)) 
    color = (255, 0, 0)   
    thickness = 1  
    frame = cv2.rectangle(frame, start_point, end_point, color, thickness) 

    start_point = (0, iCent) 
    end_point = (nS, iCent) 
    color = (255, 0, 255)   
    thickness = 1  
    frame = cv2.rectangle(frame, start_point, end_point, color, thickness) 





    cv2.imshow("Frame ", frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
