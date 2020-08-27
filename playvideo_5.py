import numpy as np
import cv2
import os


#os.system("cls")

path = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\"
path0 = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\oregon_us\\"

#videoFile = "oregon_us\\oreclip.mp4"
videoFile = "oreclip.mp4"
#videoFile = "california\\california1.mp4"

cap = cv2.VideoCapture(path0+videoFile)

totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

frameIdx = 0
ratio = 0.05

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


#def getRefPat(R,G,B, xR, xG, xB):
def getRefPat(accR,accG,accB,accH,accS,accV,R,G,B, H, S, V):
    M = R.shape[0]
    N = R.shape[1]

#    accR = 0
#    accG = 0
#    accB = 0

#    accH = 0
#    accS = 0
#    accV = 0

    for i in range(M):
        for j in range(N):
            accR = accR + R[i,j]
            accG = accG + G[i,j]
            accB = accB + B[i,j]

            accH = accH + H[i,j]
            accS = accS + S[i,j]
            accV = accV + V[i,j]

    xR = round(accR/(M*N))
    xG = round(accG/(M*N))
    xB = round(accB/(M*N))


    xH = round(accH/(M*N))
    xS = round(accS/(M*N))
    xV = round(accV/(M*N))

    return xR, xG, xB, xH, xS, xV


def getSegment(frame,xR, xG, xB, xH, xS, xV, Thr):
    M = frame.shape[0]
    N = frame.shape[1]

    rgbBin = np.zeros((M,N), np.uint8)

    for i in range(M):
        for j in range(N):
            dR = (xR - frame[i,j,2])**2
            dG = (xG - frame[i,j,1])**2
            dB = (xB - frame[i,j,0])**2

            dH = (xH - H[i,j])**2
            dS = (xS - S[i,j])**2
            dV = (xV - V[i,j])**2

            dist = np.sqrt(dR + dG + dB +dH + dS + dV)

            #print(dist)

            if dist<Thr:
                rgbBin[i,j] = 255

    return rgbBin 


ret, frame = cap.read()
M = frame.shape[0]
N = frame.shape[1]

frame = cv2.resize(frame, (int(ratio*N),int(ratio*M)))
mm = frame.shape[0]
nn = frame.shape[1]


oldFrame = np.zeros((mm,nn),np.uint8)
deltaFrame = oldFrame.copy() 

print(frameIdx)

xR = 0
xG = 0
xB = 0


accR = 0
accG = 0
accB = 0

accH = 0
accS = 0
accV = 0



#while(True) and (frameIdx<(totalFrames-1)):
while(True) and (frameIdx<500):
    

    fileName = str(frameIdx+10000)
    fileName = fileName[1:]
    fileName = fileName+".png"
    print(fileName)



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

    imgStack1 = np.hstack((R,G,B))
    imgStack2 = np.hstack((H,S,V))

    imgStack3 = np.vstack((imgStack1,imgStack2))

    cv2.imwrite(path0+"stack3\\"+fileName, imgStack3)


    #cv2.imshow("RGB",imgStack3)


#    [xR, xG, xB] = getRefPat(R, G, Blue, xR, xG, xB)
#    [xR, xG, xB] = getRefPat(R, G, Blue)


    [xR, xG, xB, xH, xS, xV] = getRefPat(accR,accG,accB,accH,accS,accV,R,G,B, H, S, V)

    #[xR, xG, xB, xH, xS, xV] = getRefPat(R, G, Blue, H, S, V)



    print("%d R: %d  G: %d  B: %d  H: %d  S: %d  V: %d"%(frameIdx, xR,xG,xB,xH,xS,xV))

    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    


    rgbBin = getSegment(frame,xR, xG, xB, xH, xS, xV, 100)

    cv2.imshow("rgbBin ",rgbBin)





    frameIdx = frameIdx + 1

    imgC = B.copy()

    imgC = cv2.blur(imgC,(3,3))
    mS = imgC.shape[0]
    nS = imgC.shape[1]


    
    #cv2.imshow("Channel image", imgC)
    deltaH = round(0.1*nS)
    deltaV = round(0.2*mS)
    deltaVb = 10

    nS_D2 = nS // 2

    centArea = imgC[(mS-deltaV):(mS-deltaVb),(nS_D2-deltaH):(nS_D2+deltaH)]

    #cv2.imshow("Cent area",centArea)

    centAVG = round(np.average(centArea))
    centSTD = round(np.std(centArea))
    #print(str(frameIdx)+"   "+str(centAVG)+"   "+str(centSTD))

    start_point = ((nS_D2-deltaH), (mS-deltaV)) 
    end_point = ((nS_D2+deltaH), (mS-deltaVb)) 
    color = (0, 255, 0)   
    thickness = 1
    #frame = cv2.rectangle(frame, start_point, end_point, color, thickness) 


   
    ret1,th1 = cv2.threshold(imgC,centAVG,255,cv2.THRESH_BINARY)

    #cv2.imshow("TH 1",th1)


    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(th1,kernel,iterations = 1)
    cv2.imwrite(path0+"bin\\"+fileName, dilation)

    [iCent, jCent] = getCentroid(dilation,0)

    #rgb = cv2.cvtColor(dilation,cv2.COLOR_GRAY2RGB)

    #for iii in range((iCent-10),(iCent+10),1):
    #    for jjj in range((jCent-10),(jCent+10),1):
    #        rgb[iii,jjj,0] = 0
    #        rgb[iii,jjj,1] = 255
    #        rgb[iii,jjj,2] = 255


    #cv2.imshow("S2 value", th1)
    #cv2.imshow("opening", dilation)
    #cv2.imshow("RGB", rgb)



    start_point = ((jCent-10), (iCent-10)) 
    end_point = ((jCent+10), (iCent+10)) 
    color = (255, 0, 0)   
    thickness = 1  
  #  frame = cv2.rectangle(frame, start_point, end_point, color, thickness) 

    start_point = (0, iCent) 
    end_point = (nS, iCent) 
    color = (255, 0, 255)   
    thickness = 1  
    frame = cv2.rectangle(frame, start_point, end_point, color, thickness) 


    start_point = (jCent, 0) 
    end_point = (jCent, mS) 
    color = (0, 255, 0)   
    thickness = 1  
    frame = cv2.rectangle(frame, start_point, end_point, color, thickness) 
    cv2.imwrite(path0+"line\\"+fileName, frame)




    cv2.imshow("Frame ", frame)

    accR = xR
    accG = xG
    accB = xB
    accH = xH
    accS = xS
    accV = xV

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
