import numpy as np
import cv2
import os
import statistics

os.system("cls")

path = "C:\\Users\\INKOM06\\Pictures\\NormalSmall\\"
videoFile = "NO20200216-145746-000495_s.mp4"
#videoFile = "NO20200216-143946-000477_s.mp4"
#videoFile = "NO20200206-083245-000471_s.mp4"


cap = cv2.VideoCapture(path+videoFile)
totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(totalFrames)

frameIdx = 0

ret, frame = cap.read()
M = frame.shape[0]
N = frame.shape[1]

scaleRatio = 0.5
rM = int(scaleRatio*M)
rN = int(scaleRatio*N)

## =======================================
kfold = "_fold3"





def classfSingleSubRGB(subFrame):
    M = subFrame.shape[0]
    N = subFrame.shape[1]
    Ravg = 0
    Gavg = 0
    Bavg = 0
    for i in range(M):
        for j in range(N):
            Ravg = Ravg + subFrame[i,j,2]
            Gavg = Gavg + subFrame[i,j,1]
            Bavg = Bavg + subFrame[i,j,0]
    Ravg = int(Ravg/(M*N))
    Gavg = int(Gavg/(M*N))
    Bavg = int(Bavg/(M*N))

    rgbAvg = [Ravg, Gavg, Bavg]
    sdRGB = statistics.mean(rgbAvg)
    sdRGB = abs(sdRGB - 128)

    b,g,r = cv2.split(subFrame)
    zr = np.zeros((M,N),dtype="uint8")

    thr = 50
    if sdRGB<=thr:
        subFrame = cv2.merge([zr,g,zr])

    if sdRGB>thr:
        subFrame = cv2.merge([zr,zr,r])

    return subFrame



def classfAllSubImages(frame):
    M = frame.shape[0]
    N = frame.shape[1]
    fSz = 20

    for i in range(0,(M-fSz),fSz):
        for j in range(0,(N-fSz),fSz):
                #print("%d %d"%(i,j))
                ic = int(i)
                jc = int(j)
                subFrame = frame[ic:(ic+fSz), jc:(jc+fSz), :]
                subFrame = classfSingleSubRGB(subFrame)
                frame[ic:(ic+fSz), jc:(jc+fSz), :] = subFrame
    return frame


M2 = int(2*0.8*M)
#####  out = cv2.VideoWriter(path+'\\output\\output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (N,M2))


while(True) and (frameIdx<(totalFrames-1)):    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = frame[0:int(0.8*M),:,:]

    frame0 = frame.copy()

    frame = classfAllSubImages(frame)

    frame = cv2.vconcat([frame0, frame])

#####      out.write(frame)


    cv2.imshow("Frame", frame)
    frameIdx = frameIdx + 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #frame = cv2.resize(frame,(rM,rN),interpolation = cv2.INTER_AREA)
'''

    fileIdx = 10000 + frameIdx
    fileIdxS = str(fileIdx)
    fileIdxS = fileIdxS[1:]
    imgFileName = fileIdxS+"__"+videoFile[:-4]+".jpg"
    if ((frameIdx%10)==0):
        print(imgFileName)
        cv2.imwrite((pathToSaveFiles+imgFileName),frame)
'''
#####  out.release()


cap.release()
cv2.destroyAllWindows()