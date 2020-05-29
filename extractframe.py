import numpy as np
import cv2
import os

os.system("cls")

path = "C:\\Users\\INKOM06\\Pictures\\NormalSmall\\"
videoFile = "NO20200216-145746-000495_s.mp4"
pathToSaveFiles = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\"


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




while(True) and (frameIdx<(totalFrames-1)):    # Capture frame-by-frame
    ret, frame = cap.read()

    #frame = cv2.resize(frame,(rM,rN),interpolation = cv2.INTER_AREA)

    frame = frame[0:int(0.8*M),:,:]

    fileIdx = 10000 + frameIdx
    fileIdxS = str(fileIdx)
    fileIdxS = fileIdxS[1:]
    imgFileName = fileIdxS+"__"+videoFile[:-4]+".jpg"
    if ((frameIdx%10)==0):
        print(imgFileName)
        cv2.imwrite((pathToSaveFiles+imgFileName),frame)





    
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #ret,th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    #ret,th2 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    #edges = cv2.Canny(th1,90,100,5)

    #cv2.imshow("Edges", edges)
    cv2.imshow("Frame", frame)

    frameIdx = frameIdx + 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



'''




while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    pct = (frameIdx/totalFrames)*100

    M = frame.shape[0]
    N = frame.shape[1]
    #ratio = 0.5

    #frame = cv2.resize(frame, (int(ratio*N),int(ratio*M)))

    cv2.imshow('Frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break










    frame = frame[200:400,300:650, :]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Mi = img.shape[0]
    Ni = img.shape[1]
    deltaImg = img.copy()

    if frameIdx == 0:
        imgOld = img

    if frameIdx > 0:
        for i in range(Mi):
            for j  in range(Ni):
                value = int(img[i,j] - imgOld[i,j])
                if value < 0:
                    value = 0
                if value >= 0:
                    pass
                deltaImg[i,j] = value    


        imgOld = img








    frameIdx = frameIdx + 1
    #print(frameIdx)


    #frame = cv2.
    #print('Frame index: %d Pct: %3.2f %s' % (frameIdx,pct,chr(37)))
    #frameIdx = frameIdx + 1

    # Our operations on the frame come here
    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame-Binary',deltaImg)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

'''
cap.release()
cv2.destroyAllWindows()