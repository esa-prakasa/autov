import numpy as np
import cv2
import os
from keras.models import model_from_json


os.system("cls")


def classify(iNorm, jNorm, rNorm, gNorm, bNorm):

    x = np.array([ [iNorm], [jNorm], [rNorm], [gNorm],  [bNorm] ])  ## class 0

    xT = np.array([ x[0], x[1], x[2], x[3], x[4] ])
    xT = np.transpose(xT)
    results = model.predict(xT)

    probVal = results[0][0]
    probTh = 0.5
    if (probVal>probTh):
        #print("Class 1")
        classVal = 255 #1

    if (probVal<=probTh):
        #print("Class 0")
        classVal = 0


    return classVal


#####  MODEL LOADED


#modelPath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\roads_annotated\ds\_json'
modelPath = r'C:\Users\INKOM06\Pictures\roadDataset\CSC1202T\_json'
fileList = os.listdir(modelPath)

#idx = 0
#for fileNm in fileList:
#    print("%d   %s"%(idx,fileNm))
#    idx = idx + 1

#idx = int(input("Which model that you want? "))
idx = 1

jsonPath = os.path.join(modelPath,fileList[idx])
print(jsonPath)

json_file = open(jsonPath, 'r') 
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.summary()



#idx = 0
#for fileNm in fileList:
#    print("%d   %s"%(idx,fileNm))
#    idx = idx + 1

#idx = int(input("Which weights that you want to load? "))
idx = 0
hdfPath = os.path.join(modelPath,fileList[idx])
model.load_weights(hdfPath)

########

#path = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\"
#path0 = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\oregon_us\\"
#path0 = "C:\\Users\\INKOM06\\Pictures\\_DATASET\\_CIBISurvey2\\1202_start\\"
path0 = "C:\\Users\\INKOM06\\Pictures\\_DATASET\\_CIBISurvey2\\1202_start\\test\\"
#videoFile = "oreclip.mp4"
videoFile = "csc3.mp4"
#videoFile = "oregon.mp4"

#path0 = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\california\\"
#videoFile = "california1.mp4"




cap = cv2.VideoCapture(path0+videoFile)

totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

frameIdx = 0
ratio = 0.1
ratio2 = 1
ratio2b = 0.5
ret, frame = cap.read()
M = frame.shape[0]
N = frame.shape[1]

Mr = int(ratio*M)
Nr = int(ratio*N)

#Mv = int(Mr/ratio2)
#Nv = int(Nr/ratio2)


#Mv = 324#216
#Nv = 576#384

Mv = 216
Nv = 384
saveVideo = True
#videoPathToSave = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\oregon_us\\roads_annotated\\ds\\video\\"
videoPathToSave = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\CSC1202T\\video\\"

if saveVideo == True:
    out = cv2.VideoWriter(videoPathToSave+"outputlines_csc.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 30, (Nv,Mv))


    #out = cv2.VideoWriter('Video_output.avi', fourcc, FPS, (width,height), False)
sampPixCol = 5#3
sampPixRow = 5#3
color = (0, 255, 0) 
color2 = (250, 250, 60) 
thickness = 1



while(True) and (frameIdx<(totalFrames-1)):

    ret, frame = cap.read()

    frame = cv2.resize(frame, (Nr,Mr),interpolation = cv2.INTER_AREA)
    #frame2 = np.zeros((Mr,Nr,3), dtype = np.uint8)
    frame2 = frame

    M = frame.shape[0]
    N = frame.shape[1]

    imgBin = np.zeros((M,N), dtype = np.uint8)

    iN = 0
    for i in range(0,M,sampPixRow):
        jMin = N
        jMax = 0
        idxJ = 0
        for j in range(0,N,sampPixCol):

            iNorm = i/M - 0.5
            jNorm = j/N - 0.5
            rNorm = float(frame[i,j,2]/255) - 0.5
            gNorm = float(frame[i,j,1]/255) - 0.5
            bNorm = float(frame[i,j,0]/255) - 0.5

            classVal = classify(iNorm, jNorm, rNorm, gNorm, bNorm)
            imgBin[i,j] = classVal

            if classVal==255:
                frame2[i,j,2] = 255   
                frame2[i,j,1] = 255   
                frame2[i,j,0] = 0
                if j<jMin:
                    jMin = j
                if j>jMax:
                    jMax = j
        
        if ((jMin !=N) & (jMax != 0)):
            #print("i: %d    jMin: %d   jMax: %d "%(i,jMin,jMax))


            if (iN == 0):
                start_point = (jMin,i)
                start_point2 = (jMax,i)

            if ((iN>0)&(jMin != 0)):
                end_point = (jMin,i)
                frame2 = cv2.line(frame2, start_point, end_point, color, thickness)
                start_point = end_point                 


            if ((iN>0)&(jMax != N)):
                end_point2 = (jMax,i)
                frame2 = cv2.line(frame2, start_point2, end_point2, color2, thickness)
                start_point2 = end_point2                 


            iN = iN + 1





    #frame3 = cv2.resize(frame2, (int(N/ratio2b),int(M/ratio2b)))
    frame3 = cv2.resize(frame2, (Nv,Mv))
    cv2.imshow("RGB masked",frame3)
    print("Frame index: %d  of Total Frames: %d" %(frameIdx, totalFrames))


    if saveVideo == True:
        out.write(frame3)







    frameIdx = frameIdx + 1
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
