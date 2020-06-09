import numpy as np
import cv2
import os
import statistics

os.system("cls")

path = "C:\\Users\\INKOM06\\Pictures\\NormalSmall\\"
videoFile = "NO20200216-145746-000495_s.mp4"
#videoFile = "NO20200216-143946-000477_s.mp4"  # kota baru
#videoFile = "NO20200206-083245-000471_s.mp4"  # lipi
#videoFile = "NO20200216-145646-000494_s.mp4"


cap = cv2.VideoCapture(path+videoFile)
totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#print(totalFrames)

frameIdx = 0

ret, frame = cap.read()
M = frame.shape[0]
N = frame.shape[1]

scaleRatio = 0.5
rM = int(scaleRatio*M)
rN = int(scaleRatio*N)

## =======================================
from keras.models import load_model
from keras.preprocessing import image

kfold = "_fold4"

rootPath   = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\Purbaleunyi_Highway_1\\ds\\"
modelPath  = rootPath+kfold+"\\xmodel\\"
outputPath = rootPath+kfold+"\\output\\"
#modelPath = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"+kfold+"\\xmodel\\"
modelName = "best_model_"+kfold+".h5"
model = load_model(modelPath+modelName)

print(modelPath+modelName)


def classfSingleSubRGB_CNN(subFrame):
    M = subFrame.shape[0]
    N = subFrame.shape[1]    
    test_image = image.img_to_array(subFrame)
    test_image = np.expand_dims(test_image, axis = 0)
    result = int(model.predict(test_image))
    #print(result)
    
    b,g,r = cv2.split(subFrame)
    zr = np.zeros((M,N),dtype="uint8")

    if (result==0):
        subFrame = cv2.merge([zr,g,zr])
    if result==1:
        subFrame = cv2.merge([zr,zr,r])

    return subFrame
    #print(str(i)+" --- "+files[i]+"--"+str(clsSelected)+" >>> "+str(result))


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
    #print(M)
    N = frame.shape[1]
    fSz = 30
    fSz2 = fSz//2

#    for i in range(int(M*0.5),(M-fSz),fSz):
    for i in range((M-(5*fSz)-1),(M-fSz),fSz):
        for j in range(0,(N-fSz2),fSz):
                #print("%d %d"%(i,j))
                ic = int(i)
                jc = int(j)
                #print(ic)
                subFrame = frame[ic:(ic+fSz), jc:(jc+fSz), :]
                mm = subFrame.shape[0]
                nn = subFrame.shape[1]
                #print(str(mm)+" "+str(nn))
                
                #if subFrame
                #subFrame = classfSingleSubRGB(subFrame)
                
                subFrame = classfSingleSubRGB_CNN(subFrame)
                frame[ic:(ic+fSz), jc:(jc+fSz), :] = subFrame
    return frame


M2 = int(2*0.8*M)
out = cv2.VideoWriter(outputPath+"output_"+kfold+videoFile[:-4]+".avi",cv2.VideoWriter_fourcc('M','J','P','G'), 30, (N,M2))


while(True) and (frameIdx<(totalFrames-1)):    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = frame[0:int(0.8*M),:,:]

    frame0 = frame.copy()

    frame = classfAllSubImages(frame)

    frame = cv2.vconcat([frame0, frame])
    
    out.write(frame)

    cv2.imshow("Frame", frame)
    frameIdx = frameIdx + 1

    print(str(frameIdx))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #frame = cv2.resize(frame,(rM,rN),interpolation = cv2.INTER_AREA)
'''

    #fileIdx = 10000 + frameIdx
    #fileIdxS = str(fileIdx)
    #fileIdxS = fileIdxS[1:]
    #imgFileName = fileIdxS+"__"+videoFile[:-4]+".jpg"
    #if ((frameIdx%10)==0):
    #    print(imgFileName)
     #   cv2.imwrite((pathToSaveFiles+imgFileName),frame)
'''
#####  out.release()


cap.release()
cv2.destroyAllWindows()
