import numpy as np
import cv2
import os

def classify(img, ratio):
    M = img.shape[0]
    N = img.shape[1]
    img = cv2.resize(img, (int(ratio*N),int(ratio*M)),interpolation = cv2.INTER_AREA)
    return img


os.system("cls")

saveVideo = False
frameIdx = 0
ratio = 0.1


path0 = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\oregon_us\\"
videoFile = "oreclip.mp4"


cap = cv2.VideoCapture(path0+videoFile)
totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

ret, frame = cap.read()
M = frame.shape[0]
N = frame.shape[1]

M = int(ratio*M)
N = int(ratio*N)


videoPathToSave = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\oregon_us\\roads_annotated\\ds\\video\\"
if saveVideo == True:
    out = cv2.VideoWriter(videoPathToSave+"output.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 30, (N,M))


frameIdxMax = 2000
while(True) and (frameIdx<=frameIdxMax):

    ret, frame = cap.read()

    ## Fungsi deteksi letakkan di baris berikut ini
    frame = classify(frame, ratio)
    cv2.imshow("RGB frames",frame)
    

    print("Frame index: %d  of %d [%3.2f %%]" %(frameIdx, frameIdxMax,(float(frameIdx/frameIdxMax*100))))

    if saveVideo == True:
        out.write(frame)

    frameIdx = frameIdx + 1
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
