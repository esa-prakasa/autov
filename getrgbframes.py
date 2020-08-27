import numpy as np
import cv2
import os


#os.system("cls")

path = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\"
path0 = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\oregon_us\\"

videoFile = "oreclip.mp4"

cap = cv2.VideoCapture(path0+videoFile)

totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

frameIdx = 0
ratio = 0.3

'''


'''
while(True) and (frameIdx<1000):
    
   

    fileName = str(frameIdx+100000)
    fileName = fileName[1:]
    fileName = fileName+".png"
    print(fileName)



    ret, frame = cap.read()
    pct = (frameIdx/totalFrames)*100

    M = frame.shape[0]
    N = frame.shape[1]

    frame = cv2.resize(frame, (int(ratio*N),int(ratio*M)))
    cv2.imshow("RGB",frame)


    cv2.imwrite(path0+"rgb\\"+fileName,frame)

    frameIdx = frameIdx + 1
    
    '''


    '''

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
