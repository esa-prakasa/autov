import numpy as np
import cv2
import os

os.system('cls')


pathFolder = "C:\\Users\\INKOM06\\Pictures\\NormalSmall\\" 
#fileName = "NO20200216-150246-000500_s.mp4"
fileName = "NO20200216-150446-000502_s.mp4"
cap = pathFolder+fileName


cap = cv2.VideoCapture(cap)

totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(totalFrames)


scRatio = 0.9

ret, frame = cap.read()
M = frame.shape[0]  # column
N = frame.shape[1]  # row

print("M: row (height) ",M,"  N: col (width)", N)

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
M = frame.shape[0]  # row
N = frame.shape[1]  # col

print("Grayscale M: row (height) ",M,"  N: col (width)", N)

newRow = int(M*scRatio)
newCol = int(N*scRatio)


frameIdx = 0

for frameIdx in range(int(totalFrames-1)):
#while(True):
    ret, frame = cap.read()
    pct = (frameIdx/totalFrames)*100
    print('Frame index: %d Pct: %3.2f %s' % (frameIdx,pct,chr(37)))
    frameIdx = frameIdx + 1

    frame = cv2.resize(frame,(newCol,newRow) , interpolation = cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame2 = frame[int(newRow/2):newRow, 0:newCol]
    cv2.imshow('Original image',frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()

