import numpy as np
import cv2
import os


#cap = cv2.VideoCapture('C:\\Users\\INKOM06\\Documents\\[0--KEGIATAN-Ku-2020\\2020.01-006-Autonomous Vehicle Project\\road.mp4')


cap = cv2.VideoCapture("C:\\Users\\INKOM06\\Videos\\Captures\\traffic.mp4")

totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(totalFrames)


frameIdx = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    pct = (frameIdx/totalFrames)*100

    M = frame.shape[0]
    N = frame.shape[1]
    #print("M: %d and N: %d "%(M,N))
    ratio = 0.5

    frame = cv2.resize(frame, (int(ratio*N),int(ratio*M)))

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

cap.release()
cv2.destroyAllWindows()


'''

    #edges = cv2.Canny(img,50,150,apertureSize = 3)
    #lines = cv2.HoughLines(edges,1, np.pi/180,200)

    #if np.isnan(lines) == true:
    #    print('Nan')

    #N = lines.size

    #print("Hello %d" % len(lines))

    

    if lines != {}:
        for line in lines:
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))

            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow('Image',frame)


 


    #ret,th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    #ret,th2 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    #edges = cv2.Canny(th1,90,100,5)



    # Display the resulting frame
    #cv2.imshow('frame-Binary',frame)
    #cv2.imshow('frame2-Inv Binary',th2)
    #cv2.imshow('frame3- Edges',edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

'''