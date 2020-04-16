import cv2
import os
import matplotlib.pyplot as plt

os.system("cls")



#path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\largeDataSet\\true\\"
labelPath ="C:\\Users\\INKOM06\\Pictures\\roadlane-detection-evaluation-2013\\data_road\\training\\gt_image_2\\"
oriPath ="C:\\Users\\INKOM06\\Pictures\\roadlane-detection-evaluation-2013\\data_road\\training\\image_2\\"

#idx = int(input("Image index that needs to be analysed? "))
idx = 0



labFiles = []
# r=root, d=directories, f = files
for r, d, f in os.walk(labelPath):
	for file in f:
		labFiles.append(os.path.join(r, file))


oriFiles = []
# r=root, d=directories, f = files
for r, d, f in os.walk(oriPath):
	for file in f:
		oriFiles.append(os.path.join(r, file))



labImg = cv2.imread(labFiles[idx])
oriImg = cv2.imread(oriFiles[idx])



M = labImg.shape[0]
N = labImg.shape[1]
ratio = 0.8

labImg = cv2.resize(labImg, (int(ratio*N),int(ratio*M)))
oriImg = cv2.resize(oriImg, (int(ratio*N),int(ratio*M)))




layImg =oriImg.copy()

M = layImg.shape[0]
N = layImg.shape[1]

print(M)
print(N)


for i in range(M):
	for j in range (N):
		b = labImg[i,j,0]
		g = labImg[i,j,1]
		r = labImg[i,j,2]
		if ((r==255)&(b==255)):
			layImg[i,j,0] = 0
			#layImg[i,j,1] = 0
			#layImg[i,j,2] = 0


b,g,r =  cv2.split(labImg)
rgb_labImg = cv2.merge([r,g,b])

b,g,r = cv2.split(oriImg)       # get b,g,r
rgb_oriImg = cv2.merge([r,g,b])     # switch it to rgb

b,g,r = cv2.split(layImg)       # get b,g,r
rgb_layImg = cv2.merge([r,g,b])     # switch it to rgb


#cv2.imshow("labImg ",labImg)
#cv2.imshow("oriImg ",oriImg)
#cv2.imshow("layImg ",layImg)


fig, axs = plt.subplots(3,1)
fig.suptitle("Lane area segmentation")
axs[0].imshow(rgb_oriImg)
#axs[1].imshow(labImg)
axs[1].imshow(rgb_labImg)
axs[2].imshow(rgb_layImg)
plt.show()


cv2.waitKey()
cv2.destroyAllWindows()
