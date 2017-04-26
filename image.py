import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time

IMAGEPATH="./merge_data/"
NUM=4
flag=1
insert_matrix=np.zeros((22,2))
idd=0
def transf(image,num):
	for i in range(num):
		if i==0:
			im=np.hstack((image[:,:18],insert_matrix))
		else:
			im=np.hstack((im,np.hstack((image[:,14*i:14*i+18],insert_matrix))))
	return im

for i in os.listdir(IMAGEPATH):
	path=IMAGEPATH+i
	image=np.array(Image.open(path).convert('L'),dtype=np.float32)
	im=transf(image,4)
	print(np.shape(im))
	'''
	if idd==0:
		print(i)
		plt.imshow(im)
		plt.show()
		time.sleep(100)
	'''	
	im=im[np.newaxis,:]
	if flag:
		flag=0
		image_data=im
	else:
		image_data=np.vstack((image_data,im))
		'''
		if idd==10:
			print(np.shape(image_data))
			
			a=np.reshape(image_data[10],[22,80])
			plt.imshow(a)
			plt.show()
		
			time.sleep(100)
		'''	
	idd+=1

image_data=image_data[:,:,:,np.newaxis]
image_data=image_data[:,:20,:,:]
image_data=image_data.astype(np.float32)
print(np.shape(image_data))
a=np.reshape(image_data[10],[20,80])
plt.imshow(a)
plt.show()

print(image_data.dtype)
np.save("train_image",image_data)
print("ok",np.shape(image_data))
