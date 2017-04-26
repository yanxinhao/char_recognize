import os
import numpy as np
import string
import time
import tensorflow as tf

IMAGEPATH="./merge_data"
LABELPATH="label.npy"
DINUM=36
numb=np.arange(10)
num=[]
for i in range(10):
	num.append(str(numb[i]))
DICT1=dict(zip(num,numb))
DICT2=dict(zip(string.ascii_lowercase,np.arange(26)+10))
DICT3=dict(zip(string.ascii_uppercase,np.arange(26)+10))
d=dict(DICT1,**DICT2)
DICT=dict(d,**DICT3)

def dict_find(x):
	label=np.zeros([len(x),4,DINUM])
	for n in  range(len(x)):	
		for i in range(4):
			for key,value in DICT.items() :
				if x[n][i]==key:
					label[n,i,value]=1
					break
	return label
	
		
def main(): 
	idd=0
	a=np.array(os.listdir(IMAGEPATH))
	b=[]
	for name in a:
		if idd==22:	
			print(name)
			#print(b[9])
			print(b)
			#time.sleep(100)
		b.append(name[:4])
		print(b[idd])
		label_string=np.array(b)
		idd+=1
	label=dict_find(label_string)
	print(np.shape(label))
	np.save('label',label)

def decode(arr):
	#print(tf.argmax(arr,0))
	for key,value in DICT.items():
		if value==arr:
			return key
		

def decode1(arr):
	for i in range(len(arr)):	
		if arr[i]==1:
			for key,value in DICT.items():
				if i==value:
					return key
#a=list([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
#print(decode(a))
#main()
