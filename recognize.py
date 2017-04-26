import tensorflow as tf
import numpy as np
import time
from PIL import Image
from data import *
import matplotlib.pyplot as plt

IMAGE_HEIGHT=20
IMAGE_WIDTH=80
IMAGE_CHANNEL=1
BATCH_SIZE=1
VOCALENTH=36
VOCADIM=4
TRAIN_IMAGE_PATH='./train_image.npy'
TRAIN_LABEL_PATH='./label.npy'
LEARNING_RATE=0.01
NUM_EPOCHS=1
TRAIN_SIZE=1829
num=4
TEST_PATH='./merge_data/'

input_data=tf.placeholder(tf.float32,shape=(BATCH_SIZE,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNEL),name='input_data')
conv_w1=tf.Variable(tf.truncated_normal([5,5,1,32],mean=0.0,stddev=1.0,dtype =tf.float32))
conv_b1=tf.Variable(tf.zeros([32],dtype=np.float32))
conv_w2=tf.Variable(tf.truncated_normal([3,3,32,64],mean=0.0,stddev=1.0,dtype=tf.float32))
conv_b2=tf.Variable(tf.zeros([64],dtype=tf.float32))
fcon_w1=tf.Variable(tf.truncated_normal([25*64,6400],mean=0.0,stddev=1.0,dtype=tf.float32))
fcon_b1=tf.Variable(tf.truncated_normal([6400],mean=0.0,stddev=1.0,dtype=tf.float32))
fcon_w2=tf.Variable(tf.truncated_normal([6400,VOCALENTH],mean=0.0,stddev=1.0,dtype=tf.float32))
fcon_b2=tf.Variable(tf.truncated_normal([VOCALENTH],mean=0.0,stddev=1.0,dtype=tf.float32))
keep_prob=tf.placeholder(tf.float32)

def model(data):
	data_splits=tf.split(value=data,num_or_size_splits=4,axis=2)
	final_outputs=[]
	for i in data_splits:
		conv_1=tf.nn.conv2d(i,conv_w1,strides=[1,1,1,1],padding='SAME')
		conv1_relu=tf.nn.relu(tf.nn.bias_add(conv_1,conv_b1))
		pool_1=tf.nn.max_pool(conv1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
		conv_2=tf.nn.conv2d(pool_1,conv_w2,strides=[1,1,1,1],padding='SAME')
		conv2_relu=tf.nn.relu(tf.nn.bias_add(conv_2,conv_b2))
		pool_2=tf.nn.max_pool(conv2_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
		outs=tf.reshape(pool_2,[BATCH_SIZE,-1])
		s1_out=tf.nn.relu(tf.matmul(outs,fcon_w1)+fcon_b1)
		s_dropout=tf.nn.dropout(s1_out,keep_prob)
		s2_out=tf.matmul(s1_out,fcon_w2)+fcon_b2
		final_outputs.append(s2_out)
	return final_outputs



def image_process(image_path):
	insert_matrix=np.zeros((22,2))
	image=np.array(Image.open(image_path).convert('L'),dtype=np.float32)
	for i in range(num):
		if i==0:
			im=np.hstack((image[:,:18],insert_matrix))	
		else:
			im=np.hstack((im,np.hstack((image[:,14*i:14*i+18],insert_matrix))))
	image=im[:20,:]
	image_data=image.astype(np.float32)
	image_data=image_data[np.newaxis,:,:,np.newaxis]
	return image_data

cnn_outputs=model(input_data)
saver=tf.train.Saver(tf.all_variables())

with tf.Session() as sess:
	tf.initialize_all_variables().run()
	saver.restore(sess,'./session')
	print('-------------------------Initialized----------------------------------')
	for i in os.listdir(TEST_PATH):
		fedict={input_data:image_process(TEST_PATH+i)}
		plt.imshow(Image.open(TEST_PATH+i))
		plt.show()
		outs=sess.run([cnn_outputs],feed_dict=fedict)
		idd=np.zeros((4))
		for i in range(num):
			idex=tf.argmax(outs[0][i][0],0)
			idd[i]=sess.run(idex)
		print(decode(idd[0])+decode(idd[1])+decode(idd[2])+decode(idd[3]))
		

