import tensorflow as tf
import numpy as np
import time
from PIL import Image
from data import *

IMAGE_HEIGHT=20
IMAGE_WIDTH=80
IMAGE_CHANNEL=1
BATCH_SIZE=100
VOCALENTH=36
VOCADIM=4
TRAIN_IMAGE_PATH='./train_image.npy'
TRAIN_LABEL_PATH='./label.npy'
LEARNING_RATE=0.01
NUM_EPOCHS=1000
TRAIN_SIZE=1829

input_data=tf.placeholder(tf.float32,shape=(BATCH_SIZE,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNEL),name='input_data')
label_data=tf.placeholder(tf.float32,shape=(BATCH_SIZE,VOCADIM,VOCALENTH),name='label_data')
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

def cross_loss(x,y):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=x))

def get_data(train_x_path,train_y_path):
	x=np.load(train_x_path)
	y=np.load(train_y_path)
	return x,y

	
#def error_rate(predictions,labels):


def train(data_x,data_y,batch_size):
	cnn_outputs=model(data_x)
	data_y=tf.transpose(data_y,[1,0,2])
	full_loss=0.0
	accuracy=0.0
	for i in range(VOCADIM):
		full_loss+=cross_loss(cnn_outputs[i],data_y[i])
		correct_prediction=tf.equal(tf.argmax(cnn_outputs[i],1),tf.argmax(data_y[i],1))
		accuracy+=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	accuracy/=4.0
	regularizer=tf.nn.l2_loss(fcon_w1)+tf.nn.l2_loss(fcon_b1)+tf.nn.l2_loss(fcon_w2)+tf.nn.l2_loss(fcon_b2)
	loss=tf.reduce_mean(full_loss)#+1e-4*regularizer
	return loss,accuracy,cnn_outputs


loss,accuracy,cnn_outputs=train(input_data,label_data,BATCH_SIZE)
optimizer=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
saver=tf.train.Saver(tf.all_variables())

with tf.Session() as sess:
	tf.initialize_all_variables().run()
	saver.restore(sess,'./session')
	print('Initialized')
	data,labels=get_data(TRAIN_IMAGE_PATH,TRAIN_LABEL_PATH)
	print(np.shape(data[:100]),np.shape(labels[:100]))
	for i in range(NUM_EPOCHS):
		idex=np.random.choice(TRAIN_SIZE,BATCH_SIZE)
		batch_data=data[idex]
		batch_labels=labels[idex]
		fedict={input_data:batch_data,label_data:batch_labels,keep_prob:1.0}
		cost,_,train_rate,outs=sess.run([loss,optimizer,accuracy,cnn_outputs],feed_dict=fedict)
		idex=tf.argmax(outs[0][0],0)
		idd=sess.run(idex)
		print(train_rate)
		if i%20==0:
			print(decode(idd),decode1(batch_labels[0][0]))
		#print(cost)
	saver_path=saver.save(sess,"./session")



