# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 22:57:50 2017
MNIST using Logic Model
@author: Administrator
"""
import time
import numpy as np
import tensorflow as tf
from data_prepare import read_data_sets

#Step1: Read in data
MNIST = read_data_sets(".\\data",one_hot=True)

#Step 2: Define parameters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 25

#Step 3: create placeholders for features and labels
#each image in the MINST data is of shape 28*28 =784
#therefore, each image is represented with a 1X787 tensor
#there are 10 classes for each image, corresponding to digits 0-9
#each label is one hot vector
X = tf.placeholder(tf.float32,shape=(batch_size,784),name='X')
Y = tf.placeholder(tf.float32,shape=(batch_size,10),name='Y')

#Step 4: create weights and bias
#w is initialized to random variable with mean of 0, stddev of 0.01
#b is initialized to 0
#shape of w depends on the dimension of X and Y so that Y=tf.matmul(X,w)
#shape of b depends on Y
w = tf.Variable(tf.random_normal(shape=[784,10],stddev=0.01),name='weight')
b = tf.Variable(tf.zeros([1,10]),name="bias")

#Step 5: predict Y from X and w,b
#the model that returns probability distribution of possible lable of image
#through the softmax layer
#a batch_size x 10 tensor that represents the possibility of the digits
logits = tf.matmul(X,w)+b

#Step 6: define loss function
#use softmax cross entropy with logits as the loss function
#compute mean cross entropy, softmax is applied internally
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=logits)
loss = tf.reduce_mean(entropy) #compute the mean over examples in the batch

#Step 7: define training op
#using gradient descent with learning rate of 0.01 to minimize cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    n_batches = MNIST.train.num_examples//batch_size
    for _ in range(n_epochs):
        start = time.time()
        for _ in range(n_batches):
            X_batch,Y_batch = MNIST.train.next_batch(batch_size)
            sess.run(train,feed_dict={X:X_batch,Y:Y_batch})
        end = time.time()     
        print("each epoch model run time:%s" %(end-start))
    
    #test the model
    total_correct_preds = 0
    n_batches = MNIST.test.num_examples//batch_size
    for _ in range(n_batches):
        X_batch, Y_batch=MNIST.test.next_batch(batch_size)
        preds = tf.nn.softmax(sess.run(logits,feed_dict={X:X_batch,Y:Y_batch}))
        correct_preds = tf.equal(tf.argmax(preds,1),tf.argmax(Y_batch,1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds,tf.float32))
        total_correct_preds += sess.run(accuracy)
    print("Accuracy:%f" % (total_correct_preds/MNIST.test.num_examples))

        


