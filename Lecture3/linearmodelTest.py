# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 22:09:47 2017
Linear Model to test
@author: Administrator
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

X_input = np.linspace(-1,1,100)
Y_input = X_input*3 + np.random.randn(X_input.shape[0])*0.5

w=tf.Variable(0.0,name='Weight',trainable=True)
b=tf.Variable(0.0,name='bias',trainable=True)

X=tf.placeholder(tf.float32,name='X')
Y=tf.placeholder(tf.float32,name='Y')

Y_predicted = X * w + b
loss = tf.square(Y-Y_predicted,name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(100):
        for x,y in zip(X_input,Y_input):
            sess.run(train,feed_dict={X:x,Y:y})
        
    w_value,b_value=sess.run([w,b])
    print(w_value,b_value)
    
plt.plot(X_input,Y_input,'o')
plt.plot(X_input,X_input*w_value+b_value,'g-')


