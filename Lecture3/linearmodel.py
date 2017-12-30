# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 18:12:47 2017
Dataset Description:
Name: Fire and Theft in Chicago
X = fires per 1000 housing unitsY = thefts per 1000 population
within the same Zip code in the Chicago metro area
Total number of Zip code areas: 42

Linear Model using mean squared error as the loss function
@author: Administrator
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import xlrd


DATA_FILE = '.\\data\\slr05.xls'

#Step 1 :read in data from .xls file
book = xlrd.open_workbook(DATA_FILE,encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1,sheet.nrows)])
n_samples = sheet.nrows - 1


#Step 2: create placeholders for input X(number of fire) and label Y(number of theft)
X = tf.placeholder(tf.float32,name='X')
Y = tf.placeholder(tf.float32,name='Y')

#Step 3: create weight and bias, intialized to 0
#w1 = tf.Variable(0.0,name='Weights1')
w2= tf.Variable(0.0,name='Weights2')
b = tf.Variable(0.0,name='bias')

#Step 4: construct model to predict Y(number of theft) from the number of fire
Y_predicted = X*w2 + b

#Step 5: use the square error as the loss function
loss = tf.square(Y-Y_predicted,name="loss")

#Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train=optimizer.minimize(loss)
with tf.Session() as sess:
    #Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
    
    #Step 8: train the model
    for i in range(100): #run 100 epochs
        #total_loss=0
        for x,y in data:
            sess.run(train,feed_dict={X:x,Y:y})
            #total_loss+=l
       #print("Epoch {0}:{1}".format(i,total_loss/n_samples))                
    #Step 9:output the values of w and b
    w2_value,b_value = sess.run([w2,b])
    print(w2_value,b_value)


#X,Y=data.T[0],data.T[1]
X,Y=data[:,0],data[:,1]
plt.plot(X,Y,'bo',label='Real data')
plt.plot(X,X*w2_value+b_value,'r',label='Predicted value')
plt.legend()
plt.show()

    