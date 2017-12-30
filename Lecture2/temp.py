# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf

W = tf.Variable(10)
assign_op = W.assign(100)

with tf.Session() as sess:
    sess.run(assign_op)
    print(W.eval())