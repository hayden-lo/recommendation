# Author: Hayden Lao
# Script Name: forward
# Created Date: Oct 20th 2019
# Description: Basic tensorflow 1.14 practice for forward network

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

# set log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0:INFO,1:WARNING,2:ERROR,3:FATAL

# load data
(x, y), _ = datasets.mnist.load_data()
data = datasets.mnist.load_data()

# convert to tensor
x = tf.convert_to_tensor(x, dtype=tf.float32)  # [60k,28,28]
y = tf.convert_to_tensor(y, dtype=tf.int32)  # [60k]

# have a glance on the data
with tf.Session() as sess:
    print(x.shape, y.shape, x.dtype, y.dtype)
    print("x min value:", str(sess.run(tf.reduce_min(x))))
    print("x max value:", str(sess.run(tf.reduce_max(x))))
    print("y min value:", str(sess.run(tf.reduce_min(y))))
    print("y max value:", str(sess.run(tf.reduce_max(y))))

# make tensorflow dataset
train_db = tf.data.Dataset.from_tensor_slices((x, y))
a=train_db.batch(128)
train_iter = iter(train_db)
sample = next(train_iter)  # ([128,28,28], [128])
with tf.Session() as sess:
    print(train_db)
    print(a)
    print(train_iter)
    print(sample)
    # print("x batch shape:", sess.run(sample))
    # print("y batch shape:", sample[1].shape)
