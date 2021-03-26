"""
autor:Quentin
time:2021-03-26
"""
import tensorflow as tf
import numpy as np

# 常量初始化器
v1_cons = conv1_weights =tf.get_variable("weight",[5,5,1,32],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
v2_cons = tf.get_variable('v2_cons', shape=[1, 4], initializer=tf.constant_initializer(9))
# 正太分布初始化器
v1_nor = tf.get_variable('v1_nor', shape=[1, 4], initializer=tf.random_normal_initializer())
v2_nor = tf.get_variable('v2_nor', shape=[1, 4],
                         initializer=tf.random_normal_initializer(mean=0, stddev=5, seed=0))  # 均值、方差、种子值
# 截断正态分布初始化器
v1_trun = tf.get_variable('v1_trun', shape=[1, 4], initializer=tf.truncated_normal_initializer())
v2_trun = tf.get_variable('v2_trun', shape=[1, 4],
                          initializer=tf.truncated_normal_initializer(mean=0, stddev=5, seed=0))  # 均值、方差、种子值
# 均匀分布初始化器
v1_uni = tf.get_variable('v1_uni', shape=[1, 4], initializer=tf.random_uniform_initializer())
v2_uni = tf.get_variable('v2_uni', shape=[1, 4],
                         initializer=tf.random_uniform_initializer(maxval=-1., minval=1., seed=0))  # 最大值、最小值、种子值

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("常量初始化器v1_cons:", sess.run(v1_cons))
    print("常量初始化器v2_cons:", sess.run(v2_cons))
    print("正太分布初始化器v1_nor:", sess.run(v1_nor))
    print("正太分布初始化器v2_nor:", sess.run(v2_nor))
    print("截断正态分布初始化器v1_trun:", sess.run(v1_trun))
    print("截断正态分布初始化器v2_trun:", sess.run(v2_trun))
    print("均匀分布初始化器v1_uni:", sess.run(v1_uni))
    print("均匀分布初始化器v2_uni:", sess.run(v2_uni))
