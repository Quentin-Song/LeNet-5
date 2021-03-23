import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist =input_data.read_data_sets(r'D:\data\Dataset',one_hot=True)

batch_size =100
learning_rate =0.01
learning_rate_decay =0.99
max_steps =3000


def hidden_layer(input_tensor,regularizer,avg_class,resuse):
    #创建卷积层-1
    with tf.variable_scope("C1-conv",reuse=resuse):
        conv1_weights =tf.get_variable("weight",[5,5,1,32],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases =tf.get_variable('bias',[32],
                                      initializer=tf.constant_initializer(0.0))

        conv1 =tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')

        relu1 =tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    #创建池化层

    with tf.name_scope("S2-max_pool",):
        pool1 =tf.nn.max_pool(relu1,ksize =[1,2,2,1],
                              strides =[1,2,2,1],padding='SAME')
    #卷积层-2
    with tf.variable_scope("C3-conv", reuse=resuse):
        conv2_weights = tf.get_variable("weight", [5, 5, 1, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [64],
                                       initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')

        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    #池化层-2
    with tf.name_scope("S4-max_pool",):
        pool2 =tf.nn.max_pool(relu2,ksize =[1,2,2,1],
                              strides =[1,2,2,1],padding='SAME')

        shape =pool2.get_shape().as_list()
        nodes =shape[1]*shape[2]*shape[3]
        reshape = tf.reshape(pool2, [shape[0],nodes])


    #创建第一个全连接层

    with tf.variable_scope('layer5-full1',reuse=resuse):
        Full_connection1_weights =tf.get_variable('weights',[nodes,512],
                                                  initializer=tf.truncated_normal_initializer(stddev=0.1))

        tf.add_to_collection('losses',regularizer(Full_connection1_weights))

        Full_connection1_biases = tf.get_variable('bias',[512],initializer=tf.constant_initializer(0.1))

        if avg_class ==None:

            Full_1  = tf.nn.relu(tf.matmul(reshape,Full_connection1_weights) + Full_connection1_biases)

        else:
            Full_1 = tf.nn.relu(tf.matmul(reshape,avg_class.average(Full_connection1_weights))+\
                                avg_class.average(Full_connection1_biases))

    with tf.variable_scope('layer6-full2', reuse=resuse):
        Full_connection2_weights = tf.get_variable('weights', [512, 10],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.1))

        tf.add_to_collection('losses', regularizer(Full_connection2_weights))

        Full_connection2_biases = tf.get_variable('bias', [10], initializer=tf.constant_initializer(0.1))

        if avg_class == None:

            result = tf.matmul(Full_1, Full_connection2_weights) + Full_connection2_biases

        else:
            result = tf.matmul(Full_1, avg_class.average(Full_connection2_weights)) + \
                                avg_class.average(Full_connection2_biases)

    return result


x =tf.placeholder(tf.float32,[batch_size,28,28,1],name='x-input')

y_ =tf.placeholder(tf.float32,[None,10],name ='y-input')

regularizer =tf.contrib.layers.l2_regularizer(0.01)

y =hidden_layer(x,regularizer,avg_class=None,resuse=False)

training_step =tf.Variable(0,trainable=False)

variable_average= tf.train.ExponentialMovingAverage(0.99,training_step)

variable_average_op =variable_average.apply(tf.trainable_variables())
average_y =hidden_layer(x,regularizer,variable_average,resuse=True)

cross_entropy =tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))

cross_entropy_mean =tf.reduce_mean(cross_entropy)

loss =cross_entropy_mean+tf.add_n(tf.get_collection('losses'))

learning_rate =tf.train.exponential_decay(learning_rate,training_step,mnist.train.num_examples/batch_size,
                                          learning_rate_decay,staircase=True)
training_step =tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=training_step)

with tf.control_dependencies([training_step,variable_average_op]):
    train_op =tf.no_op(name='train')
crorent_predicition =tf.equal(tf.arg_max(average_y,1),tf.argmax(y_,1))
accuracy =tf.reduce_mean(tf.case(crorent_predicition,tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(max_steps):
        if i % 1000 ==0:


            x_val,y_val =mnist.validation.next_batch(batch_size)

            reshaped_x2 =np.reshape(x_val,(batch_size,28,28,1))

            validate_feed ={x:reshaped_x2,y_:y_val}

            validate_accuracy =sess.run(accuracy,feed_dict=validate_feed)
            print('after %d training step(s),validation accuracy'
                  'using average model is %g%%'%(i,validate_accuracy*100))
            x_train,y_train =mnist.train.next_batch(batch_size)
            reshaped_xs =np.reshape(x_train,(batch_size,28,28,1))

            sess.run(train_op,feed_dict={x:reshaped_xs,y_:y_train})











