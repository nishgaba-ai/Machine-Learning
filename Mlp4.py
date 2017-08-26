# Imports and also data_provider import for CIFAR-10 and CIFAR-100 datasets
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import sys
import tarfile
from six.moves import urllib
import os
import tensorflow as tf
import numpy as np
import time
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
import matplotlib.pyplot as plt


%matplotlib inline
plt.style.use('ggplot')



train_data = CIFAR10DataProvider('train', batch_size=50)
valid_data = CIFAR10DataProvider('valid', batch_size=50)

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def new_biases(length):
    return tf.Variable(tf.constant(0.1, shape=[length],dtype=tf.float32))

def convolutional_layer(inputs,num_input_channels,filter_size,num_filters, nonlinearity=tf.nn.relu):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)
    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)
    
    outputs = tf.nn.conv2d(inputs, weights, [1, 1, 1, 1], padding='SAME') + biases
    outputs = nonlinearity(outputs)
    return outputs

def max_pooling(inputs):
    # Max Pooling
    pool = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME',name='pool')
    output = pool
    return output
# ELU GLORIOT INITIALIZATION
def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs

# ELU
def fully_connected_layer_elu(inputs, input_dim, output_dim, nonlinearity=tf.nn.elu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    
    return outputs

# Leaky RELU
def fully_connected_layer_leakrelu(inputs, input_dim, output_dim):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = (tf.matmul(inputs, weights) + biases)
    alpha = 0.01
    outputs = tf.maximum(alpha*outputs,outputs)     
    return outputs

# Dropout with RELU
def fully_connected_layer_dropout_relu(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    inputs = tf.nn.dropout(inputs, dropout)
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs

# Dropout with ELU
def fully_connected_layer_dropout_elu(inputs, input_dim, output_dim, nonlinearity=tf.nn.elu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    inputs = tf.nn.dropout(inputs, dropout)
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    
    return outputs

# Leaky RELU
def fully_connected_layer_dropout_leakrelu(inputs, input_dim, output_dim):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    inputs = tf.nn.dropout(inputs, dropout)
    outputs = (tf.matmul(inputs, weights) + biases)
    alpha = 0.01
    outputs = tf.maximum(alpha*outputs,outputs)     
    return outputs
    
    inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')

with tf.name_scope('conv-layer-1'):
    shape = [-1, 32, 32, 3]
    inputs = tf.reshape(inputs, shape=shape)
    conv_1 = convolutional_layer(inputs,3,5,32) 
    
# pooling    
with tf.name_scope('max-pool-1'):    
    conv_1 = max_pooling(conv_1)  
    # normalization
    #conv_1 = tf.nn.lrn(conv_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1') 
                          
        
        
        
    
with tf.name_scope('Conv-2'): 
    conv_2 = convolutional_layer(conv_1,32,5,64) 
    
    # normalization
    #conv_2 = tf.nn.lrn(conv_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
    # pooling
with tf.name_scope('max-pool-2'):    
    conv_2 = max_pooling(conv_2)  
    
                         
new_shape = tf.reshape(conv_2,[-1,8*8*64])

with tf.name_scope('fc-layer-1'):
    fc_1 = fully_connected_layer(new_shape, 8*8*64, 384)

with tf.name_scope('fc-layer-2'):
    fc_2= fully_connected_layer(fc_1, 384,384)
    
    
with tf.name_scope('Softmax'): 
    fc_3=fully_connected_layer(fc_2, 384,10)
   
    outputs =fc_3
with tf.name_scope('error'):
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer().minimize(error)
    
    init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(100):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            input_batch = np.reshape(input_batch,(-1, 32, 32, 3))  
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
        if (e + 1) % 5 == 0:
            valid_error = 0.
            valid_accuracy = 0.
            for input_batch, target_batch in valid_data:
                input_batch = np.reshape(input_batch,(-1, 32, 32, 3))  
                batch_error, batch_acc = sess.run(
                    [error, accuracy], 
                    feed_dict={inputs: input_batch, targets: target_batch})
                valid_error += batch_error
                valid_accuracy += batch_acc
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))
                   
# Plot the change in the validation and training set error over training.    
fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
ax_1.set_ylabel('Training Error')
ax_1.plot(epo,err_t_1,label='Adam')
ax_1.plot(epo, err_t_2,label='GradDes')
ax_1.plot(epo, err_t_3, label='AdDel')
ax_1.plot(epo, err_t_4, label='AdaGrad')
ax_1.plot(epo, err_t_6, label='Momentum')
ax_1.plot(epo, err_t_7, label='Ftrl')
ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')
