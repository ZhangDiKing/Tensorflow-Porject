# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:15:10 2017

@author: zhang
"""
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pylab
import pickle
#load data
#you can change the path to load npz
path='C:/Users/zhang/dp/dP_final/'
npzfile = np.load(path+"train_and_val.npz")
train_eye_left = npzfile["train_eye_left"]
train_eye_right = npzfile["train_eye_right"]
train_face = npzfile["train_face"]
train_face_mask = npzfile["train_face_mask"]
train_y = npzfile["train_y"]
val_eye_left = npzfile["val_eye_left"]
val_eye_right = npzfile["val_eye_right"]
val_face = npzfile["val_face"]
val_face_mask = npzfile["val_face_mask"]
val_y = npzfile["val_y"]
# %%
#definition of variables
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#definition of convolution and pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#definition of summaries
def variable_summaries(var):
  with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.histogram('histogram', var)
#definition of nn layers and cnn layers
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
    with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

def cnn_layer(input_tensor, filter_shape,input_dim, output_dim, layer_name, act=tf.nn.relu):
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
        weights = weight_variable([filter_shape[0],filter_shape[1],input_dim, output_dim])
        variable_summaries(weights)
    with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
    with tf.name_scope('convolution_w_input_plus_b'):
        preactivate = conv2d(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations
# %%
#initialize parameters
in_channels=3
batch_size=250
hc_d=32
# %%
#define input
with tf.name_scope('eye_left_input'):
    eye_left=tf.placeholder('float32',[batch_size,64,64,3]);
with tf.name_scope('eye_right_input'):
    eye_right=tf.placeholder('float32',[batch_size,64,64,3]);
with tf.name_scope('face_input'):    
    face=tf.placeholder('float32',[batch_size,64,64,3]);
with tf.name_scope('face_mask_input'):    
    face_mask=tf.placeholder('float32',[batch_size,25,25]);
with tf.name_scope('label_input'):    
    y=tf.placeholder('float32',[batch_size,2]);
# %%
#define the parameters in convolution layers of left eye
# convolution layer 1 for left eye  
h_conv1_left_eye=cnn_layer(eye_left,[5,5],in_channels, 32,'conv1_left_eye')#60,60,32
with tf.name_scope('conv1_left_eye_pooling'):
    h_pool1_left_eye=max_pool_2x2(h_conv1_left_eye) #30,30,32

# convolution layer 2 for left eye  
h_conv2_left_eye=cnn_layer(h_pool1_left_eye,[7,7],32, 32,'conv2_left_eye')#24,24,32
with tf.name_scope('conv2_left_eye_pooling'):
    h_pool2_left_eye=max_pool_2x2(h_conv2_left_eye) #12,12,32

# convolution layer 3 for left eye  
h_conv3_left_eye=cnn_layer(h_pool2_left_eye,[5,5],32, 32,'conv3_left_eye')#8,8,32
    
with tf.name_scope('conv3_left_eye_drop_out'):
    h_conv3_left_eye_flat=tf.reshape(h_conv3_left_eye, [tf.shape(h_conv3_left_eye)[0], 8*8*32])
    h_left_eye_drop      =tf.nn.dropout(h_conv3_left_eye_flat, 0.5)

# fully connected layer 1 for left eye
h_fc1_left_eye=nn_layer(h_left_eye_drop, 8*8*32, hc_d, 'fc1_left_eye')
# %%
#define the parameters in convolution layers of right eye
# convolution layer 1 for right eye  
h_conv1_right_eye=cnn_layer(eye_right,[5,5],in_channels, 32,'conv1_right_eye')#60,60,32
with tf.name_scope('conv1_right_eye_pooling'):
    h_pool1_right_eye=max_pool_2x2(h_conv1_right_eye) #30,30,32

# convolution layer 2 for eight eye  
h_conv2_right_eye=cnn_layer(h_pool1_right_eye,[7,7],32, 32,'conv2_right_eye')#24,24,32
with tf.name_scope('conv2_right_eye_pooling'):
    h_pool2_right_eye=max_pool_2x2(h_conv2_right_eye) #12,12,32

# convolution layer 3 for right eye  
h_conv3_right_eye=cnn_layer(h_pool2_right_eye,[5,5],32, 32,'conv3_right_eye')#8,8,32
    
with tf.name_scope('conv3_right_eye_drop_out'):
    h_conv3_right_eye_flat=tf.reshape(h_conv3_right_eye, [tf.shape(h_conv3_right_eye)[0], 8*8*32])
    h_right_eye_drop      =tf.nn.dropout(h_conv3_right_eye_flat, 0.5)

# fully connected layer 1 for right eye
h_fc1_right_eye=nn_layer(h_right_eye_drop, 8*8*32, hc_d, 'fc1_right_eye')      
# %%
#define the parameters in convolution layers of face
# convolution layer 1 for face  
h_conv1_face=cnn_layer(face,[5,5],in_channels, 32,'conv1_face')#60,60,32
with tf.name_scope('conv1_face_pooling'):
    h_pool1_face=max_pool_2x2(h_conv1_face) #30,30,32

# convolution layer 2 for face  
h_conv2_face=cnn_layer(h_pool1_face,[7,7],32, 32,'conv2_face')#24,24,32
with tf.name_scope('conv2_face_pooling'):
    h_pool2_face=max_pool_2x2(h_conv2_face) #12,12,32

# convolution layer 3 for face  
h_conv3_face=cnn_layer(h_pool2_face,[5,5],32, 32,'conv3_face')#8,8,32
    
with tf.name_scope('conv3_face_drop_out'):
    h_conv3_face_flat=tf.reshape(h_conv3_face, [tf.shape(h_conv3_face)[0], 8*8*32])
    h_face_drop      =tf.nn.dropout(h_conv3_face_flat, 0.5)

# fully connected layer 1 for face
h_fc1_face=nn_layer(h_face_drop, 8*8*32, hc_d, 'fc1_face')    
# %%
# fully connected layer 1 for face mask
with tf.name_scope('face_mask_drop_out'):
    h_face_mask_flat=tf.reshape(tf.cast(face_mask,tf.float32), [tf.shape(face_mask)[0], 25*25])
    h_face_mask_drop=tf.nn.dropout(h_face_mask_flat, 0.5)
# fully connected layer 1 for right eye
h_fc1_face_mask=nn_layer(h_face_mask_drop, 25*25, hc_d, 'fc1_face_mask')

# fully connected layer 2 for face mask
h_fc2_face_mask=nn_layer(h_fc1_face_mask, hc_d, hc_d, 'fc2_face_mask')
# %%
# fully connected layer 1 for eyes
#due to different version of tensorflow, this sentense may be changed
h_eye_flat=tf.concat([h_fc1_left_eye, h_fc1_right_eye],1)
#h_eye_flat=tf.concat(1,[h_fc1_left_eye, h_fc1_right_eye])
#h_eye_drop=tf.nn.dropout(h_face_mask_flat, 0.5)
h_fc1_eye =nn_layer(h_eye_flat, 2*hc_d, hc_d, 'fc1_eye')

# fully connected layer 1 for eyes,face,face mask
#due to different version of tensorflow, this sentense may be changed
h_flat             =tf.concat([tf.concat([h_fc1_eye, h_fc1_face],1),h_fc2_face_mask],1)
#h_flat             =tf.concat(1,[tf.concat(1,[h_fc1_eye, h_fc1_face]),h_fc2_face_mask])
h_fc1_face_eye_mask=nn_layer(h_flat, 3*hc_d, hc_d, 'fc1_eye_face_mask')

# %%

with tf.name_scope('final_out'):
    W_out     =weight_variable([hc_d, 2])
    B_out     =bias_variable([2])
    #h_eye_drop = tf.nn.dropout(h_face_mask_flat, 0.5)
    predict_op=tf.matmul(h_fc1_face_eye_mask, W_out) + B_out

# %%
#loss and training operation
with tf.name_scope('loss_and_error'):
    loss =tf.reduce_mean(tf.pow(predict_op-y, 2))/2
    error=tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(predict_op-y, 2),reduction_indices=1)))
tf.summary.scalar('loss', loss)
tf.summary.scalar('err', error)

with tf.name_scope('train_op'):
    learning_rate=0.001
    train_step   =tf.train.RMSPropOptimizer(learning_rate, decay=0.5).minimize(loss)
#save models
with tf.name_scope('save_model'):
    tf.get_collection("validation_nodes")
    tf.add_to_collection("validation_nodes", eye_left)
    tf.add_to_collection("validation_nodes", eye_right)
    tf.add_to_collection("validation_nodes", face)
    tf.add_to_collection("validation_nodes", face_mask)
    tf.add_to_collection("validation_nodes", predict_op)
    saver = tf.train.Saver()
# %%
# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
sess        =tf.Session()
merged      =tf.summary.merge_all()
train_writer=tf.summary.FileWriter(path + '/train',
                                      sess.graph)
#test_writer = tf.summary.FileWriter(path + '/test',sess.graph)
sess.run(tf.global_variables_initializer())
# %%
start     =time.clock()
min_err   =2.6
train_size=48000
epoch     =50
test_size =5000
train_er  =[]
test_er   =[]
cost      =[]
print('start training')
for k in range(0,epoch):
    
        
    #compute the trainning accuracy and cost
    train_error=0
    _cost      =0
    for i in range(int(train_size/batch_size)):
        train_eye_left_batch =train_eye_left[int(i*batch_size):int((i+1)*batch_size)];
        train_eye_right_batch=train_eye_right[int(i*batch_size):int((i+1)*batch_size)];
        train_face_batch     =train_face[int(i*batch_size):int((i+1)*batch_size)];
        train_face_mask_batch=train_face_mask[int(i*batch_size):int((i+1)*batch_size)];
        train_y_batch        =train_y[int(i*batch_size):int((i+1)*batch_size)];
        _cross_entropy,_error=sess.run([loss,error],
                                       feed_dict={
                                               eye_left: train_eye_left_batch/ 255.-0.5, 
                                               eye_right: train_eye_right_batch/ 255.-0.5,
                                               face:train_face_batch/ 255.-0.5,
                                               face_mask:train_face_mask_batch,
                                               y:train_y_batch
                                               })
        train_error = train_error+_error
        _cost=_cost+_cross_entropy        
    #compute the test accuracy
    test_error=0
    i         =0
    for i in range(int(test_size/batch_size)):
        val_eye_left_batch =val_eye_left[int(i*batch_size):int((i+1)*batch_size)];
        val_eye_right_batch=val_eye_right[int(i*batch_size):int((i+1)*batch_size)];
        val_face_batch     =val_face[int(i*batch_size):int((i+1)*batch_size)];
        val_face_mask_batch=val_face_mask[int(i*batch_size):int((i+1)*batch_size)];
        val_y_batch        =val_y[int(i*batch_size):int((i+1)*batch_size)];
        test_error         =test_error+error.eval(session=sess, 
                                              feed_dict={
                                                      eye_left: val_eye_left_batch/ 255.-0.5, 
                                                      eye_right: val_eye_right_batch/ 255.-0.5,
                                                      face:val_face_batch/ 255.-0.5,
                                                      face_mask:val_face_mask_batch,
                                                      y:val_y_batch
                                                      })
    test_er.append(test_error/(test_size))
    train_er.append(train_error/(train_size))
    cost.append(_cost/(train_size/batch_size))
    
    
    print("step %d, training error %g"%(k, train_error/(train_size)))
    print("step %d, test error %g"%(k, test_error/(test_size)))
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)
    start = time.clock()
    #learning_rate=learning_rate*0.7
    if(test_error/(test_size)<min_err):
        min_err   =test_error/(test_size)
        save_path =saver.save(sess, path+"my-model",global_step=k)
        print('find out ranked error!')
    
    for m in range(int(train_size/batch_size)):
        # get batch of the training data
        rand_batch=np.random.randint(0, train_size/batch_size - 1)
        #rand_batch=k%int(size/batch_size)
        train_eye_left_batch =train_eye_left[int(rand_batch*batch_size):int((rand_batch+1)*batch_size)];
        train_eye_right_batch=train_eye_right[int(rand_batch*batch_size):int((rand_batch+1)*batch_size)];
        train_face_batch     =train_face[int(rand_batch*batch_size):int((rand_batch+1)*batch_size)];
        train_face_mask_batch=train_face_mask[int(rand_batch*batch_size):int((rand_batch+1)*batch_size)];
        train_y_batch        =train_y[int(rand_batch*batch_size):int((rand_batch+1)*batch_size)];
        train_step.run(session=sess, 
                           feed_dict={
                               eye_left: train_eye_left_batch/ 255.-0.5, 
                               eye_right: train_eye_right_batch/ 255.-0.5,
                               face:train_face_batch/ 255.-0.5,
                               face_mask:train_face_mask_batch,
                               y:train_y_batch
                               })
#add summary at last
summary_train= sess.run(merged,
                        feed_dict={
                                    eye_left: train_eye_left_batch/ 255.-0.5, 
                                    eye_right: train_eye_right_batch/ 255.-0.5,
                                    face:train_face_batch/ 255.-0.5,
                                    face_mask:train_face_mask_batch,
                                    y:train_y_batch
                                    })
train_writer.add_summary(summary_train, k)
sess.close()
# %%
#print loss and final train error and test error
print(cost)
print(train_error/(train_size))
print(test_error/(test_size))
filehandler = open(path+"plot_parameters.txt","wb")
pickle.dump([cost,test_er,train_er], filehandler, protocol=2)
filehandler.close()
# %%
#plot the trainning error and test error
fig,ax1 = plt.subplots()
c_l     =ax1.plot(range(0,k+1),cost, 'r-', label='training loss')
ax1.set_xlabel('epoch')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('loss', color='b')
ax1.tick_params('y', colors='b')
ax2 =ax1.twinx()
tr_l=ax2.plot(range(0,k+1), train_er, 'g-', label='training error')
te_l=ax2.plot(range(0,k+1), test_er, 'b-', label='test error')
ax2.set_ylabel('error/cm', color='orange')
ax2.tick_params('y', colors='orange')
lns =c_l+tr_l+te_l
labs=[l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
fig.tight_layout()
plt.show()