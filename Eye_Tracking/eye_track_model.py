import tensorflow as tf
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

def img_cnn(input_tensor,in_channels,hc_d,tensor_name):
    #define the parameters in convolution layers of left eye
    # convolution layer 1 for left eye  
    h_conv1=cnn_layer(input_tensor,[5,5],in_channels, 32,'conv1_'+tensor_name)#60,60,32
    with tf.name_scope('conv1_pooling'+tensor_name):
        h_pool1=max_pool_2x2(h_conv1) #30,30,32
    
    # convolution layer 2 for left eye  
    h_conv2=cnn_layer(h_pool1,[7,7],32, 32,'conv2_'+tensor_name)#24,24,32
    with tf.name_scope('conv2_pooling'+tensor_name):
        h_pool2=max_pool_2x2(h_conv2) #12,12,32
    
    # convolution layer 3 for left eye  
    h_conv3=cnn_layer(h_pool2,[5,5],32, 32,'conv3_'+tensor_name)#8,8,32
        
    with tf.name_scope('conv3_drop_out'+tensor_name):
        h_conv3_flat=tf.reshape(h_conv3, [tf.shape(h_conv3)[0], 8*8*32])
        h_drop      =tf.nn.dropout(h_conv3_flat, 0.5)
    
    # fully connected layer 1 for left eye
    h_fc1=nn_layer(h_drop, 8*8*32, hc_d, 'fc1_'+tensor_name)
    return h_fc1

def mask_nn(face_mask,hc_d):
    # fully connected layer 1 for face mask
    with tf.name_scope('face_mask_drop_out'):
        h_face_mask_flat=tf.reshape(tf.cast(face_mask,tf.float32), [tf.shape(face_mask)[0], 25*25])
        h_face_mask_drop=tf.nn.dropout(h_face_mask_flat, 0.5)
    # fully connected layer 1 for right eye
    h_fc1_face_mask=nn_layer(h_face_mask_drop, 25*25, hc_d, 'fc1_face_mask')
    
    # fully connected layer 2 for face mask
    h_fc2_face_mask=nn_layer(h_fc1_face_mask, hc_d, hc_d, 'fc2_face_mask')
    return h_fc2_face_mask
    
class eye_track_model:
    def __init__(self, batch_size, input_tensors, labels, in_channel,hc_d=128):
        
        #define the parameters in convolution layers of left eye
        h_fc1_left_eye=img_cnn(input_tensors[0],in_channel,hc_d,'eye_left')
        #define the parameters in convolution layers of right eye
        h_fc1_right_eye=img_cnn(input_tensors[1],in_channel,hc_d,'eye_right')  
        # %%
        #define the parameters in convolution layers of face
        h_fc1_face=img_cnn(input_tensors[2],in_channel,hc_d,'face')
        
        h_face_mask=mask_nn(input_tensors[3],hc_d)
        h_eye_flat=tf.concat([h_fc1_left_eye, h_fc1_right_eye],1)
        h_fc1_eye =nn_layer(h_eye_flat, 2*hc_d, hc_d, 'fc1_eye')
        
        # fully connected layer 1 for eyes,face,face mask
        #due to different version of tensorflow, this sentense may be changed
        h_flat             =tf.concat([tf.concat([h_fc1_eye, h_fc1_face],1),h_face_mask],1)
        #h_flat             =tf.concat(1,[tf.concat(1,[h_fc1_eye, h_fc1_face]),h_fc2_face_mask])
        h_fc1_face_eye_mask=nn_layer(h_flat, 3*hc_d, hc_d, 'fc1_eye_face_mask')
        
        # %%
        
        with tf.name_scope('final_out'):
            W_out     =weight_variable([hc_d, 2])
            B_out     =bias_variable([2])
            #h_eye_drop = tf.nn.dropout(h_face_mask_flat, 0.5)
            self.predict_op=tf.matmul(h_fc1_face_eye_mask, W_out) + B_out
        
        # %%
        #loss and training operation
        with tf.name_scope('loss_and_error'):
            self.loss =tf.reduce_mean(tf.pow(self.predict_op-labels, 2))/2
            self.error=tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(self.predict_op-labels, 2),reduction_indices=1)))
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('err', self.error)
        
    def get_param(self):
        return self.loss,self.error,self.predict_op
        
        

