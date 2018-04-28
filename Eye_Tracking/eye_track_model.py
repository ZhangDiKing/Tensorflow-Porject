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

def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
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
def nn_layer(input_tensor, 
            input_dim, 
            output_dim, 
            layer_name, 
            act=tf.nn.relu, 
            reuse=False):
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    #print([input_dim, output_dim])
    with tf.variable_scope(layer_name+'weights',reuse=tf.AUTO_REUSE):
        
        weights = tf.get_variable(
                        layer_name+'_weight', 
                        shape=[input_dim, output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=0.1)
                        )
        biases = tf.get_variable(
                        layer_name+'_biases', 
                        shape=[output_dim],
                        initializer=tf.constant_initializer(0.1)
                        )
        
    variable_summaries(weights)
    variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

def cnn_layer(input_tensor, 
            filter_shape,
            input_dim, 
            output_dim, 
            layer_name, 
            act=tf.nn.relu,
            reuse=False):
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    #print([filter_shape[0],filter_shape[1],input_dim, output_dim])
    with tf.variable_scope(layer_name+'weights', reuse=tf.AUTO_REUSE):
            
            weights = tf.get_variable(
                            layer_name+'_weight', 
                            shape=[filter_shape[0],filter_shape[1],input_dim, output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.1)
                            )
            biases = tf.get_variable(
                            layer_name+'_biases', 
                            shape=[output_dim],
                            initializer=tf.constant_initializer(0.1)
                            )

    variable_summaries(weights)
    variable_summaries(biases)
    with tf.name_scope('convolution_w_input_plus_b'):
        preactivate = conv2d(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

def img_cnn(input, 
            in_channels,
            tensor_name, 
            cnn_d,
            filter_size, 
            fc_d):
    # define the parameters in convolution layers of left eye
    # convolution layer 1 for left eye 
    if tensor_name=='face':
        reuse=False 
    else:
        reuse=True
    for i in range(len(cnn_d)):
        with tf.name_scope('conv'+str(i)+tensor_name):
            if i > 0:
                in_channels=cnn_d[i-1]
            input=cnn_layer(input,
                            filter_size[i], 
                            in_channels, 
                            cnn_d[i],
                            'conv'+str(i)+'_'+tensor_name,
                            reuse=reuse)
        if i<len(cnn_d)-1:
            with tf.name_scope('conv'+str(i)+'_pooling'+tensor_name):
                input=max_pool(input)
        if i<2:
            with tf.name_scope('conv'+str(i)+'_LRN'+tensor_name):
                input=tf.nn.local_response_normalization(input,
                                                    depth_radius=5,
                                                    bias=1,
                                                    alpha=0.0001,
                                                    beta=0.75)
    
    with tf.name_scope('conv3_drop_out'+tensor_name):
        dim=input.get_shape().as_list()
        input = tf.reshape(input, 
                        [tf.shape(input)[0], dim[1]*dim[2]*dim[3]])
        #input = tf.nn.dropout(input, 0.5)
    #print(input.get_shape().as_list())

    for i in range(len(fc_d)):
    # fully connected layer for image
        if i == 0:
            ch_in = input.get_shape().as_list()[1]
        else:
            ch_in = fc_d[i-1]
        input = nn_layer(input, 
                        ch_in, 
                        fc_d[i], 
                        'fc'+str(i)+'_'+tensor_name,
                        reuse=reuse)
        #input = tf.nn.dropout(input, 0.5)
    return input

def mask_nn(input,fc_d):
    with tf.name_scope('face_mask_flat'):
        dim=input.get_shape().as_list()
        input = tf.reshape(input, 
                        [tf.shape(input)[0], dim[1]*dim[2]])
        #input = tf.nn.dropout(input, 0.5)

    for i in range(len(fc_d)):
    # fully connected layer 
        if i == 0:
            ch_in = input.get_shape().as_list()[1]
        else:
            ch_in = fc_d[i-1]
        input=nn_layer(input, ch_in, fc_d[i], 'fc'+str(i)+'_'+'mask')
        #input=tf.nn.dropout(input, 0.5)
    return input
    
class eye_track_model:
    def __init__(self, 
                input_tensors, 
                labels, 
                in_channel, 
                fc_d=[[]], 
                mask_fc_d=[],
                cat_fc_d=[],
                cnn_d=[],
                filter_size=[]):
        
        
        #define the parameters in convolution layers of eyes and faces
        h_fc1_left_eye = img_cnn(input_tensors[0], 
                                in_channel,
                                'eye', 
                                cnn_d[0],
                                filter_size[0], 
                                fc_d[0])

        #define the parameters in convolution layers of right eye
        h_fc1_right_eye = img_cnn(input_tensors[1], 
                                in_channel,
                                'eye', 
                                cnn_d[0],
                                filter_size[0], 
                                fc_d[0])  

        #define the parameters in convolution layers of face
        h_fc1_face = img_cnn(input_tensors[2], 
                            in_channel,
                            'face', 
                            cnn_d[1],
                            filter_size[1], 
                            fc_d[1])
        
        #the mask nn layer
        h_face_mask = mask_nn(input_tensors[3],mask_fc_d)

        #cat eyes together
        with tf.name_scope('eyes_sum'):
            h_eye_flat = tf.concat([h_fc1_left_eye, h_fc1_right_eye],1)
            #print('shape is',h_eye_flat.get_shape())
            ch_in = h_eye_flat.get_shape().as_list()[1]
            h_fc1_eye = nn_layer(h_eye_flat, ch_in, cat_fc_d[0], 'fc1_eye')
            #h_fc1_eye = tf.nn.dropout(h_fc1_eye, 0.5)
        
        # fully connected layer 1 for eyes,face,face mask
        with tf.name_scope('eyes_face_sum'):
            h_flat = tf.concat([tf.concat([h_fc1_eye, h_fc1_face],1),h_face_mask],1)

            ch_in=h_flat.get_shape().as_list()[1]
            h_fc1_face_eye_mask = nn_layer(h_flat, ch_in, cat_fc_d[1], 'fc1_eye_face_mask')
            #h_fc1_face_eye_mask = tf.nn.dropout(h_fc1_face_eye_mask, 0.5)        
        
        with tf.name_scope('final_out'):
            W_out = weight_variable([cat_fc_d[1], 2])
            B_out = bias_variable([2])
            self.predict_op = tf.matmul(h_fc1_face_eye_mask, W_out) + B_out
        
        #loss and training operation
        with tf.name_scope('loss_and_error'):
            self.loss = tf.reduce_mean(tf.pow(self.predict_op - labels, 2))/2.0
            self.error = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(self.predict_op-labels, 2),reduction_indices=1)))
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('err', self.error)
        
    def get_param(self):
        return self.loss, self.error, self.predict_op
        
        

