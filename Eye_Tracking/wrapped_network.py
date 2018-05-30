import tensorflow as tf

def weight_variable(shape, std = 0.1):
    initial = tf.truncated_normal(shape, stddev = std)
    return tf.Variable(initial)

def bias_variable(shape, init_val = 0.1):
    initial = tf.constant(init_val, shape = shape)
    return tf.Variable(initial)

#definition of convolution and pooling
def conv2d(x, weights, s = 1, padding = 'VALID'):
    return tf.nn.conv2d(x, weights, strides = [1, s, s, 1], padding = padding)

def max_pool(x, k = 3, s = 2, padding = 'SAME'):
    return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, s, s, 1], padding = padding)

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
            act = tf.nn.relu, 
            reuse = False):
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    #print([input_dim, output_dim])
    with tf.variable_scope(layer_name + 'weights',reuse = tf.AUTO_REUSE):
        
        weights = tf.get_variable(
                        layer_name + '_weight', 
                        shape = [input_dim, output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=0.1)
                        )
        biases = tf.get_variable(
                        layer_name + '_biases', 
                        shape = [output_dim],
                        initializer = tf.constant_initializer(0.1)
                        )
        
    variable_summaries(weights)
    variable_summaries(biases)

    with tf.name_scope('preactive'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

def cnn_layer(input_tensor,
            in_channels,
            index,
            hyper_para, 
            layer_name,
            act = tf.nn.relu,
            reuse = False):
    '''
    The cnn layer for the face and eyes 
    '''
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):

        with tf.variable_scope(layer_name+'weights', reuse=tf.AUTO_REUSE):
                shape = [hyper_para["filter_size"][index][0], 
                        hyper_para["filter_size"][index][1], 
                        in_channels,  
                        hyper_para["cnn_d"][index]
                        ]
                #normally the initialization for the weights would be sqrt(2/n)
                initializer = tf.truncated_normal_initializer(stddev = 0.1)

                weights = tf.get_variable(
                                layer_name + '_weight', 
                                shape,
                                initializer = initializer
                                )
                biases = tf.get_variable(
                                layer_name + '_biases', 
                                shape = [hyper_para["cnn_d"][index]],
                                initializer = tf.constant_initializer(0.1)
                                )

        variable_summaries(weights)
        variable_summaries(biases)

        with tf.name_scope('preactive'):
            preact = conv2d(input_tensor, 
                            weights, 
                            hyper_para["cnn_strides"][index],
                            hyper_para["cnn_padding"][index]) + biases
            tf.summary.histogram('pre_activations', preact)

        activations = act(preact, name = 'activation')
        tf.summary.histogram('activations', activations)
        return activations
