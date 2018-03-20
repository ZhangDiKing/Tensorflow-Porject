import tensorflow as tf
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import pickle
import time

train_size    = 50000
test_size     = 5000
batch_size    = 100
dim           = 28*28
class_no      = 10
dim_h         = 100
learning_rate = 0.008

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def mat_to_tensor_w(mat, m):
    tensor_return = tf.tile(mat, [m, 1, 1], name = None)
    return tensor_return

def forward_propgation(w1, w10, w2, w20, w3, w30, x):
    #transfer mat to tensor
    w1_mat  = tf.reshape(w1,  [dim,dim_h])
    w10_mat = tf.reshape(w10, [dim_h,1])
    w2_mat  = tf.reshape(w2,  [dim_h,dim_h])
    w20_mat = tf.reshape(w20, [dim_h,1])
    w3_mat  = tf.reshape(w3,  [dim_h,class_no])
    w30_mat = tf.reshape(w30, [class_no,1])
    
    #compute hidden layer 1
    h1 = tf.nn.relu(w10_mat + tf.matmul(tf.transpose(w1_mat), x))
    
    #compute hidden layer 2
    h2 = tf.nn.relu(w20_mat + tf.matmul(tf.transpose(w2_mat), h1))
    
    #compute softmax of the output layer
    f_k_x         = tf.exp(tf.matmul(tf.transpose(w3_mat), h2) + w30_mat)
    sigmoid_f_k_x = f_k_x / tf.reduce_sum(f_k_x, 0, keep_dims = True)
    
    return h1, h2, sigmoid_f_k_x

def update_weight(w, g_w, lamda, learning_rate):
    return tf.assign_sub(w,  learning_rate * (tf.reduce_sum(g_w,  0, keep_dims=True)) + lamda *2 * w)

def compute_error(w1, w10, w2, w20, w3, w30, dataY, dataX):
    #transfer tensor to mat
    w1_mat  = tf.reshape(w1,  [dim, dim_h])
    w10_mat = tf.reshape(w10, [dim_h, 1])
    w2_mat  = tf.reshape(w2,  [dim_h, dim_h])
    w20_mat = tf.reshape(w20, [dim_h, 1])
    w3_mat  = tf.reshape(w3,  [dim_h, class_no])
    w30_mat = tf.reshape(w30, [class_no, 1])
    
    #compute hidden layer 1
    h1 = tf.nn.relu(w10_mat + tf.matmul(tf.transpose(w1_mat), dataX))
    
    #compute hidden layer 2
    h2 = tf.nn.relu(w20_mat + tf.matmul(tf.transpose(w2_mat), h1))
    
    #compute softmax of the output layer
    f_k_x         = tf.exp(tf.matmul(tf.transpose(w3_mat), h2) + w30_mat)
    sigmoid_f_k_x = f_k_x / tf.reduce_sum(f_k_x, 0, keep_dims = True)
    
    #compute accuracy and loss of the nerual network
    correct_list = tf.equal(tf.argmax(sigmoid_f_k_x, axis = 0), tf.argmax(dataY, axis = 0))
    accuracy     = tf.reduce_mean(tf.cast(correct_list, 'float'))
    loss         = tf.reduce_sum((dataY - sigmoid_f_k_x) * (dataY - sigmoid_f_k_x))
    
    return accuracy,loss
def main():
    #use panda to read the file
    path    = 'C:/Users/zhang/dp/dp3/'
    train_X = pd.read_csv(path + 'train_data.txt', sep = " ", header = None)
    train_Y = pd.read_csv(path + 'train_label.txt', sep = " ", header = None)
    test_X  = pd.read_csv(path + 'test_data.txt', sep=" ", header = None)
    test_Y  = pd.read_csv(path + 'test_label.txt', sep=" ", header = None)
    
    
    #tensorflow allow float32 datatype
    train_X = numpy.float32(train_X.values)
    train_Y = numpy.float32(train_Y.values)
    test_X  = numpy.float32(test_X.values)
    test_Y  = numpy.float32(test_Y.values)
    
    
    #initial the parameter 
    w1  = weight_variable([1, dim, dim_h])
    w2  = weight_variable([1, dim_h, dim_h])
    w3  = weight_variable([1, dim_h, class_no])
    w10 = bias_variable([1, dim_h, 1])
    w20 = bias_variable([1, dim_h, 1])
    w30 = bias_variable([1, class_no, 1])
    
    #define the input x and label y
    X = tf.placeholder('float32', [dim, None])
    Y = tf.placeholder('float32')
    
    #transfer mat of the input x and label y to tensor
    Y_tensor_batch = tf.reshape(tf.transpose(Y), [batch_size, class_no, 1])
    X_tensor_batch = tf.reshape(tf.transpose(X), [batch_size, dim, 1])
    
    #forward propgation
    h1_mat, h2_mat, sigmoid_f_k_x_mat = forward_propgation(w1, w10, w2, w20, w3, w30, X)
    
    #transfer mat of the h1 and h2 to tensor
    h1            = tf.reshape(tf.transpose(h1_mat), [batch_size, dim_h, 1])
    h2            = tf.reshape(tf.transpose(h2_mat), [batch_size, dim_h, 1])
    sigmoid_f_k_x = tf.reshape(tf.transpose(sigmoid_f_k_x_mat), [batch_size, class_no, 1])
    
    
    #backward propgation#
    #BP of output layer
    g_y       = sigmoid_f_k_x - Y_tensor_batch
    w3_tensor = mat_to_tensor_w(w3, batch_size)
    y_pre     = sigmoid_f_k_x
    y_pre_t   = tf.transpose(y_pre, perm=[0, 2, 1])
    g_y_t     = tf.transpose(g_y, perm = [0, 2, 1])
    g_w30_t   = y_pre_t * g_y_t - tf.reduce_sum(tf.matmul(y_pre * g_y, y_pre_t), 1, keep_dims = True)
    g_w30     = tf.transpose(g_w30_t, perm=[0, 2, 1])
    g_w3      = tf.matmul(h2, g_w30_t)
    g_h2      = tf.matmul(w3_tensor - tf.matmul(w3_tensor, y_pre), y_pre * g_y)
    
    #BP of hidden layer
    g_w20     = tf.cast(tf.greater(h2, 0), dtype = tf.float32) * g_h2
    g_w2      = tf.matmul(h1, tf.transpose(g_w20, perm = [0, 2, 1]))
    w2_tensor = mat_to_tensor_w(w2, batch_size)
    g_h1      = tf.matmul(w2_tensor, g_w20)
    
    #BP of input layer
    g_w10     = tf.cast(tf.greater(h1,0), dtype = tf.float32) * g_h1
    g_w1      = tf.matmul(X_tensor_batch, tf.transpose(g_w10, perm=[0, 2, 1]))
    
    #compute the accuracy and loss of trainning data#    
    accuracy_train, loss = compute_error(w1, w10, w2, w20, w3, w30, train_Y, train_X)
    accuracy_test, _     = compute_error(w1, w10, w2, w20, w3, w30, test_Y, test_X)
    
    #update the weight
    lamda=0.01
    w1  = update_weight(w1,  g_w1,  lamda, learning_rate)
    w10 = update_weight(w10, g_w10, lamda, learning_rate)
    w2  = update_weight(w2,  g_w2,  lamda, learning_rate)
    w20 = update_weight(w20, g_w20, lamda, learning_rate)
    w3  = update_weight(w3,  g_w3,  lamda, learning_rate)
    w30 = update_weight(w30, g_w30, lamda, learning_rate)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    result_test_max = 0
    loss_list       = []
    er_train        = []
    er_test         = []
    epoch           = 40
    
    start = time.clock()
    for k in range(0,epoch):
        #compute the loss and error every epoch
        _accuracy_test         = sess.run(accuracy_test)
        _accuracy_train, _loss = sess.run([accuracy_train, loss])
                                                                   
        loss_list.append(_loss)
        er_train.append(1 - _accuracy_train)
        er_test.append(1 - _accuracy_test)
        
        #print the result
        print('epoch=', k)
        print('test error=',  1 - _accuracy_test)
        print('train error=', 1 - _accuracy_train)
        print('loss=', _loss)
        print("Time used:", time.clock() - start)
        start = time.clock()
    
        if(epoch >= 10):
            if(_accuracy_test > result_test_max):
                result_test_max = _accuracy_test
                print('get best')
                print('test error=', 1 - _accuracy_test)
                print('train error=', 1 - _accuracy_train)
                
        #decay of the learning rate
        if(k >= 3):
            learning_rate = 0.01 * (0.7 ** int((k - 100) / 200))
        elif(k >= 10):
            learning_rate = 0.01 * (0.6 ** int((k - 100) / 200))
        else:
            learning_rate = 0.01
            
        #training through the whole trainning data
        for i in range(train_size / batch_size):
            rand_batch  = numpy.random.randint(0, train_size / batch_size - 1)
            #rand_batch  = k%int(size / batch_size)
            dataX_batch = train_X[:, int(rand_batch * batch_size):int((rand_batch + 1) * batch_size)];
            dataY_batch = train_Y[:, int(rand_batch * batch_size):int((rand_batch + 1) * batch_size)];
            _w1, _w10, _w2, _w20, _w3, _w30 = sess.run([w1, w10, w2, w20, w3, w30], 
                                                       feed_dict = {
                                                                    X: dataX_batch,
                                                                    Y: dataY_batch })
    
    sess.close()
    
    # plots of the loss, average trainning classification error & average test classification error
    fig, ax1 = plt.subplots()
    c_l      = ax1.plot(range(0, k + 1), cost, 'r-', label = 'training loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color = 'b')
    ax1.tick_params('y', colors = 'b')
    
    ax2 =ax1.twinx()    
    tr_l = ax2.plot(range(0, k + 1), er_train, 'g-', label = 'training error')
    te_l = ax2.plot(range(0, k + 1), er_test, 'b-', label = 'test error')
    lns  = c_l+tr_l+te_l
    labs = [l.get_label() for l in lns]
    ax2.set_ylabel('error/cm', color = 'orange')
    ax2.tick_params('y', colors = 'orange')
    
    ax1.legend(lns, labs, loc = 0)
    fig.tight_layout()
    plt.show()
