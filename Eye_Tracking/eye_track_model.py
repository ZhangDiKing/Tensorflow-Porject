import tensorflow as tf
import pickle
import time
import numpy as np
import os

from utils import hyper_setting, get_batch, read_data, draw_plot
from wrapped_network import weight_variable, bias_variable, variable_summaries, nn_layer, cnn_layer, max_pool

class eye_track_model:
    def __init__(self, default_setting = True):
        self.default_setting = default_setting

    def model_fn(self, input_tensors, labels, hyper_para):
        
        in_channel = hyper_para["in_channels"]

        #define the parameters in convolution layers of eyes and faces
        h_fc1_left_eye = self.img_cnn(input_tensors[0], 
                                        in_channel,
                                        'eye', 
                                        hyper_para)

        #define the parameters in convolution layers of right eye
        h_fc1_right_eye = self.img_cnn(input_tensors[1], 
                                        in_channel,
                                        'eye', 
                                        hyper_para)  

        #define the parameters in convolution layers of face
        h_fc1_face = self.img_cnn(input_tensors[2], 
                                    in_channel,
                                    'face', 
                                    hyper_para)
        
        #the mask nn layer
        h_face_mask = self.mask_nn(input_tensors[3], hyper_para["mask_fc_d"])

        #cat eyes together
        with tf.name_scope('eyes_cat'):
            h_eye_flat = tf.concat([h_fc1_left_eye, h_fc1_right_eye], 1)
            #print('shape is',h_eye_flat.get_shape())
            ch_in = h_eye_flat.get_shape().as_list()[1]
            h_fc1_eye = nn_layer(h_eye_flat, ch_in, hyper_para["cat_fc_d"][0], 'fc1_eye')
            #h_fc1_eye = tf.nn.dropout(h_fc1_eye, 0.5)
        
        # fully connected layer 1 for eyes, face, face mask
        with tf.name_scope('eyes_face_cat'):
            h_flat = tf.concat([tf.concat([h_fc1_eye, h_fc1_face],1),h_face_mask],1)

            ch_in = h_flat.get_shape().as_list()[1]
            h_fc1_face_eye_mask = nn_layer(h_flat, ch_in, hyper_para["cat_fc_d"][1], 'fc1_eye_face_mask')
            #h_fc1_face_eye_mask = tf.nn.dropout(h_fc1_face_eye_mask, 0.5)        
        
        with tf.name_scope('final_out'):
            W_out = weight_variable([hyper_para["cat_fc_d"][1], 2])
            B_out = bias_variable([2])
            self.predict_op = tf.matmul(h_fc1_face_eye_mask, W_out) + B_out
        
        #loss and training operation
        with tf.name_scope('loss_and_error'):
            self.loss = tf.reduce_mean(tf.pow(self.predict_op - labels, 2)) / 2.0
            self.error = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(self.predict_op - labels, 2), reduction_indices = 1)))
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('err', self.error)

        with tf.name_scope('train_op'):
            self.train_step = tf.train.RMSPropOptimizer(hyper_para["learning_rate"], decay = hyper_para["decay"]).minimize(self.loss)

    def img_cnn(self, input, in_channels, tensor_name, hyper_para):
        # define the parameters in convolution layers of left eye
        # convolution layer 1 for left eye 
        if tensor_name == 'face':
            reuse = False 
            fc_d = hyper_para["fc_d"][1]
        else:
            reuse = True
            fc_d = hyper_para["fc_d"][0]
        for i in range(len(hyper_para["cnn_d"])):
            with tf.name_scope('conv' + str(i) + '_' + tensor_name):
                if i > 0:
                    in_channels = hyper_para["cnn_d"][i-1]
                input = cnn_layer(input, 
                                in_channels,
                                i, 
                                hyper_para, 
                                'conv' + str(i) + '_' + tensor_name, 
                                act = tf.nn.relu, 
                                reuse = reuse)
            
            if i < 2:
                #following the setting of the orignal paper
                with tf.name_scope('conv'+str(i)+'_LRN_'+tensor_name):
                    input = tf.nn.local_response_normalization(input,
                                                                depth_radius = 5,
                                                                bias = 1,
                                                                alpha = 0.0001,
                                                                beta = 0.75)
            if i < len(hyper_para["cnn_d"]) - 2:
                with tf.name_scope('conv' + str(i) + '_pooling_' + tensor_name):
                    input = max_pool(input, 
                                    hyper_para["pooling_k_size"][i], 
                                    hyper_para["pooling_strides"][i],
                                    hyper_para["pooling_padding"][i])
        
        with tf.name_scope('conv3_flat_' + tensor_name):
            dim = input.get_shape().as_list()
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
                            'fc' + str(i) + '_' + tensor_name,
                            reuse = reuse)
            #input = tf.nn.dropout(input, 0.5)
        return input

    def mask_nn(self, input, fc_d):
        with tf.name_scope('face_mask_flat'):
            dim = input.get_shape().as_list()
            input = tf.reshape(input, 
                            [tf.shape(input)[0], dim[1] * dim[2]])
            #input = tf.nn.dropout(input, 0.5)

        for i in range(len(fc_d)):
        # fully connected layer 
            if i == 0:
                ch_in = input.get_shape().as_list()[1]
            else:
                ch_in = fc_d[i-1]
            input = nn_layer(input, ch_in, fc_d[i], 'fc' + str(i) + '_' + 'mask')
            #input=tf.nn.dropout(input, 0.5)
        return input
        
    def get_param(self):
        return self.loss, self.error, self.predict_op, self.train_step
    
    def fit(self, 
            train_data, 
            val_data, 
            path = ".",
            batch_size = 250,
            epoch = 80):
        start = time.time()
        print("start to build model")

        ###########################################################
        #########<configuration of neural network here>############
        hyper_para = hyper_setting(self.default_setting)
        min_err = 2.5
        train_size = train_data[0].shape[0]
        test_size = val_data[0].shape[0]
        ##########################################################

        #define input
        _, w, h, _ = train_data[0].shape
        with tf.name_scope('eye_left_input'):
            eye_left = tf.placeholder('float32', [None, w, h, 3])

        _, w, h, _ = train_data[1].shape
        with tf.name_scope('eye_right_input'):
            eye_right = tf.placeholder('float32', [None, w, h, 3])

        _, w, h, _ = train_data[2].shape
        with tf.name_scope('face_input'):  
            face = tf.placeholder('float32', [None, w, h, 3])

        _, w, h = train_data[3].shape
        with tf.name_scope('face_mask_input'):    
            face_mask = tf.placeholder('float32', [None, w, h])

        with tf.name_scope('label_input'):    
            y = tf.placeholder('float32', [None, 2])

        start = time.time()
        self.model_fn([eye_left, eye_right, face, face_mask], y, hyper_para)
        loss, error, predict_op, train_step = self.get_param()
        
        print("Model construction complete in %s seconds." % (round(time.time() - start,4)))

        #save models
        with tf.name_scope('save_model'):
            tf.get_collection("validation_nodes")
            tf.add_to_collection("validation_nodes", eye_left)
            tf.add_to_collection("validation_nodes", eye_right)
            tf.add_to_collection("validation_nodes", face)
            tf.add_to_collection("validation_nodes", face_mask)
            tf.add_to_collection("validation_nodes", predict_op)
            saver = tf.train.Saver(max_to_keep = 1)
        
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        sess = tf.Session()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(path + '/train', sess.graph)
    
        sess.run(tf.global_variables_initializer())
        
        start = time.time()
        ######################################################
        train_er = []
        test_er = []
        cost = []
        error_step = []
        test_step = []
        g_step = 0
        ######################################################
        
        print('start training...')
        for k in range(0, epoch):
            start_time = time.time()
            #compute the trainning accuracy and cost
            train_error = 0
            _cost = 0
            
            #compute the test accuracy
            test_error = 0
            i = 0
            
            start = time.time()
            for i in range(int(test_size / batch_size)):
                batch_data = get_batch(val_data, batch_size, i)
                test_error = test_error+error.eval(session = sess, 
                                                    feed_dict = {
                                                            eye_left: batch_data[0] / 255.-0.5, 
                                                            eye_right: batch_data[1] / 255.-0.5,
                                                            face: batch_data[2] / 255.-0.5,
                                                            face_mask: batch_data[3],
                                                            y: batch_data[4]
                                                            })
                
            test_er.append(test_error / test_size)
            test_step.append(g_step)
            
            print("epoch %s, test error %s"%(k, round(test_error / test_size, 5)))
            
            #learning_rate=learning_rate*0.7
            if(test_error / test_size < min_err):
                min_err = test_error / test_size
                if path == ".":
                    save_path = os.getcwd()
                save_path = saver.save(sess, save_path + "my-model", global_step = k)
                print('find out ranked error! Min error is %s', (round(min_err, 4)))
            
            start = time.time()
            
            for m in range(int(train_size / batch_size)):
                batch_data = get_batch(train_data, batch_size, m)
                
                #compute batch error every 100 steps
                if g_step % 100 == 0:
                    _cross_entropy,_error = sess.run([loss,error],
                                                    feed_dict={
                                                    eye_left: batch_data[0] / 255.-0.5, 
                                                    eye_right: batch_data[1] / 255.-0.5,
                                                    face: batch_data[2] / 255.-0.5,
                                                    face_mask: batch_data[3],
                                                    y: batch_data[4]
                                                    })
                    train_error = _error
                    _cost = _cross_entropy
                    print("epoch %d, step %d training error for one batch %g"%(k, m, train_error / batch_size))
                    print("epoch %d, step %d cost %g"%(k, m, _cost / batch_size))
                    train_er.append(train_error / batch_size)
                    cost.append(_cost / batch_size)
                    error_step.append(g_step)
                
                train_step.run(session = sess, 
                                feed_dict={
                                        eye_left: batch_data[0] / 255.-0.5, 
                                        eye_right: batch_data[1] / 255.-0.5,
                                        face: batch_data[2] / 255.-0.5,
                                        face_mask: batch_data[3],
                                        y: batch_data[4]
                                        })
                g_step+=1
        
            elapsed = (time.time() - start_time)
            print("total Time for one epoch used is %d s."% (round(elapsed, 4)))
        
        #add summary at last
        summary_train= sess.run(merged,
                                feed_dict = {
                                            eye_left: batch_data[0] / 255.-0.5, 
                                            eye_right: batch_data[1] / 255.-0.5,
                                            face: batch_data[2] / 255.-0.5,
                                            face_mask: batch_data[3],
                                            y: batch_data[4]
                                            })
        train_writer.add_summary(summary_train, k)
        sess.close()
        print("training completed")
        print("the min error we have is %s" %(round(min_err, 4)))
        
        filehandler = open(path + "plot_parameters.txt", "wb")
        pickle.dump([cost, test_er, train_er, error_step, test_step], filehandler, protocol = 2)
        filehandler.close()

        draw_plot(cost, test_er, train_er, error_step, test_step)
        
        

