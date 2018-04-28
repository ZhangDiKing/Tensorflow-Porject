# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:34:27 2018

@author: zhang
"""
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pickle

import argparse
from eye_track_model import eye_track_model 

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='C:/Users/zhang/dp/dP_final/',type=str)

def get_batch(data, batch_size, i):
    out=[]
    for k in range(5):
        out.append(data[k][int(i*batch_size):int((i+1)*batch_size)])
    return out

def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    
    in_channels=3
    batch_size=250
    epoch=70
    decay=0.5
    path=opt.data_path
    print("start preprocesing data...")
    start=time.time()
    npzfile = np.load(path+"train_and_val.npz")

    if not npzfile:
        print("Not data found")

    train_data=[
        npzfile["train_eye_left"],
        npzfile["train_eye_right"],
        npzfile["train_face"],
        npzfile["train_face_mask"],
        npzfile["train_y"]
    ]
    val_data=[
        npzfile["val_eye_left"],
        npzfile["val_eye_right"],
        npzfile["val_face"],
        npzfile["val_face_mask"],
        npzfile["val_y"]
    ]
    print("load dataset in %s seconds." % (round(time.time() - start,4)))
    
    start=time.time()
    print("start to build model")
    #define input
    with tf.name_scope('eye_left_input'):
        eye_left=tf.placeholder('float32',[None,64,64,3])
    with tf.name_scope('eye_right_input'):
        eye_right=tf.placeholder('float32',[None,64,64,3])
    with tf.name_scope('face_input'):    
        face=tf.placeholder('float32',[None,64,64,3])
    with tf.name_scope('face_mask_input'):    
        face_mask=tf.placeholder('float32',[None,25,25])
    with tf.name_scope('label_input'):    
        y=tf.placeholder('float32',[None,2])

    #network config here, suit your own model
    fc_d = [
        #eye fc in cnn config
        [32],
        #face fc in cnn config
        [64, 32]
    ]
    cnn_d = [
        #eye cnn channels config
        [32, 32, 32, 32],
        #face cnn channels config
        [32, 32, 32, 32]
    ]
    filter_size = [
        #face filter in cnn config
        [ [5, 5], [7, 7], [5, 5], [1, 1] ],
        #face filter in cnn config
        [ [5, 5], [7, 7], [5, 5], [1, 1] ]
    ]
    mask_fc_d = [32 ,32]
    cat_fc_d = [32, 32]
    
    
    model = eye_track_model([eye_left, eye_right, face, face_mask], # four place-holders you have
                            y, #x-y coordinate you have
                            in_channels, 
                            fc_d, 
                            mask_fc_d,
                            cat_fc_d,
                            cnn_d,
                            filter_size)
    loss, error, predict_op = model.get_param()
    
    
    with tf.name_scope('train_op'):
        learning_rate=0.001
        train_step   =tf.train.RMSPropOptimizer(learning_rate, decay=decay).minimize(loss)
    #save models
    with tf.name_scope('save_model'):
        tf.get_collection("validation_nodes")
        tf.add_to_collection("validation_nodes", eye_left)
        tf.add_to_collection("validation_nodes", eye_right)
        tf.add_to_collection("validation_nodes", face)
        tf.add_to_collection("validation_nodes", face_mask)
        tf.add_to_collection("validation_nodes", predict_op)
        saver = tf.train.Saver(max_to_keep=1)
    
    print("Model construction complete in %s seconds." % (round(time.time() - start,4)))
    
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    sess        =tf.Session()
    merged      =tf.summary.merge_all()
    train_writer=tf.summary.FileWriter(path + '/train',
                                          sess.graph)
    #test_writer = tf.summary.FileWriter(path + '/test',sess.graph)
    sess.run(tf.global_variables_initializer())
    
    start = time.time()
    min_err = 2.0
    train_size = 48000
    test_size = 5000
    train_er = []
    test_er = []
    cost = []
    error_step = []
    test_step = []
    g_step=0
    print('start training...')
    for k in range(0,epoch):
        start_time=time.time()
        #compute the trainning accuracy and cost
        train_error = 0
        _cost = 0
        
        #compute the test accuracy
        test_error=0
        i         =0
        
        
        start=time.time()
        for i in range(int(test_size/batch_size)):
            [val_eye_left_batch, \
                val_eye_right_batch, \
                val_face_batch, \
                val_face_mask_batch, \
                val_y_batch] = get_batch(val_data, batch_size, i)
            with tf.device("/gpu:0"):
                test_error = test_error+error.eval(session=sess, 
                                                      feed_dict={
                                                              eye_left: val_eye_left_batch/ 255.-0.5, 
                                                              eye_right: val_eye_right_batch/ 255.-0.5,
                                                              face:val_face_batch/ 255.-0.5,
                                                              face_mask:val_face_mask_batch,
                                                              y:val_y_batch
                                                              })
            
        test_er.append(test_error/(test_size))
        test_step.append(g_step)
        
        
        
        print("epoch %s, test error %s"%(k, round(test_error/test_size,5)))
        
        #learning_rate=learning_rate*0.7
        if(test_error/test_size<min_err):
            min_err   =test_error/(test_size)
            save_path =saver.save(sess, path+"my-model", global_step=k)
            print('find out ranked error! Min error is %s',(round(min_err,4)))
        
        start=time.time()
        
        for m in range(int(train_size/batch_size)):
            [train_eye_left_batch, \
                train_eye_right_batch, \
                train_face_batch, \
                train_face_mask_batch, \
                train_y_batch] = get_batch(train_data, batch_size, m)
            
            with tf.device("/gpu:0"):
                #compute batch error every 100 steps
                if g_step%100==0:
                    _cross_entropy,_error=sess.run([loss,error],
                                           feed_dict={
                                                   eye_left: train_eye_left_batch/ 255.-0.5, 
                                                   eye_right: train_eye_right_batch/ 255.-0.5,
                                                   face:train_face_batch/ 255.-0.5,
                                                   face_mask:train_face_mask_batch,
                                                   y:train_y_batch
                                                   })
                    train_error = _error
                    _cost = _cross_entropy
                    print("epoch %d, step %d training error for one batch %g"%(k, m, train_error/batch_size))
                    print("epoch %d, step %d cost %g"%(k, m, _cost/batch_size))
                    train_er.append(train_error/train_size)
                    cost.append(_cost/train_size)
                    error_step.append(g_step)
                
            
                train_step.run(session=sess, 
                                   feed_dict={
                                       eye_left: train_eye_left_batch/ 255.-0.5, 
                                       eye_right: train_eye_right_batch/ 255.-0.5,
                                       face:train_face_batch/ 255.-0.5,
                                       face_mask:train_face_mask_batch,
                                       y:train_y_batch
                                       })
                g_step+=1
    
        elapsed = (time.time() - start_time)
        print("total Time for one epoch used is %d s."% (round(elapsed,4)))
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
    print("training completed")
    print("the min error we have is %s" %(round(min_err,4)))
    #print loss and final train error and test error
    #print(cost)
    #print(train_error/(train_size))
    #print(test_error/(test_size))
    filehandler = open(path+"plot_parameters.txt","wb")
    pickle.dump([cost,test_er,train_er,error_step,test_step], filehandler, protocol=2)
    filehandler.close()

    #plot the trainning error and test error
    fig,ax1 = plt.subplots()
    c_l     =ax1.plot(error_step,cost, 'r-', label='training loss')
    ax1.set_xlabel('step')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('loss', color='b')
    ax1.tick_params('y', colors='b')
    ax2 =ax1.twinx()
    tr_l=ax2.plot(error_step, train_er, 'g-', label='batch training error')
    te_l=ax2.plot(test_step, test_er, 'b-', label='test error')
    ax2.set_ylabel('error/cm', color='orange')
    ax2.tick_params('y', colors='orange')
    lns =c_l+tr_l+te_l
    labs=[l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    fig.tight_layout()
    plt.show()
    
    
if __name__=="__main__":
    main()

