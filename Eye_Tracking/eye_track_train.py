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

import pre_processing as pre
from eye_track_model import eye_track_model 

def main():
    in_channels=3
    batch_size=125
    hc_d=64
    epoch=40
    decay=0.5
    path='C:/Users/zhang/dp/dP_final/'
    print("start preprocesing data...")
    start=time.time()
    training_init_op,validation_init_op,next_element=pre.transfer_data(path,batch_size,epoch)
    print("transfer dataset into tf.dataset in %s seconds." % (round(time.time() - start,4)))
    
    start=time.time()
    print("start to build model")
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

    model=eye_track_model(batch_size, 
                          [eye_left, eye_right, face, face_mask],
                          y, in_channels, hc_d)
    loss,error,predict_op=model.get_param()
    
    
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
        saver = tf.train.Saver()
    
    print("Model construction complete in %s seconds." % (round(time.time() - start,4)))
    
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    sess        =tf.Session()
    merged      =tf.summary.merge_all()
    train_writer=tf.summary.FileWriter(path + '/train',
                                          sess.graph)
    #test_writer = tf.summary.FileWriter(path + '/test',sess.graph)
    sess.run(tf.global_variables_initializer())
    
    start     =time.time()
    min_err   =2.0
    train_size=48000
    test_size =5000
    train_er  =[]
    test_er   =[]
    cost      =[]
    error_step=[]
    test_step=[]
    g_step=0
    print('start training...')
    for k in range(0,epoch):
        start_time=time.time()
        #compute the trainning accuracy and cost
        train_error=0
        _cost      =0
        
        #compute the test accuracy
        test_error=0
        i         =0
        
        
        start=time.time()
        sess.run(validation_init_op)
        print('validation op initialzation finished in in %s seconds.' % (round(time.time() - start,4)))
        
        start=time.time()
        for i in range(int(test_size/batch_size)):
            with tf.device("/cpu:0"):
                [val_eye_left_batch, \
                 val_eye_right_batch, \
                 val_face_batch, \
                 val_face_mask_batch, \
                 val_y_batch]        =sess.run(next_element)
            with tf.device("/gpu:0"):
                test_error           =test_error+error.eval(session=sess, 
                                                      feed_dict={
                                                              eye_left: val_eye_left_batch/ 255.-0.5, 
                                                              eye_right: val_eye_right_batch/ 255.-0.5,
                                                              face:val_face_batch/ 255.-0.5,
                                                              face_mask:val_face_mask_batch,
                                                              y:val_y_batch
                                                              })
            
        test_er.append(test_error/(test_size))
        test_step.append(g_step)
        
        
        
        print("epoch %d, test error %g"%(k, test_error/(test_size)))
        
        #learning_rate=learning_rate*0.7
        if(test_error/test_size<min_err):
            min_err   =test_error/(test_size)
            save_path =saver.save(sess, path+"my-model")
            print('find out ranked error! Min error is ',min_err)
        
        start=time.time()
        sess.run(training_init_op)
        print('training op initialzation finished in in %s seconds.' % (round(time.time() - start,4)))
        
        for m in range(int(train_size/batch_size)):
            with tf.device("/cpu:0"):
                [train_eye_left_batch, \
                 train_eye_right_batch, \
                 train_face_batch, \
                 train_face_mask_batch, \
                 train_y_batch]=sess.run(next_element)
            
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
                    train_error=_error
                    _cost      =_cross_entropy
                    print("epoch %d, step %d training error for one batch %g"%(k, m, train_error/batch_size))
                    print("epoch %d, step %d cost %g"%(k, m, _cost/batch_size))
                    train_er.append(train_error)
                    cost.append(_cost)
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
        print("total Time for one epoch used is %d", % (round(elapsed,4)))
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

