# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 22:40:03 2018

@author: zhang
"""
import numpy as np
import tensorflow as tf
import time
import pre_processing as pre
def main():
    in_channels=3
    batch_size=125
    hc_d=64
    epoch=1
    decay=0.5
    train_size=48000
    test_size =5000
    path='C:/Users/zhang/dp/dP_final/'
    print("start preprocesing data...")
    start=time.time()
    training_init_op,validation_init_op,next_element=pre.transfer_data(path,batch_size,epoch)
    print("transfer dataset into tf.dataset in %s seconds." % (round(time.time() - start,4))) 
    
    sess        =tf.Session()
    sess.run(tf.global_variables_initializer())
    print('start training...')
    for k in range(0,epoch):
        
        sess.run(training_init_op)
        print('output training data')
        for i in range(int(train_size/batch_size)):
            (train_eye_left_batch, \
             train_eye_right_batch, \
             train_face_batch, \
             train_face_mask_batch, \
             train_y_batch)=sess.run(next_element)
            
            #print shape to compare result
            print(train_eye_left_batch.shape, \
                 train_eye_right_batch.shape, \
                 train_face_batch.shape, \
                 train_face_mask_batch.shape, \
                 train_y_batch.shape)
        
        sess.run(validation_init_op)
        print('output validation data')
        for i in range(int(test_size/batch_size)):
            (val_eye_left_batch, \
             val_eye_right_batch, \
             val_face_batch, \
             val_face_mask_batch, \
             val_y_batch)        =sess.run(next_element)
            #print shape to compare result
            print(val_eye_left_batch.shape, \
                 val_eye_right_batch.shape, \
                 val_face_batch.shape, \
                 val_face_mask_batch.shape, \
                 val_y_batch.shape)
if __name__=="__main__":
    main()