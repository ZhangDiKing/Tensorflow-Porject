# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 11:14:37 2018

@author: zhang
"""
import numpy as np
import tensorflow as tf

def transfer_data(path,batch_size,epoch,test_mode=False):
    #load data
    #you can change the path to load npz
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

    training_dataset = tf.data.Dataset.from_tensor_slices((train_eye_left, train_eye_right,
                                                  train_face,train_face_mask,train_y))
    
    validation_dataset = tf.data.Dataset.from_tensor_slices((val_eye_left, val_eye_right,
                                                  val_face,val_face_mask,val_y))

    if test_mode:
        print(training_dataset)
        print(validation_dataset)
    #separate batch
    training_dataset=get_batch(training_dataset,batch_size,epoch)
    validation_dataset=get_batch(validation_dataset,batch_size,epoch)
    
    #generate iterator
    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                   training_dataset.output_shapes)
    next_element = iterator.get_next()

    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)
    
    return training_init_op,validation_init_op,next_element

def normalize(data,min_v,max_v):
    avg=(max_v-min_v)/2.0
    return (data-avg)/(max_v-min_v)

def get_batch(dataset,batch_size,epoch):
    dataset =dataset.shuffle(5000)
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size)
    return dataset

