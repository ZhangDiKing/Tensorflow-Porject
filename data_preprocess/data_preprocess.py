# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:04:54 2018
Most of the following part is following the repo:
    https://github.com/ycszen/TensorFlowLaboratory/blob/master/reading_data/example_tfrecords.py
Except I add parsor as input options and the labels-names can be taken as input
for the speciifc dataset I use only use filename as labels.
@author: zhang
"""
import os
import tensorflow as tf
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/fruits-360/Training/',type=str)
parser.add_argument('--save_path', default='/data_tfrecords/',type=str)
parser.add_argument('--tfrecord_name',default='data.tfrecords',type=str)
parser.add_argument('--label_name', default='',type=str)
parser.add_argument('--width', default='32',type=int)
parser.add_argument('--height', default='32',type=int)


def transfer_to_tf_recorder(writer_name, path, filenames, label_names, width, height):
    """
    This fuction basically follow the code on the repo, which transfer the iamge data into tf recorder;
    I used it to preprocess the fruit 360 data
    Args:
        writer_name (str): the name of the tf record writer(.tfrecords)
        path (str): the absolute path of the data
        filenames (list if str): the list of image filenames
        label_names (dict): dictioinary with label_name (str) as key and index (int) as val
    """

    #the writer of tf record
    writer = tf.python_io.TFRecordWriter(writer_name)
    size=0
    for folder in filenames:
        index=label_names[folder]
        for file in os.listdir(path+'/'+folder+'/'):
            img = Image.open(path+'/'+folder+'/'+file)

            if not img:
                raise ValueError("The image file", file,"is empty.")
            img = img.resize((width,height))
            #transfer img to btyes
            img_raw = img.tobytes()

            #transfer the data into the tf recorders
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
            size+=1
        #comment this line if you don,t want too much information on the cmd/bash
        print("Successfully convert file with index", index)
    print("totally There are", size, "samples")
    writer.close()
    
    return 
def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    
    #access the absolute path of the data
    cwd = os.getcwd()
    filenames=os.listdir(cwd+opt.data_path)
    
    if not filenames:
        raise ValueError("Cannot Find any folder. Please input a validate data path.")
        
    if not opt.label_name:
        label_names=dict (zip(filenames, range(len(filenames))))
    else:
        label_names=opt.label_name.split('-')
        label_names=dict (zip(label_names, range(len(label_names))))

    transfer_to_tf_recorder(cwd + opt.save_path + opt.tfrecord_name, cwd + opt.data_path, 
                            filenames, label_names, opt.width, opt.height)
    
if __name__ == '__main__':
    main()
