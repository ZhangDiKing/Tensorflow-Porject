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
import random

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/train/train/',type=str)
parser.add_argument('--is_for_train',default='True',type=bool)
parser.add_argument('--validate_rate',default='0.3',type=float)
parser.add_argument('--label_name', default='',type=str)

def transfer_to_tf_recorder(writer_name, path, filenames, label_names):
    """
    This fuction basically follow the code on the repo, which transfer the iamge data into tf recorder
    Args:
        writer_name (str): the name of the tf record writer(.tfrecords)
        path (str): the absolute path of the data
        filenames (list if str): the list of image filenames
        label_names (dict): dictioinary with label_name (str) as key and index (int) as val
    """

    #the writer of tf record
    writer = tf.python_io.TFRecordWriter(writer_name)

    #initialize gt
    gt_init=[0 for i in range(len(label_names.keys()))]

    for file in filenames:
        img = Image.open(path+file)

        if not img:
            raise ValueError("The image file", file,"is empty.")

        #transfer img to btyes
        img_raw = img.tobytes()

        #find corresponding index of the name
        index, gt=-1, gt_init[:]
        for k in label_names:
            if k in file:
                index=label_names[k]
                break

        if index==-1:
            raise ValueError("Label_name does not exist on the file name. Please change file name or reinput label name.")
        gt[index]=1
        #transfer the data into the tf recorders
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[gt])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))

        #comment this line if you don,t want too much information on the cmd/bash
        print("Successfully convert", file, "with gt", gt)
        writer.write(example.SerializeToString())
    writer.close()
    return 0

def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))

    if not opt.label_name:
        raise ValueError("Label_name is empty.Please input labels name.")

    #access the absolute path of the data
    cwd = os.getcwd()
    filenames=os.listdir(cwd+opt.data_path)
    if not filenames:
        raise ValueError("Cannot Find any file. Please input a validate data path.")

    #transfer the label name into dictionary to get correspnding index
    label_names=opt.label_name.split('-')
    label_names=dict (zip(label_names, range(len(label_names))))

    if opt.is_for_train:
        n_val = int(opt.validate_rate * len(filenames))
        print("The number of validation data is", n_val)

        #shuffle to splite trainning tf-reccotds and validation tf-records
        random.seed(0)
        random.shuffle(filenames)
        train_filenames = filenames[n_val:]
        validation_filenames = filenames[:n_val]

        #separately transfer the image data with labels to tf recorder
        transfer_to_tf_recorder("train.tfrecords", cwd+opt.data_path, train_filenames, label_names)
        transfer_to_tf_recorder("validate.tfrecords", cwd+opt.data_path, validation_filenames, label_names)

    else:
        transfer_to_tf_recorder("test.tfrecords", cwd+opt.data_path, filenames, label_names)

if __name__ == '__main__':
    main()
