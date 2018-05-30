# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:34:27 2018
@author: zhang
"""
import tensorflow as tf
import argparse

from eye_track_model import eye_track_model
from utils import read_data

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default = '', type = str)
parser.add_argument('--save_path', default = '', type = str)

def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))

    train_data, val_data = read_data(opt.data_path)

    model = eye_track_model()
    model.fit(train_data, 
            val_data, 
            opt.save_path,
            batch_size = 250,
            epoch = 80)
    
if __name__=="__main__":
    main()

