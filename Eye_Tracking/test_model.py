import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pickle
import os

from eye_track_model import eye_track_model 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='C:/Users/zhang/dp/dP_final/',type=str)

def get_batch(data, batch_size, i):
    out=[]
    for k in range(5):
        out.append(data[k][int(i*batch_size):int((i+1)*batch_size)])
    return out

def load_model(sess,path):
    meta_file = path + "my-model-72.meta"

    saver = tf.train.import_meta_graph(meta_file)
    
    saver.restore(sess, path+'./'+'my-model-72')

    valid_nodes = tf.get_collection_ref("validation_nodes")
    eye_left = valid_nodes[0]
    eye_right = valid_nodes[1]
    face = valid_nodes[2]
    face_mask = valid_nodes[3]
    predict = valid_nodes[4]
    return [eye_left, eye_right, face, face_mask, predict]

def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))

    batch_size=50
    test_size =5000
    path=opt.data_path
    print("start preprocesing data...")
    start=time.time()
    npzfile = np.load(path+"train_and_val.npz")
    val_data=[
        npzfile["val_eye_left"],
        npzfile["val_eye_right"],
        npzfile["val_face"],
        npzfile["val_face_mask"],
        npzfile["val_y"]
    ]
    print("load dataset in %s seconds." % (round(time.time() - start,4)))

    with tf.Session() as sess:
        test_error=0
        
        sess.run(tf.global_variables_initializer())
        start=time.time()
        '''
        Attenttion! You may change the way of load your data here
        '''
        eye_left, eye_right, face, face_mask, predict_op = load_model(sess, path)
        print("lood model successfully in %s seconds." % (round(time.time() - start,4)))
        start=time.time()
        
        for i in range(int(test_size/batch_size)):
            [val_eye_left_batch, \
                val_eye_right_batch, \
                val_face_batch, \
                val_face_mask_batch, \
                val_y_batch] = get_batch(val_data, batch_size, i)
            with tf.device("/gpu:0"):
                y_batch = sess.run(predict_op, 
                                        feed_dict = {
                                                eye_left: val_eye_left_batch/ 255.-0.5, 
                                                eye_right: val_eye_right_batch/ 255.-0.5,
                                                face:val_face_batch/ 255.-0.5,
                                                face_mask:val_face_mask_batch
                                                })
                err = np.sum(np.sqrt(np.sum((val_y_batch - y_batch)**2, axis=1)))
                test_error += err
            print(str(i)+" batch finished")
        print("the validation error is %s cm." %(round(test_error/test_size,4)))

if __name__=="__main__":
    main()