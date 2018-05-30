import numpy as np
import tensorflow as tf
import time
import argparse

from utils import get_batch, load_model

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default = '', type = str)

def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))

    batch_size = 50
    test_size = 5000
    path = opt.data_path
    print("start preprocesing data...")
    start = time.time()
    npzfile = np.load(path+"train_and_val.npz")
    val_data = [
        npzfile["val_eye_left"],
        npzfile["val_eye_right"],
        npzfile["val_face"],
        npzfile["val_face_mask"],
        npzfile["val_y"]
    ]
    print("load dataset in %s seconds." % (round(time.time() - start,4)))

    with tf.Session() as sess:
        test_error = 0
        
        sess.run(tf.global_variables_initializer())
        start = time.time()
        '''
        You may change the way of load your data here
        '''
        eye_left, eye_right, face, face_mask, predict_op = load_model(sess, path)
        print("lood model successfully in %s seconds." % (round(time.time() - start,4)))
        start = time.time()
        
        for i in range(int(test_size / batch_size)):
            batch_data = get_batch(val_data, batch_size, i)
            with tf.device("/gpu:0"):
                y_batch = sess.run(predict_op, 
                                        feed_dict={
                                                eye_left: batch_data[0]/ 255.-0.5, 
                                                eye_right: batch_data[1]/ 255.-0.5,
                                                face: batch_data[2]/ 255.-0.5,
                                                face_mask: batch_data[3],
                                                })
                err = np.sum(np.sqrt(np.sum((batch_data[4] - y_batch)**2, axis=1)))
                test_error += err
            print(str(i) + " batch finished")
        print("the validation error is %s cm." %(round(test_error / test_size, 4)))

if __name__ == "__main__":
    main()