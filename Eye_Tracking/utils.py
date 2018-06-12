import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def hyper_setting(default_setting = True):
    hyper_para = {}
    if default_setting:
        hyper_para["fc_d"] = [
            #eye fc in cnn config
            [],
            #face fc in cnn config
            [64, 32]
        ]

        #cnn setting
        hyper_para["cnn_d"] = [96, 256, 384, 64]   
        hyper_para["filter_size"] = [[11, 11], [5, 5], [3, 3], [1, 1]] #filter size in cnn config
        hyper_para["cnn_strides"] = [4, 1, 1, 1]
        hyper_para["cnn_padding"] = [0, 2, 1, 0]

        #pooling layer setting
        hyper_para["pooling_k_size"] = [3, 3]
        hyper_para["pooling_strides"] = [2, 2]
        hyper_para["pooling_padding"] = ['SAME', 'SAME']

        #number of LRN 
        hyper_para["n_LRN"] = 2

        #face mask neural network 
        hyper_para["mask_fc_d"] = [256 ,128]

        #the layer config for cat different features
        hyper_para["cat_fc_d"] = [128, 128]
        
        #input channel
        hyper_para["in_channels"] = 3

        #learning rate and decay
        hyper_para["decay"] = 0.5
        hyper_para["learning_rate"] = 0.001
    else:
        '''
        Your own setting here.
        '''

        hyper_para["fc_d"] = [
            #eye fc in cnn config
            [],
            #face fc in cnn config
            [64, 32]
        ]

        #cnn setting
        hyper_para["cnn_d"] = [32, 32, 64, 32]   
        hyper_para["filter_size"] = [[5, 5], [7, 7], [5, 5], [1, 1]] #filter size in cnn config
        hyper_para["cnn_strides"] = [1, 1, 1, 1]
        hyper_para["cnn_padding"] = [0, 0, 0, 0]

        #pooling layer setting
        hyper_para["pooling_k_size"] = [3, 3]
        hyper_para["pooling_strides"] = [2, 2]
        hyper_para["pooling_padding"] = ['SAME', 'SAME']

        #number of LRN 
        hyper_para["n_LRN"] = 2

        #face mask neural network 
        hyper_para["mask_fc_d"] = [32 ,32]

        #the layer config for cat different features
        hyper_para["cat_fc_d"] = [32, 32]
        
        #input channel
        hyper_para["in_channels"] = 3

        #learning rate and decay
        hyper_para["decay"] = 0.5
        hyper_para["learning_rate"] = 0.001

    return hyper_para

def get_batch(data, batch_size, i):
    out = []
    for k in range(5):
        out.append(data[k][int(i*batch_size):int((i+1)*batch_size)])
    return out

def read_data(path, debug = False):
    if debug:
        print("start preprocesing data...")
        start = time.time()
    npzfile = np.load(path + "train_and_val.npz")

    if not npzfile:
        print("No data found")

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
    if debug:
        print("load dataset in %s seconds." % (round(time.time() - start,4)))
    return train_data, val_data

def draw_plot(cost, test_er, train_er, error_step, test_step):
    #plot the trainning error and test error
    fig,ax1 = plt.subplots()
    c_l = ax1.plot(error_step, cost, 'r-', label = 'training loss')
    ax1.set_xlabel('step')
    
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('loss', color = 'b')
    ax1.tick_params('y', colors = 'b')
    ax2 = ax1.twinx()
    tr_l = ax2.plot(error_step, train_er, 'g-', label = 'batch training error')
    te_l = ax2.plot(test_step, test_er, 'b-', label = 'test error')
    ax2.set_ylabel('error/cm', color = 'orange')
    ax2.tick_params('y', colors='orange')
    lns = c_l+tr_l+te_l
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc = 0)
    fig.tight_layout()
    plt.show()

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
