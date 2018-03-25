# Experiment of uitlizaing the offical resnet to train your own dataset
    Recently tensorflow has released its own resnet model, so followed by the cifar10_main.py
    in the repo:
    https://github.com/tensorflow/models/tree/master/official/resnet
    I change part of the code to make the model accept for my own dataset. 
    Maybe you can use it as a tutorial for feeding your own data into the built in resnet.
    
    The model is still updated.
    
## Environment: Tensorflow 1.5 + CUDA 9.0 + win10 
    Make sure you already transfer your dataset into tfrecords
    Make sure you already installed the model in your system
    

## Parameters setting
### 1. image size
    Modfiy 
    
    _HEIGHT = 32
    _WIDTH = 32
    _NUM_CHANNELS = 3
    
    in the resnet_train_general.py

### 2. datasets size:
    Modfiy 
    
    _NUM_IMAGES = {
    'train': train_size,
    'validation': test_size,
    }
    
    in the resnet_train_general.py
    
### 3. number of classes:
    Modfiy 
    
    _NUM_CLASSES = number_of_classes
    
    in the resnet_train_general.py
    
### 3. datasets form:   
    training dataset: 
        train.tfrecords 
    test dataset:
        test.tfrecords
    
    Which has:
    features = {
        'label':tf.FixedLenFeature([],tf.int64),
        'image':tf.FixedLenFeature([],tf.string)
    }
    The label for each sample is a number in the range [0, _NUM_CLASSES)

### 3. learning rate setting
    The officical model assume the intial learning rate = 0.1 * batch_size / batch_denom and have a built in weight_decay program
    
    
    if 
    boundary_epochs=[e0, e1, e2],
    decay_rates=[d0, d1, d2, d3]
    
    By modify d0, you can make intial learning rate = 0.1 * batch_size / batch_denom * d0
    
    After e0 epoch:
    learning rate = 0.1 * batch_size / batch_denom * d1
    
    After e1 epoch:
    learning rate = 0.1 * batch_size / batch_denom * d2
    
    ...
	
    However, if your batch size is very large, the learning rate is obivious too large. I tried a different dataset and find out that the gradients dropped very fast.
    
    Therefore, maybe smaller setting of boundary_epochs and smaller initial learning rate is a better choice

### 4. running time
    I appreicate the efficiency of the model very well for I only have a laptop with a gpu GTX960m
    
    For a 32 layer resnet, with input size (128, 32, 32 ,3), it takes 20s to finish 100 steps!
    
    tf.dataset is trully fast methods.


    

 

