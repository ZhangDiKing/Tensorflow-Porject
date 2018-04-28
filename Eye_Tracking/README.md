# Eye-Gaze-Estimation
The final course project for deep learning in RPI.
The training datasets are provided by the professor, you need read the paper to download the data.
The test datasets are hold by the professor to test the accuracy of the CNN model.
    
## Convolutional Neural Network Visualization
![graph_visualization](https://github.com/dzk9528/Tensorflow-Project/blob/master/Eye_Tracking/graph_vis.png)

## Error Plot
The validation error is 1.77 cm for my course, but after fine-tuning the paramters as well as refering the official model, the validation error is now 1.65 cm.<br />
![error_plot](https://github.com/dzk9528/Tensorflow-Project/blob/master/Eye_Tracking/error_plot.png| width=100)

## Reference
The CNN design follows the paper [Eye Tracking for Everyone](http://gazecapture.csail.mit.edu/cvpr2016_gazecapture.pdf). <br />
By the way, they have just release their orignal caffe/matlab/pytorch version on github. Here is their [link](https://github.com/CSAILVision/GazeCapture).
    
## update
Due to different datasets, My model setting is different from the orignal ones. But I change the structure of model to make it similar to the model in the paper, in which LRN is added for the first and second convolutional layers, max pooling for CNN layers has kernel size=3 with a stride of 2 and no dropout in fully connected layers. But the weight size is still different for the limitation of mt GPU, but You can configure the model in the follwing way.
```
    #network config here, suit your own model
    fc_d = [
        #eye fc in cnn config
        [32],
        #face fc in cnn config
        [64, 32]
    ]
    cnn_d = [
        #eye cnn channels config
        [32, 32, 32, 32],
        #face cnn channels config
        [32, 32, 32, 32]
    ]
    filter_size = [
        #face filter in cnn config
        [ [5, 5], [7, 7], [5, 5], [1, 1] ],
        #face filter in cnn config
        [ [5, 5], [7, 7], [5, 5], [1, 1] ]
    ]
    #face mask nn cofig
     mask_fc_d = [32 ,32]

    #the layer config for cat different features
    cat_fc_d = [32, 32]
    
    model = eye_track_model([eye_left, eye_right, face, face_mask], # four place-holders you have
                            y, #x-y coordinate you have
                            in_channels, 
                            fc_d, 
                            mask_fc_d,
                            cat_fc_d,
                            cnn_d,
                            filter_size)
    loss, error, predict_op = model.get_param()
```
By the way, if you find it useful, I would appreicate your stars.
