# Eye-Gaze-Estimation
The final course project for deep learning in RPI.
The training datasets are provided by the professor, you need read the paper to download the data.
The test datasets are hold by the professor to test the accuracy of the CNN model.
    
## Convolutional Neural Network Design
![graph_visualization](https://user-images.githubusercontent.com/24198258/32213439-9d9856e4-bdf1-11e7-92b2-d1010ca584e4.png)

|Layer Name	|Input Dimension|Weight Size|Output Dimension|Activation Function |
|:---------:|:-------------:|:---------:|:--------------:|:------------------:|
|conv1_xx|(250, 64, 64, 3)|	(5, 5, 3, 32)|	(250, 60, 60, 32)|	Relu|
|conv1_xx_pooling|(250, 60, 60, 32)| |(250, 30, 30, 32)| |
|conv2_xx|(250, 30, 30, 3)|	(7, 7, 32, 32)| (250, 24, 24, 32) |	Relu|
|conv2_xx_pooling|(250, 24, 24, 32)| |	(250, 12, 12, 32)| |
|conv3_xx |(250, 12, 12, 3)|(5, 5, 3, 32)|	(250, 8, 8, 32)|	Relu|
|conv3_xx_dropout|(250, 8, 8, 32) |		|(250, 8, 8, 32)| |
|fc1_xx|(250, 8 * 8 * 32)|(8 * 8 * 32, 32)|(250, 32)|Relu|
|face_mask_dropout|(250, 25 * 25)| |(250, 25 * 25)| |	
|fc1_face_mask|(250, 25 * 25)|	(25 * 25, 32) |	(250, 32) |	Relu |
|fc2_face_mask|	(250, 32) |(32, 32) |	(250, 32) |	Relu |
|fc1_eye|(250, 32 * 2)|(32 * 2, 32) |	(250, 32) |	Relu |
|fc1_face_eye_mask |(250, 32 * 3) |	(32 * 3, 32) |	(250, 32)  |	Relu|
|final_out |(250, 32) |	(32, 2) |	(250, 2)||	


## Error Plot
Finally, the test error is 1.77.<br />
![error_plot](https://user-images.githubusercontent.com/24198258/32213509-ec1c6a58-bdf1-11e7-936e-cbcce6e6f56e.png)

## Reference
The CNN design follows the paper [Eye Tracking for Everyone](http://gazecapture.csail.mit.edu/cvpr2016_gazecapture.pdf). <br />
By the way, they have just release their orignal caffe/matlab/pytorch version on github. Here is their [link](https://github.com/CSAILVision/GazeCapture).
    
## update
Due to different datasets, My model setting is different from the orignal ones so that I change the structure of model to make it similar to the model in the paper, in which LRN is added and no dropout in fully connected layers. You can use the model in the follwing way.
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
   
