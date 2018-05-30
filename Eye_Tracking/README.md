# Eye-Gaze-Estimation
The final course project for deep learning in RPI.
The training datasets are provided by the professor, you need read the paper to download the data.
The test datasets are hold by the professor to test the accuracy of the CNN model.
    
## Convolutional Neural Network Visualization
![graph_visualization](https://github.com/dzk9528/Tensorflow-Project/blob/master/Eye_Tracking/graph_vis.png)

## Error Plot
The validation error is 1.77 cm when I finish it in my course. After fine-tuning the paramters as well as refering the official model, the validation error is now 1.65 cm.<br />
<img src="https://github.com/dzk9528/Tensorflow-Project/blob/master/Eye_Tracking/error_plot.png" width="400">

## Reference
The CNN design follows the paper [Eye Tracking for Everyone](http://gazecapture.csail.mit.edu/cvpr2016_gazecapture.pdf). <br />
By the way, they have just release their orignal caffe/matlab/pytorch version on github. Here is their [link](https://github.com/CSAILVision/GazeCapture).
    
## Update
Due to different datasets, My model setting is different from the orignal ones. But I change the structure of model to make it similar to the model in the paper, in which LRN is added for the first and second convolutional layers, max pooling for CNN layers has kernel size=3 with a stride of 2 and no dropout in fully connected layers. But the weight size is still different for the limitation of my GPU, but You can configure the model in the hyper_setting() function in the uitls.py. <br />

By the way, if you find it useful, I would appreicate your stars.
