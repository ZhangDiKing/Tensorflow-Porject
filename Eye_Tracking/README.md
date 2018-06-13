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
Due to different datasets, My model setting is different from the orignal ones. But I change the structure of model to make it similar to the model in the paper, in which LRN is added for the first and second convolutional layers, max pooling for CNN layers has kernel size=3 with a stride of 2 and no dropout in fully connected layers. You can configure the model in the hyper_setting() function in the [uitls.py](https://github.com/dzk9528/Tensorflow-Project/blob/master/Eye_Tracking/utils.py). <br />

## AWS experiment
I tried p2-xlarge Deep Learning AMI -Ubuntu--10-0 ec2 instance to trained the model faster and by given more epoches the final error is 1.65 cm. I use wget to download from AWS s3 to AWS ec2 instance. The speed for AWS GPU Tesla K80 for my model is 90s / epoch which is 15 % faster than my local GPU gtx 960m is 120s / epoch. Maybe a better data pipline or GPU ec2 instance is a better idea?


By the way, if you find it useful, I would appreicate your stars.
