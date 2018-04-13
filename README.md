[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
# Abstract
The goals / steps of this project are the following:
* Use the [simulator](https://github.com/udacity/self-driving-car-sim) to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report (see [rubric points](https://review.udacity.com/#!/rubrics/432/view) )


[//]: # (Image References)

[imageLeftTurn]: ./images/center_2018_04_13_11_07_07_167.jpg "Turn Left with Side Marks"
[imageSandRight]: ./images/center_2018_04_13_11_05_49_878.jpg "Straight With Sand on the Right"
[imageLaneLines]: ./images/center_2018_04_13_11_07_54_308.jpg "Straight With Lane Lines"
[imageBridge]: ./images/center_2018_04_13_11_07_58_683.jpg "Bridge"
[imageCenter]: ./images/center_2018_04_13_11_07_08_656.jpg "Center Camera"
[imageCenterFlipped]: ./images/center_2018_04_13_11_07_08_656_flipped.jpg "Center Camera Flipped"
[imageLeft]: ./images/left_2018_04_13_11_07_08_656.jpg "Left Camera"
[imageLeftFlipped]: ./images/left_2018_04_13_11_07_08_656_flipped.jpg "Left camera Flipped"
[imageRight]: ./images/right_2018_04_13_11_07_08_656.jpg "Right Camera"
[imageRightFlipped]: ./images/right_2018_04_13_11_07_08_656_flipped.jpg "Right Camera Flipped"
[imageNvidiaNet]: ./images/NvidiaNet.png "NvidiaNet"


# Overview of Files

My project includes the following files:
* [README.md](https://github.com/MarkBroerkens/CarND-Behavioral-Cloning-P3/blob/master/README.md) (this writeup report) summarizing the results 
* [model.py](https://github.com/MarkBroerkens/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to train the model
* [nvidianet.py](https://github.com/MarkBroerkens/CarND-Behavioral-Cloning-P3/blob/master/nvidianet.py) containing the script to create the model
* [drive.py](https://github.com/MarkBroerkens/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/MarkBroerkens/CarND-Behavioral-Cloning-P3/blob/master/model.h5) containing a trained convolution neural network 
* [video.mp4](https://github.com/MarkBroerkens/CarND-Behavioral-Cloning-P3/blob/master/video.mp4) shows the simulator in autonomous mode using the trained convolutional network

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

# Model Architecture 
The overall strategy for deriving a model architecture was to start with a very simple architecture in order to first setup a working end-to-end framework and to check all functionality (training, driving, simulator, video playbacks) and detect potential technical problems.

Then I replaced the simple network by the  [convolution model from Nvidia](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) that was introduced in the class. The original Nvidia Net is described in the following image.

![Architecture of Conv Net by Nvidia][imageNvidiaNet]

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added dropout layers into model.

Then I added a cropping layer in order to support focusing on the relevant parts of the images.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. It turned out that the cv2 lib is reading images in BGR format, and the drive.py providing images in RGB. Thus I added the conversion from BGR to RGB after loading the images.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

My model is based on the [convolution model from Nvidia](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) that was introduced in the class. The original Nvidia Net is described in the following image.

My final network consists of 11 layers, including 
* 1 cropping layer ([nvidianet.py](https://github.com/MarkBroerkens/CarND-Behavioral-Cloning-P3/blob/master/nvidianet.py) line 15)
* 1 normalization layer ([nvidianet.py](https://github.com/MarkBroerkens/CarND-Behavioral-Cloning-P3/blob/master/nvidianet.py) line 18)
* 5 convolutional layers and ([nvidianet.py](https://github.com/MarkBroerkens/CarND-Behavioral-Cloning-P3/blob/master/nvidianet.py) lines 21-25)
* 4 fully connected layers with dropouts ([nvidianet.py](https://github.com/MarkBroerkens/CarND-Behavioral-Cloning-P3/blob/master/nvidianet.py) lines 31-37)

The input image is split into RGB planes and passed to the network.

The first layer of the network crops the images and removes the bottom and top parts that do not contribute to the calculation of the steering angle (bottom part contains the hood of the car and the top part captures trees and hills and sky). The normalizer is hard-coded and is not adjusted in the learning process. Performing normalization in the network allows the normalization scheme to be altered with the network architecture and to be accelerated via GPU processing.
The convolutional layers were designed to perform feature extraction. The network uses strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel  and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers.
After that fully connected layers leading to an output the steering angle.

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (TODO code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually (TODO model.py line 25).

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

# Model Training Documentation

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][imageLeftTurn]
![alt text][imageSandRight]
![alt text][imageLaneLines]
![alt text][imageBridge]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

In order to avoid left turn bias I added flipped images and angles to the data set. For example, here is an image that has then been flipped:

![alt text][imageCenter]
![alt text][imageCenterFlipped]

Additionally, I used the images from the left and right cameras. For right camera I augmented the angle so that the car should drive more to the left and for the left camera I augmented the angle so that the car should drive more to the right.
![alt text][imageLeft]
![alt text][imageLeftFlipped]
![alt text][imageRight]
![alt text][imageRightFlipped]

Then I repeated this process on track two in order to get more data points.

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
