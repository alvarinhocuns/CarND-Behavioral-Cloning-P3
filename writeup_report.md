

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/center_2017_09_12_19_51_48_903.jpg "Recovery Image"
[image3]: ./images/center_2017_09_13_20_05_54_061.jpg "Recovery Image"
[image4]: ./images/center_2017_09_13_20_07_24_259.jpg "Recovery Image"
[image5]: ./images/center_2017_09_13_20_09_10_731.jpg "Recovery Image"
[image6]: ./images/center_2017_09_13_20_18_22_859.jpg "Recovery Image"
[image7]: ./images/normal.jpg "Normal Image"
[image8]: ./images/flipped.jpg "Flipped Image"
[image9]: ./images/error_loss.png "Error loss"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 video demonstration of the network performance

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 8x8 (fist layer) and 5x5 (second and third layers) filter sizes and depths between 16 and 64 (model.py lines 57-79) 

The model includes RELU layers after the convolutional layers and ELU layers after the flatten and first fully connected layer to introduce nonlinearity. This configuration was selected empirically after a some experiments. The data is normalized in the model using a Keras lambda layer (code line 60). 

####2. Attempts to reduce overfitting in the model

The model contains one dropout layer in order to reduce overfitting (model.py line 75). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 83). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 82).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 
For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the one used in the [nVidia self-driving car](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). I thought this model might be appropriate because is was thought to solve the same kind of problem as I am working on.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on the training and validation sets. I thought that this NN is too big for such a simple problem, so I decided to start over with a different and more simple achitecture.
Then I tried a network based on the [model](https://github.com/commaai/research/blob/master/train_steering_model.py) from comma.ai. but much more simple in order to get an overfitted model. At this point I found that the model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model including a dropout layer with a rate of 0.5. After doing this the mean squared errors on the training and validation sets were low and similar, so I decided to move on and try the model in the simulator.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, for exaple the brige and the turns with that dirt road entrances. To improve the driving behavior in these cases, I took more data in this regions.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to the center of the road in case of deviation. These images show what a recovery looks like starting from:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]


To augment the data set, I also flipped images and angles thinking that this would augment the data sample. For example, here is an image that has then been flipped:

![alt text][image7]
![alt text][image8]


After the collection process, I had 9677 number of data points. Each of these points provide three different points of view to the road, centre, left and right. Son the actual number of points is 29031. So finally, after flipping the images I got 58062 images. I then preprocessed this data by cropping the upper part and lower part of the images, and also normalizing them to obtain values in the range (-1,1).

I finally randomly shuffled the data set and put 0.2% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by the fact that more epochs didn't improve the result. In the next image can be seen the reduction in the mean squared error per epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image9]