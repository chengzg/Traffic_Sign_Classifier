# **Traffic Sign Recognition** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/original_data_visualization.png "Distribution before Augmentation"
[image2]: ./examples/Rotation_Augmented.png "rotation"
[image3]: ./examples/Scaling_Augmented.png "scaling"
[image4]: ./examples/Affined_Augmented.png "Affine Transform"
[image5]: ./examples/Distribution_After_Augmentation.png "Distribution After Augmentation"
[image6]: ./examples/After_Enhance_Contrast.png "Enhance contrast"
[image7]: ./examples/After_Grayout.png "Grayout"
[image8]: ./examples/After_Normalization.png "After Normalization"


[image9]: ./German_Traffic_Signs/1.jpg "1"
[image10]: ./German_Traffic_Signs/2.jpg "2"
[image11]: ./German_Traffic_Signs/3.jpg "3"
[image12]: ./German_Traffic_Signs/4.png "4"
[image13]: ./German_Traffic_Signs/5.png "5"
[image14]: ./German_Traffic_Signs/6.png "6"
[image15]: ./German_Traffic_Signs/7.png "7"
[image16]: ./German_Traffic_Signs/8.png "8"
[image17]: ./examples/Correct_5.png "Correct 5 examples"
[image18]: ./examples/Missing_3.png "Missing 3 examples"
[image19]: ./examples/missing_3_validation_images.png "Missing 3 validation images"
[image20]: ./examples/conv1_featuremap_100km.png "100km convolution layer 1 output feature map"
[image21]: ./examples/conv1_featuremap_general_caution.png "general caution convolution layer 1 output feature map"

[image22]: ./examples/validation_accuracy_graph.png "After Normalization"
[image23]: ./examples/conv1_featuremap_straight_or_right.png "go straight or right"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
# **Here is a link to my [project code](https://github.com/chengzg/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)**


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43 which includes all training, validation and test data

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data categories are distributed.
From the graph, we can see that the training data is not even distributed. The biggest amount is around 8 times of the smallest amount. 
This gives a hint that we probably should augment the training data to make them more evenly distributed

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
As mentioned earlier, the original training data set is not even distributed. So the first step of preprocessing the dataset is to augment the original dataset to make it move evenly distributed. Based on the original dataset, i use serveral steps to augment the data:
* rotate the original image by certain degree
* scale the original image by some factor
* Affine tranform the original image

To decided which category to augment, i use the current training set average number as the pivot number. If the current category is less than the average, then i will add the augmented image into the training set. The original image to be augmented is randomly choosen among the original dataset.
Refer to the image below for more information:

![alt text][image2]

![alt text][image3]

![alt text][image4]

After the augmentation step, we can see that the training dataset is more evenly distributed now

![alt text][image5]

The total training dataset now has increaed from 33479 to 59155

After the augmentation step, i used the follow steps to preprocess the training dataset:

    1. Enhance Contrast
After examining some of the dataset, i found that some original images are very difficult to recognized. It would be easier to find the features after enhancing the image contrast. 

![alt text][image6]

    2. Grayout
The reason i decided to grayout the image is that i feel the color is not very important to the meaning of the traffic sign rather the shape is more important. By converting it to gray image, i can reduce the computation to only 1/3 of the color image becase now there is only a single channel or depth.

![alt text][image7]

    3. Normaliztion

![alt text][image8]

From the above picture, it seems that the difference is not as good as the contrast enhancement and the image grayout technique. But it is very important to the success of the training model. Normalization can make the training faster and reduce the chances of getting stuck in local optima.
It transposes the input variables into the data range of [-1, 1] so that the mse is small. In this case, the training rate can have a relative small number which could help reduce the saturation.  




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28X24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10X64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				    |
| Fully connected 1		| outputs 480  									|
| Fully connected 2		| outputs 320  									|
| Fully connected 3		| outputs 160  									|
| Fully connected 4		| outputs 43  									|
|						|												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an updated version of LeNet. Since the original LeNet is developed for MNIST classification which has only 10 outputs. In our scenario, we have 43 outputs, so i scale the neural network by a factor of 4. So the output of the first layer becaome  6*4 = 24. And all subsequent layers are scaled by this factor 4. The last full connected layer output ramained the same as 43. From the second FC layer output 320 to the last 43 is a big change, so i introducted another hidden layer in between which has a output of 160 nodes to reduce the gap in between.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of approximately 98%  
* test set accuracy of 96.3%

And the validation accuracy graph is shown below:

![alt text][image22]

If an iterative approach was chosen:

* What architecture was chosen?

    
    The LeNet architecture was choose because it is a proven good architecture for hand writing digits classification problem.

* Why did you believe it would be relevant to the traffic sign application?
    
    
    In our cases, we want to classify the traffic sign and the input data sets shape is also 32*32. The only difference is the final otput nodes is 43 which we can easily adjust.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    
    
    It is able to achieve 96.3% of test accuracy and 98% of validation accuracy. Refer to above graph for more information on training process.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 8 German traffic signs that I found on the web:

![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12]

![alt text][image13] ![alt text][image14] ![alt text][image15] ![alt text][image16] 

In theory, the 3rd image(Children crossing) should be relatively difficulty to classify because it has a lot of noisy data inside. The reset of the images should be easy to classify since they are pretty clear.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                                        |     Prediction	        				|              result             | 
|:-----------------------------------------------------:|:-----------------------------------------:|:-------------------------------:| 
| Pedestrians      		                                | Children crossing                  		|		       wrong		      | 
| Right-of-way at the next intersection     			| Right-of-way at the next intersection		|		       correct	     	  |
| Children crossing					                    | Bicycles crossing							|		       wrong              |
| Speed limit (100km/h)	                        		| Speed limit (80km/h)					 	|		       wrong	          |
| Go straight or right		                         	| Go straight or right      				|		       correct            |
| General caution                                       | General caution                           |              correct            |
| Wild animals crossing                                 | Wild animals crossing                     |              correct            |
| Keep left                                             | Keep left                                 |              correct            |

    The model was able to correctly guess 5 of the 8 traffic signs, which gives an accuracy of 62.5%. This is comparably low with respect to the test accuracy of 96.3%.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Below image shows the correct prediction of the images. From the image we can see that 4 of 5 correct prediction are very close to 1 which means they are quite sure about the prediction.
Only the last one is around 0.75 percent of certainity. 

![alt text][image17]

For the wrong prediction, the below image shows the 3 wrong prediction pictures. We can see that the 2 out of 3 prediction are quite close to the actual image. Category 27 is close to categroy 28. Category 28 is close to category 29.
## **However, it is not expected that 100km/h is predicted as 80km/h.** 
 
![alt text][image18]

The actual manually classified image from the validation dataset looks like:

![alt text][image19] 

From the above two images, we can see that the input 100km/h image quality is even better than the validation image. Why it is predicted as 80km/h? The top 5 probablity for this image is [0.9352   0.0428   0.0219  0.00007 0.00002]. It is quite close to 1 already. In order to find 
out why, i display the feature map of the convoltion layer 1 of the training model architecture, and it is shown below:

![alt text][image20]

However, it is not really very meaningful as compare to the feature map of the general caution which is a triangular sign(shown below).

![alt text][image21]

## **TODO: I need to figure out what are the actual kernal filter for the 24 kernals choose in the 1st convolution layer of the training model. So to understand why the 100km is predicted as 80km.**
## **Please give some hints on this.** 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

the feature map for convolution layer 1 of image 'go ahead or right'

![alt text][image23]
