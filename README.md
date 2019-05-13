# MachineLearning

The code is used in my linkedin article : 
https://www.linkedin.com/pulse/how-different-conventional-programming-machine-learning-sharma/


## Description:
The project compares the two problem solving approaches to a computing problem. The problem requires to get the output as ‘fizz’ in case a given number is divisible by, output as ‘buzz’ if the given number is divisible by and output as ‘fizzbuzz’ if the  given number is divisible by both of them. 

First approach solves 16 the problem logically. The program code verifies the given number is divisible by 3 or 5 or both by executing a division operation.  

Second approach is a Machine Learning driven approach wherein a mathematical model is trained on a set of numbers whose output is known. Then the trained model is used to predict the outcomes of unknown numbers


### 1.	Key Concepts
	
#### 1.1. 	Machine Learning
Machine Learning is a method of data analysis which allows computer systems
to learn from previously available data whose results are known. And then using
that	data,	the	prediction	is	made	for	unseen	values	in	the	future.
Mathematically, it is about prediction of a variable(s)’ as a function of past
variable(s).

#### 1.2	Neural Networks
The  technique,  inspired  by the  functioning  of  the brain,  is  used  by systems
which can learn by themselves from the previous data and become capable of
taking decisions  on their  own without much human intervention. It  is  heavily
used in Artificial Intelligence and Deep  learning.  It consists  of neural layers,
which in turn consists of neural nodes.
A visual representation of neural can be seen as follows:  

>> Insert Image
Image 1: Neural Network

#### 1.3 Tensor Flow

It is an open source framework from Google which provides high level of APIs
for   ML  computation  and   training  models.   It   uses   data  flow   graphs for
computations. 


#### 2. Demo Code
It consists of two approaches: Normal Code & Machine Learning code

##### Normal Code 
It	is	simple	logical programming	implementation.	In this
implementation, python code is written to execute divisibility of the given
number by diving it by 3 and 5. And based on the outcome, output is given i.e.
if the number doesn’t leave any reminder when divisible by 3 and 5 then output
is  given as  ‘fizzbuzz’, if it  is  divisible by 3  then output is  given ‘fizz’ and  if
output is not divisible by 5 then the output is given as ‘buzz’.

##### Machine Learning Code
It is solved the problem using Machine Learning technique of neural
networks. The solution consists of three neural layers – input layer, hidden layer
and output layers. In this solution there is only one layer, however there can be
many layers and underlying neural nodes depending upon the complexity of the
problem.
Here is the visual representation of the solution:

![alt text](https://github.com/sharmaavneesh/MachineLearning/blob/master/NNDemo.png)

Image 2: The Visual representation of the solution 

Some of the key terms are explained below:
###### 2.2.1 Training Data
The correct data, which is already available, is divided into multiple portions. Out
of which one the parts is called the Training Data. It  is used to feed to machine
learning algorithm for training the model i.e. to make the model learn from past
experiences. In this case a set of 900 rows is used to feed which is a list of number
and its corresponding fizz, buzz or ‘fizzbuzz’ value. 

###### 2.2.2 Testing Data
The second part of the data is called Testing Data. It is used to test the model
which is being already training on the training data. The test data tells the
reliability of the trained model. In the solution 100 rows are used to test the data. 

###### 2.2.3 Validation Data
Usually some of the data is taken out from the testing data  which is further used to
validate the model. Sometimes it is also used to finetune the model further before
deploying it into the actual application. 

###### 2.2.4 Input Neural Layer
This layer is primarily for getting the input into the neural model. The number
of neural nodes is decided by the number of data points for input. For out
solution, there are 10 neural nodes. All the numbers are converted into binary
before being fed to the neural network. As the maximum number is 1000
therefore only 10 nodes are needed because less 10-digit binary number can
sufficiently accommodate the input.

###### 2.2.5  Hidden Layer
Hidden layers are inside layers where the most complex computations happen.
In this solution there is a hidden layer of 100 neural nodes. This means there
will be 10(input layer) * 100(hidden layer) connections will be made.

###### 2.2.6 Output Layer
Output layer provides the final outcome of the computations. In this solution
there are four neural nodes because there are four categories for the prediction.

###### 2.2.7 Weights
It is the value corresponding to each connection. In the forward propagation the
weights are multiplied to the input layers neural node for further neural node in
hidden layers. The number of weights of significant variables is higher as
compared to the non-significant ones.

###### 2.2.8 Activaation Layer
Activation functions are applied to the nodes to provide the actual output. In
the solution, rectified activation function is applied to the hidden.

###### 2.2.9 Error Function
Error function calculates the error in the predicted value and the expected value.
The solution uses the mean of the individual outputs in the output layer.

###### 2.2.10 Optimizer Function
Optimizer functions optimize the weights to a point where the value of error
functions is minimum. At this point the prediction rate is greatest for the model.
In the solution the Gradient Descent Optimizer.

###### 2.2.11 Epochs
This is the number of iterations of all the training data is fed into the model. In
the solution 5000 epochs is being used.

###### 2.2.12 Batches
At a time, whole data is not fed into the system, rather it is fed into the batches.
In the solution 128 data points are there in a batch.

###### 3 Results
![alt text](https://github.com/sharmaavneesh/MachineLearning/blob/master/NN%20Visual.png)

![alt text](https://github.com/sharmaavneesh/MachineLearning/blob/master/NN%20Results.png)



 



























































