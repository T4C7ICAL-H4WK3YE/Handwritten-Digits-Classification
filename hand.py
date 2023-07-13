import os
import cv2 #"Computer Vision" - To load and process images of data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import sklearn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

hidden = [100, 150] #Number of neurons in hidden layer
Accuracy = []
Max_Accuracy_1 = 0 #Initializing Max_Accuracy for 2 hidden layers
Max_Model_1 = '' #Name of model having maximum accuracy for 2 hidden layers
Max_Accuracy_2 = 0 #Initializing Max_Accuracy for 3 hidden layers
Max_Model_2 = '' #Name of model having maximum accuracy for 2 hidden layers

mnist = tf.keras.datasets.mnist #Getting the MNIST Dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data() #Loads MNIST data into 2 tuples (train and test) [Numpy arrays]

# Combine the training and testing datasets
images = list(x_train) + list(x_test)
labels = list(y_train) + list(y_test)

#Converting list back to numpy array
images = np.array(images)
labels = np.array(labels)

#We have excluded the commented part below from our code:
# Shuffle the images and labels
#random.seed(42)  # Set a fixed seed for reproducibility
#random.shuffle(images)
#random.shuffle(labels)

#Code continues from here:

for i in range(10): #For 10 random training and testing splits of the data. Also number of hidden layers = 2
  x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)

  x_train = tf.keras.utils.normalize(x_train, axis=1)
  x_test = tf.keras.utils.normalize(x_test, axis=1)

  model = tf.keras.models.Sequential() #Creates a basic Sequential Neural Network model

  for x in hidden:

    for y in ['tanh', 'sigmoid', 'relu']:

      model.add(tf.keras.layers.Flatten(input_shape = (28,28))) #Input layer
      model.add(tf.keras.layers.Dense(x, activation = y))
      model.add(tf.keras.layers.Dense(x, activation = y))
      model.add(tf.keras.layers.Dense(10, activation = 'softmax')) #Output layer having 10 units (neurons) representing output = 1,..,10 respectively.
      #Softmax makes sure all the 10 output neurons add up to 1. Acts like confidence. Each of the 10 neurons has value from 0 to 1 (after normalization)
      #This signals how likely the image is a specific digit. Value here is like probability and the one with highest value is most likely the digit shown by image.
      #Softmax gives probability for each digit to be the right answer.

      model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']) #Compiling the model

      model.fit(x_train, y_train, epochs = 3) #Training the data. Also gives accuracy of fitting the training data to model.

      #epochs is how many iterations we are going to see. Basically like how many times the model will see the data all over again (iterates).
      #fits the model with training data over and over again randomly (random 67% train) with each epoch.

      model.save('handwritten_digits.model')

      model = tf.keras.models.load_model('handwritten_digits.model')

      loss, accuracy = model.evaluate(x_test, y_test) #Evaluating model created with training data with testing data
      
      Accuracy.append(accuracy)

      #For max accuracy model
      if accuracy > Max_Accuracy_1:
        Max_Accuracy_1 = accuracy
        Max_Model_1 = y + 'model with' + str(x) + ' neurons in each layer'

      print(f'loss = {loss}') #lesser the loss function, the better. higher the accuracy, the better
      print(f'Accuracy = {accuracy*100}')

for i in range(10): #For 10 random training and testing splits of the data. Also number of hidden layers = 3
  x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)

  x_train = tf.keras.utils.normalize(x_train, axis=1)
  x_test = tf.keras.utils.normalize(x_test, axis=1)

  
  # Define the true labels and predicted labels as lists
  #true_labels = y_test
  #predicted_labels = 

  # Calculate the confusion matrix using the confusion_matrix() function
  #cm = confusion_matrix(true_labels, predicted_labels)

  model = tf.keras.models.Sequential() #Creates a basic Sequential Neural Network model

  for x in hidden:

    for y in ['tanh', 'sigmoid', 'relu']:

      model.add(tf.keras.layers.Flatten(input_shape = (28,28))) #Input layer
      model.add(tf.keras.layers.Dense(x, activation = y))
      model.add(tf.keras.layers.Dense(x, activation = y))
      model.add(tf.keras.layers.Dense(x, activation = y))
      model.add(tf.keras.layers.Dense(10, activation = 'softmax')) #Output layer having 10 units (neurons) representing output = 1,..,10 respectively.
      #Softmax makes sure all the 10 output neurons add up to 1. Acts like confidence. Each of the 10 neurons has value from 0 to 1 (after normalization)
      #This signals how likely the image is a specific digit. Value here is like probability and the one with highest value is most likely the digit shown by image.
      #Softmax gives probability for each digit to be the right answer.

      model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']) #Compiling the model

      model.fit(x_train, y_train, epochs = 3) #Training the data. Also gives accuracy of fitting the training data to model.

      #epochs is how many iterations we are going to see. Basically like how many times the model will see the data all over again (iterates).
      #fits the model with training data over and over again randomly (random 67% train) with each epoch.

      model.save('handwritten_digits.model')

      model = tf.keras.models.load_model('handwritten_digits.model')

      loss, accuracy = model.evaluate(x_test, y_test) #Evaluating model created with training data with testing data

      Accuracy.append(accuracy)

      #For max accuracy model
      if accuracy > Max_Accuracy_2:
        Max_Accuracy_2 = accuracy
        Max_Model_2 = y + 'model with' + str(x) + ' neurons in each layer'

      print(f'loss = {loss}') #lesser the loss function, the better. higher the accuracy, the better
      print(f'Accuracy = {accuracy*100}')

#To find out which model has highest Accuracy
if Max_Accuracy_1 > Max_Accuracy_2:
  print('The model with the highest accuracy is the ' + Max_Model_1 + ' with 2 hidden layers.')
elif Max_Accuracy_2 > Max_Accuracy_1:
  print('The model with the highest accuracy is the ' + Max_Model_2 + ' with 3 hidden layers.')

# To calculate average and variance of performance metrics.

variance = np.var(Accuracy)
Avg_Accuracy = np.mean(Accuracy)*100

print(f'Average accuracy = {Avg_Accuracy}')
print(f'Variance = {variance}')
