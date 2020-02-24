# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:09:57 2018

@author: ashley
"""

#help("modules")
#help("tensorflow")

#restrict to only using CPU  (might want to b/c:  a) I'm just curious about performance.  b) note online that maybe GPU has less memory than CPU, can't hold all the data?)
#technique 1:  set env variables to turn off GPU.   seems to be considered simpler and more holistic
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#technique #2:   configure your session.  cons:  a little more code maybe, needs to happen for each session??
    #  pros;   more flexible (could limit rather than turn off; apply when desired but not always, etc
    
#Alternatively, a probably more portable solution is to have Python itself set the environment variable before TensorFlow is imported by any module:
#
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#import tensorflow as tf
#
#
#
#Another user suggests creating your session in the following way:
#
#import tensorflow as tf
#
#session_conf = tf.ConfigProto(
#    device_count={'CPU' : 1, 'GPU' : 0},
#    allow_soft_placement=True,
#    log_device_placement=False
#)
#
#with tf.Session(config=session_conf) as sess:
#    sess.run(...)
#This should allow you for more fine-grained control too (e.g. I have two GPUs but only want TensorFlow to use one of them).


#technique 3 - I think i've created a different conda environment with "normal" tensorflow and a conda environment with tf_gpu.
    
    

# Initial deep neural network set-up from 
# Géron, A. 2017. Hands-On Machine Learning with Scikit-Learn 
#    & TensorFlow: Concepts, Tools, and Techniques to Build 
#    Intelligent Systems. Sebastopol, Calif.: O'Reilly. 
#    [ISBN-13 978-1-491-96229-9] 
#    Source code available at https://github.com/ageron/handson-ml
#    See file 10_introduction_to_artificial_neural_networks.ipynb 
#    Revised from MNIST to Cats and Dogs to begin Assignment 7
#    #CatsDogs# comment lines show additions/revisions for Cats and Dogs

# To support both python 2 and python 3
#from __future__ import division, print_function, unicode_literals

# Common imports for our work
import numpy as np
import tensorflow as tf
import math


RANDOM_SEED = 9999

# To make output stable across runs
def reset_graph(seed= RANDOM_SEED):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

#dimensions of the image (in pixels I believe)
img_height = 64
img_width = 64   
    
#CatsDogs# 
# Documentation on npy binary format for saving numpy arrays for later use
#     https://towardsdatascience.com/
#             why-you-should-start-using-npy-file-more-often-df2a13cc0161
# Under the working directory, data files are in directory cats_dogs_64_128 
# Read in cats and dogs grayscale 64x64 files to create training data

wkg_dir = 'C:/Users/ashle/Documents/Personal Data/Northwestern/2018_4 FALL  PREDICT 422 Machine Learning/wk7 - deep learning for image classification/image_data__cats_dogs_64-128'
array_cats = np.load(wkg_dir + '/cats_1000_64_64_1.npy')
array_dogs = np.load(wkg_dir + '/dogs_1000_64_64_1.npy')
    #presumably 1000 ==>   array of 1000 images
    #           64_64_1  =>  64 pixels X 64 pixels by 1 color channel (ie. grayscale)



array_cats.shape

#first dim (grayscale channel) 
array_cats[:,0,0,0].shape
array_cats[:,1,1,0].shape
  #0
np.max(array_cats[:,:,:,:])  #0
np.max(array_cats[0])  #234

np.min(array_cats[0])  #2
np.max(array_cats[0])  #234

#last dim (grayscale channel) 
np.min(array_cats[:,:,:,0])  #0
np.max(array_cats[:,:,:,0])  #255


#last dim (grayscale channel) 
np.min(array_cats[:,:,:,0])  #0
np.max(array_cats[:,:,:,0])  #255

from matplotlib import pyplot as plt  # for display of images
def show_grayscale_image(image_dtls):
    plt.imshow(image_dtls, cmap='gray')
    plt.axis('off')
    plt.show()
# Examine first cat and first dog grayscale images
show_grayscale_image(array_cats[0,:,:,0])
show_grayscale_image(array_dogs[0,:,:,0])



# 300 and 100 nodes for layers 1 and 2 as used with MNIST from Geron
NodeCount__HiddenLyr1 = 300
NodeCount__HiddenLyr2 = 100

channels = 1  # When working with color images use channels = 3

n_inputs = img_height * img_width

#CatsDogs# Has two output values # MNIST had ten digits n_outputs = 10  
n_outputs = 2  # binary classification for Cats and Dogs, 1 output node 0/1

reset_graph()

# dnn... Deep neural network model from Geron Chapter 10
# Note that this model makes no use of the fact that we have
# pixel data arranged in rows and columns
# So a 64x64 matrix of raster values becomes a vector of 4096 input variables

#construct the graph
#===================
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z

with tf.name_scope("dnn"):
    hidden_layer1 = neuron_layer(X, NodeCount__HiddenLyr1, name="hidden1",
                           activation=tf.nn.relu)
    hidden_layer2 = neuron_layer(hidden_layer1, NodeCount__HiddenLyr2, name="hidden2",
                           activation=tf.nn.relu)
    logit_layer = neuron_layer(hidden_layer2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logit_layer)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope("eval"):
    correct_predn = tf.nn.in_top_k(logit_layer, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_predn, tf.float32))    

init = tf.global_variables_initializer()
saver = tf.train.Saver()



# Work the data for cats and dogs numpy arrays 
# These numpy arrays were generated in previous data prep work
# Stack the numpy arrays for the inputs
X_cat_dog = np.concatenate((array_cats, array_dogs), axis = 0) 
X_cat_dog = X_cat_dog.reshape(-1,img_width*img_height) # note coversion to 4096 inputs

X_cat_dog.shape  
    #now we have 2000 items/images (1000 cats, 1000 dogs) 
         #each item/image is a list of 4096 elements (4096 pixels, arranged in a 1D vector), 
             #each element/pixel  is a value between 0 and 255 indicating the grayscale value


# Scikit Learn for min-max scaling of the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#fit a standard scaler for values between 0 and 255
scaler.fit(np.array([0., 255.]).reshape(-1,1))
#transform the greyscal value of the images using that scaler
X_cat_dog_min_max = scaler.transform(X_cat_dog)



# Define the labels to be used.  the 1000 cat images are first; label them as = 0
    #the 1000 dog images are next; label them as 1
y_cat_dog = np.concatenate((np.zeros((1000), dtype = np.int32), 
                      np.ones((1000), dtype = np.int32)), axis = 0)



# Scikit Learn for random splitting of the data  
from sklearn.model_selection import train_test_split

# Random splitting of the data in to training (80%) and test (20%)  
X_train, X_test, y_train, y_test = \
    train_test_split(X_cat_dog_min_max, y_cat_dog, test_size=0.20, 
                     random_state = RANDOM_SEED)

init = tf.global_variables_initializer()    

n_epochs = 50
batch_size = 100
nbr_batches = math.ceil(y_train.shape[0] // batch_size)


#execute the graph
from time import time

t_start = time()
print('Starting graph construction:  ', t_start)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iter_nbr in range(nbr_batches):
            X_batch = X_train[iter_nbr*batch_size:(iter_nbr + 1)*batch_size,:]
            y_batch = y_train[iter_nbr*batch_size:(iter_nbr + 1)*batch_size]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        #evaluate the accuracy of the latest batch of training data, as well as the accuracy of the entire set of test data
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        save_path = saver.save(sess, "./my_catdog_model")


t_end = time()
print('Ending graph construction:  ', t_end)
print('Elapsed time:  ', t_end - t_start)



with tf.Session() as sess2:
    saver.restore(sess2, "./my_catdog_model")  #this appears to put it in the current wkg dir
    Z = logit_layer.eval(feed_dict={X: X_test})


    logit_lyr_rslts__test = logit_layer.eval(feed_dict={X: X_test})
    #use the softmax function to change them results to probabilities of being in each class
    #    so we now have the same # of columns but they range from 0 to 1 and sum across columns to 1 b/c the cols in a row are probabilities of being that class
    prob_preds__test = tf.nn.softmax(logit_lyr_rslts__test).eval()
    #the predicted class is the field (0 ==> cats, 1 ==> dogs) with the higher probability.  Logit columns will give the same conclusion, just based on a diff scale/unit that is not a probability
    final_class_predn__test = tf.argmax(prob_preds__test, axis=1).eval()
    
    #type(logit_lyr_rslts)  #array
    #logit_lyr_rslts.shape   #400, 2.   
        #pretty sure this is one row per image/record in the input set (in this case, the test set)
        # and one column per output class
        
    #logit_lyr_rslts[0:5,:]   
    #np.argmax(logit_lyr_rslts[0:5], axis=1)
    #the most likely class has the highest value in the corresponding column.,
    #   In this case, since cats were labeled 0, column 0 is higher if the prediction is cat; 
    #       column1 is higher if the predn is dog
    #type(final_class_predn__test)   #numpy.ndarray
    #final_class_predn__test.shape  #(400,)
    #final_class_predn__test[0:5]


    # look at the correct predictions, just to make sure my logic is correct
    # ----------------------------------------------------------------------
    #were the predns for these images correct yes/no?
    correct_predn__test = final_class_predn__test == y_test


    #filter down to an array of the images that were correctly categorized
    imgs_correct_predn__test = X_test[correct_predn__test, :]
    #get the predictions (== labels, as these are all correct predns)
    #predns_correct_predn__test = final_class_predn__test[correct_predn__test]

    ##208 were predicted correctly
    print(imgs_correct_predn__test.shape)
    #predns_correct_predn__test.shape

    ##look at them - is my logic working as desired?
    #for img_nbr in range(predns_correct_predn__test.shape[0]):
    #    print(predns_correct_predn__test[img_nbr])
    #    show_grayscale_image(imgs_correct_predn__test[img_nbr,:,:,0])




#review the confusion matrix
with tf.Session() as sess2:
    confusn_mtrx = tf.confusion_matrix(labels=y_test, predictions=final_class_predn__test, num_classes=2, name="confusn_matrix").eval()

print(confusn_mtrx)































type(Z)  #array
Z.shape   #400, 2.   
    #pretty sure this is one row per image/record in the input set (in this case, the test set)
    # and one column per output class
    
Z[0:5,:]   
np.argmax(Z[0:5], axis=1)
#the most likely class has the highest value in the corresponding column.,
#   In this case, since cats were labeled 0, column 0 is higher if the prediction is cat; 
#       column1 is higher if the predn is dog

#use the softmax function to change them results to probabilities of being in each class
#    so we now have the same # of columns but they range from 0 to 1 and sum across columns to 1 b/c the cols in a row are probabilities of being that class
with tf.Session() as sess3:
    preds = tf.nn.softmax(Z).eval()

 
preds[0:10]
preds[0:10, 0]

preds[0:20].shape
type(preds)

#check the ranges - are they within 0 to 1 as expected?
np.max(preds[:,0])  #0.958386
np.min(preds[:,0])  #0.05    

np.max(preds[:,1])  #0.946
np.min(preds[:,1])  #0.042    

np.sum(preds, axis=1)
    #gives a 1D array of 400 obs.  They are all 1 - except a few within machine rounding of that??
    

    



softmax_rslts = tf.nn.softmax(logit_layer)


#os.getcwd()
#'C:\\Users\\ashle'


