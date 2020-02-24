# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:09:57 2018

@author: Ashley Wenger
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

print(tf.__version__)   #1.12.0



RANDOM_SEED = 9999

# To make output stable across runs
def reset_graph(seed= RANDOM_SEED):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    
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



#array_cats.shape
#
##first dim (grayscale channel) 
#array_cats[:,0,0,0].shape
#array_cats[:,1,1,0].shape
#  #0
#np.max(array_cats[:,:,:,:])  #0
#np.max(array_cats[0])  #234
#
#np.min(array_cats[0])  #2
#np.max(array_cats[0])  #234
#
##last dim (grayscale channel) 
#np.min(array_cats[:,:,:,0])  #0
#np.max(array_cats[:,:,:,0])  #255
#
#
##last dim (grayscale channel) 
#np.min(array_cats[:,:,:,0])  #0
#np.max(array_cats[:,:,:,0])  #255

from matplotlib import pyplot as plt  # for display of images
def show_grayscale_image(image_dtls):
    plt.imshow(image_dtls, cmap='gray')
    plt.axis('off')
    plt.show()
# Examine first cat and first dog grayscale images
show_grayscale_image(array_cats[456,:,:,0])
show_grayscale_image(array_dogs[234,:,:,0])




# split the cats, and then split the dogs
# =======================================
from sklearn.model_selection import train_test_split

# Define the labels to be used.  the 1000 cat images are first; label them as = 0
    #the 1000 dog images are next; label them as 1
y_cats = np.zeros(1000, dtype = np.int32)
y_dogs = np.ones(1000, dtype = np.int32)


# Cats:  random splitting of the cats data  into training (80%) and test (20%)  
# ----------------------------------------------------------------------------
X_train_cats, X_test_cats, y_train_cats, y_test_cats = \
                train_test_split(array_cats, y_cats, test_size=0.20, 
                                 random_state = RANDOM_SEED)

#split train again to get an eval set of records
X_train_cats, X_eval_cats, y_train_cats, y_eval_cats = \
    train_test_split(X_train_cats, y_train_cats, test_size=0.125,  
                     random_state = RANDOM_SEED)


# Dogs:  random splitting of the dogs data  into training (80%) and test (20%)  
# ----------------------------------------------------------------------------
X_train_dogs, X_test_dogs, y_train_dogs, y_test_dogs = \
                train_test_split(array_dogs, y_dogs, test_size=0.20, 
                                 random_state = RANDOM_SEED)

#split train again to get an eval set of records
X_train_dogs, X_eval_dogs, y_train_dogs, y_eval_dogs = \
    train_test_split(X_train_dogs, y_train_dogs, test_size=0.125,  
                     random_state = RANDOM_SEED)

#X_train_cats.shape  #(700, 64, 64, 1)          #y_train_cats.shape  #(700,)
#X_eval_cats.shape   #(100, 64, 64, 1)          #y_eval_cats.shape   #(100,)
#X_test_cats.shape   #(200, 64, 64, 1)          #y_test_cats.shape   #(200,)




#concatenate cats_train and dogs_train;   cats_eval and dogs_eval;   cats_test and dogs_test 
# =========================================================================================
X_train = np.concatenate( (X_train_cats, X_train_dogs), axis=0)
y_train = np.concatenate( (y_train_cats, y_train_dogs), axis=0)

X_eval = np.concatenate( (X_eval_cats, X_eval_dogs), axis=0)
y_eval = np.concatenate( (y_eval_cats, y_eval_dogs), axis=0)

X_test = np.concatenate( (X_test_cats, X_test_dogs), axis=0)
y_test = np.concatenate( (y_test_cats, y_test_dogs), axis=0)

#X_train.shape  #(1400, 64, 64, 1)     #y_train.shape  #(1400,)
#X_eval.shape  #(200, 64, 64, 1)       #y_eval.shape  #(200,)
#X_test.shape  #(400, 64, 64, 1)       #y_test.shape  #(400,)


#shuffle them up so not all cats are first and all dogs are last
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)
X_eval, y_eval = shuffle(X_eval, y_eval)
X_test, y_test = shuffle(X_test, y_test)


# scale the greyscale value to be between 0 and 1
# ===============================================
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#fit a min-max scaler for values between 0 and 255
scaler.fit(np.array([0., 255.]).reshape(-1,1))

#transform the greyscal value of the images using that scaler, without changing it into one long string
#    - the 2nd (ie, [1] dimension/axis of the array is the height or width dimension;  the 1st, ie [0] is the image #/ID
#    - loop over each height and width cell, and get the collection of greyscale values (4th dim value) for that cell
#      across the set of images ( use ":" ) for the first dimension filter
for axis1_val in range(X_train.shape[1]):
    for axis2_val in range(X_train.shape[2]):
            X_train[:, axis1_val, axis2_val, 0] = scaler.transform(X_train[:, axis1_val, axis2_val, 0].reshape(-1, 1)).reshape(1, -1)
            X_eval[:, axis1_val, axis2_val, 0] = scaler.transform(X_eval[:, axis1_val, axis2_val, 0].reshape(-1, 1)).reshape(1, -1)
            X_test[:, axis1_val, axis2_val, 0] = scaler.transform(X_test[:, axis1_val, axis2_val, 0].reshape(-1, 1)).reshape(1, -1)
            






#define functions to construct the graph and execute it
#======================================================



# construct the graph
# -------------------            
#set some basic params that won't be changing
#dimensions of the image (in pixels I believe)
img_height = 64
img_width = 64   
n_channels = 1  # When working with color images use n_channels = 3

#CatsDogs# Has two output values # MNIST had ten digits n_outputs = 10  
n_outputs = 2  # binary classification for Cats and Dogs, 1 output node 0/1



def Construct_Graph_Layers():
    global X
    #input layer is placeholders that will be fed with values from each batch when the graph is executed
    X = tf.placeholder(tf.float32, shape=(None, img_height, img_width, n_channels), name="X")

    #layer 1
    #try a convolution layer
    lyr2 = tf.layers.conv2d(X, filters=2, kernel_size=3, strides=[3,3], padding="SAME", name="Hidden1")

    #layer2
    #now try adding a pooling layer
    lyr3 = tf.nn.max_pool(lyr2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="Hidden2")

    #layer 3
    #try a convolution layer
    lyr4 = tf.layers.conv2d(lyr3, filters=2, kernel_size=3, strides=[3,3], padding="SAME", name="Hidden3")

    #create an output layer
    global logit_lyr
    logit_lyr = tf.layers.dense(tf.layers.Flatten()(lyr4), n_outputs, name="outputs")



#with tf.name_scope('dnn'):
#    hidden_layer1 = tf.layers.dense(X, n_hidden1, name='hidden1', activation=tf.nn.relu)

#lyr2.shape
#lyr3.shape
#lyr4.shape
#lyr4.set_shape([1, 16])
#
#logit_lyr.shape

from datetime import datetime

#set up computations to calculate (and then minimize) the cost function
#======================================================================
def Construct_Graph_Ops():
    global y
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit_lyr)
        loss_var = tf.reduce_mean(xentropy, name='loss')
        
    
    with tf.name_scope('train'):
        optmzr = tf.train.GradientDescentOptimizer(learning_rate)
        global training_op
        training_op = optmzr.minimize(loss_var)
        
    
    #model evaluation
    with tf.name_scope('eval_ops'):
        #    correct_preds = tf.nn.in_top_k(logit_layer, y, 1)
        predns = tf.argmax(logit_lyr, axis=1)  
        img_properly_ctgrzd = tf.equal(predns, tf.cast(y, tf.int64))
        global mdl_accuracy
        mdl_accuracy = tf.reduce_mean(tf.cast(img_properly_ctgrzd, tf.float16))    
    
    #initialize the variables    
    global init
    init = tf.global_variables_initializer()
    #set up a saver to save results at various checkpoints
    global saver
    saver = tf.train.Saver()

    #set up logging so we can view the graph with TensorBoard
    curr_time = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = 'C:/Users/ashle/Documents/Personal Data/Northwestern/2018_4 FALL  PREDICT 422 Machine Learning/wk7 - deep learning for image classification'
    log_dir = "{}/run-{}/".format(root_logdir, curr_time)

    #create a node in the graph to write the mdl_accuracy out
    global summry_node__mdl_accuracy
    summry_node__mdl_accuracy = tf.summary.scalar('Mdl_Accuracy', mdl_accuracy)
    global logfl_wrtr
    logfl_wrtr = tf.summary.FileWriter(log_dir, tf.get_default_graph())    
    

#execute the graph's training operation
def Execute_Graph__TrainingOp(nbr_epochs, nbr_batches):
    with tf.Session() as sess:
        init.run()
        for epoch in range(nbr_epochs):
            for iter_nbr in range(nbr_batches):
                #get the indices of the images to be processed in this batch
                batch_start_num = iter_nbr*batch_size
                batch_end_num = (iter_nbr + 1)*batch_size
                #get the chunk of data we will be processing in this batch
                X_batch = X_train[batch_start_num:batch_end_num,:]
                y_batch = y_train[batch_start_num:batch_end_num]
                #
                #train on this new batch of data
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                #
                #evaluate the accuracy of the latest batch of training data, as well as the accuracy of the entire set of test data
                acc_train = mdl_accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                acc_eval = mdl_accuracy.eval(feed_dict={X: X_eval, y: y_eval})
                print(epoch, "Accuracy on current batch of training data:", acc_train, "Evaluation dataset accuracy:", acc_eval)
                #
            #write some progress to the log file, after every epoch
            smry__acc_eval = summry_node__mdl_accuracy.eval(feed_dict={X: X_eval, y: y_eval})
            logfl_wrtr.add_summary(smry__acc_eval, epoch)
            #save the results to a checkpoint file 
            wkg_dir = 'C:/Users/ashle/Documents/Personal Data/Northwestern/2018_4 FALL  PREDICT 422 Machine Learning/wk7 - deep learning for image classification'
            saver.save(sess, wkg_dir+"/my_catdog_model")



#calculate the accuracy on the test set
def Execute_Graph__EvaluationOps(ResultDict, run_name):
    with tf.Session() as sess2:
        wkg_dir = 'C:/Users/ashle/Documents/Personal Data/Northwestern/2018_4 FALL  PREDICT 422 Machine Learning/wk7 - deep learning for image classification'
        saver.restore(sess2, wkg_dir+"/my_catdog_model")  #this appears to put it in the current wkg dir
        logit_lyr_rslts__test = logit_lyr.eval(feed_dict={X: X_test})
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
    imgs_correct_predn__test = X_test[correct_predn__test, :, :, :]
    #get the predictions (== labels, as these are all correct predns)
    #predns_correct_predn__test = final_class_predn__test[correct_predn__test]

    ##208 were predicted correctly
    #imgs_correct_predn__test.shape
    #predns_correct_predn__test.shape

    ##look at them - is my logic working as desired?
    #for img_nbr in range(predns_correct_predn__test.shape[0]):
    #    print(predns_correct_predn__test[img_nbr])
    #    show_grayscale_image(imgs_correct_predn__test[img_nbr,:,:,0])


    #review the confusion matrix
    with tf.Session() as sess2:
        confusn_mtrx = tf.confusion_matrix(labels=y_test, predictions=final_class_predn__test, num_classes=2, name="confusn_matrix").eval()
 
    #print(confusn_mtrx)
    #[[104  96]
    # [ 88 112]]    
    ResultDict.update({run_name:
                            {'nbr_correct_predns': imgs_correct_predn__test.shape[0],
                             'confusn_mtrx': confusn_mtrx.copy()}
                      })    



    
    
    

# =================================================================================================================    
# use the functions to run various experiments, with different hyperparameter settings, diff network topology, etc.
# =================================================================================================================    
    

from time import time
    
#store the results for various experiments (diff hyperparam settings, diff layers,etc)  in a dict    
Result_Dict = dict()
    

reset_graph()




# ========================================================
#   try a graph with 4 layers;  2 pairs of (conv + pooling)
#                don't specify an activation function ==> it uses none ==>  final output = wtd sum of inputs
# ========================================================



#define this up front so it can be used in layer lists
  #  !!  need to recreate if we reset the graph!
X = tf.placeholder(tf.float32, shape=(None, img_height, img_width, n_channels), name="X")
lyr2 = tf.layers.conv2d(X, filters=2, kernel_size=3, strides=[3,3], padding="SAME", name="Hidden1")
lyr3 = tf.nn.max_pool(lyr2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="Hidden2")
lyr4 = tf.layers.conv2d(lyr3, filters=2, kernel_size=3, strides=[3,3], padding="SAME", name="Hidden3")
logit_lyr = tf.layers.dense(tf.layers.Flatten()(lyr4), n_outputs, name="outputs")

#Construct_Graph_Layers()



learning_rate = 0.01    
Construct_Graph_Ops()
        

# execute the graph
# =================
n_epochs = 50
batch_size = 100
nbr_batchez = math.ceil(y_train.shape[0] // batch_size)


t_start = time()
print('Starting graph construction:  ', t_start)
Execute_Graph__TrainingOp(n_epochs, nbr_batchez)
t_end = time()
print('Ending graph construction:  ', t_end)
print('Elapsed time:  ', t_end - t_start)
      
Execute_Graph__EvaluationOps(Result_Dict, 'four layer')

Result_Dict['four layer'].update({'run_time':  t_end - t_start})

logfl_wrtr.close()




# ========================================================
#   try a graph with 4 layers;  2 pairs of (conv + pooling)  
#            this time use a ReLU activation function
# ========================================================
reset_graph()



#define this up front so it can be used in layer lists
  #  !!  need to recreate if we reset the graph!
X = tf.placeholder(tf.float32, shape=(None, img_height, img_width, n_channels), name="X")
lyr2 = tf.layers.conv2d(X, filters=2, kernel_size=3, strides=[3,3], padding="SAME", activation=tf.nn.relu, name="Hidden1")
lyr3 = tf.nn.max_pool(lyr2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="Hidden2")
lyr4 = tf.layers.conv2d(lyr3, filters=2, kernel_size=3, strides=[3,3], padding="SAME", activation=tf.nn.relu, name="Hidden3")
logit_lyr = tf.layers.dense(tf.layers.Flatten()(lyr4), n_outputs, name="outputs")

#Construct_Graph_Layers()



learning_rate = 0.01    
Construct_Graph_Ops()
        

# execute the graph
# =================
n_epochs = 50
batch_size = 100
nbr_batchez = math.ceil(y_train.shape[0] // batch_size)


t_start = time()
print('Starting graph construction:  ', t_start)
Execute_Graph__TrainingOp(n_epochs, nbr_batchez)
t_end = time()
print('Ending graph construction:  ', t_end)
print('Elapsed time:  ', t_end - t_start)
      
Execute_Graph__EvaluationOps(Result_Dict, 'four layer_w_relu')

Result_Dict['four layer_w_relu'].update({'run_time':  t_end - t_start})

logfl_wrtr.close()



# ========================================================
#   try a different graph  -- 7 layers to shrink the spatial dimension all the way down to 1x1
# ========================================================

reset_graph()




#define this up front so it can be used in layer lists
  #  !!  need to recreate if we reset the graph!
X = tf.placeholder(tf.float32, shape=(None, img_height, img_width, n_channels), name="X")
lyr2 = tf.layers.conv2d(X, filters=2, kernel_size=2, strides=[2,2], padding="SAME", name="Hidden1")
lyr3 = tf.nn.max_pool(lyr2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="Hidden2")
lyr4 = tf.layers.conv2d(lyr3, filters=2, kernel_size=2, strides=[2,2], padding="SAME", name="Hidden3")

lyr5 = tf.layers.conv2d(lyr4, filters=2, kernel_size=2, strides=[2,2], padding="SAME", name="Hidden4")
lyr6 = tf.nn.max_pool(lyr5, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="Hidden5")
lyr7 = tf.layers.conv2d(lyr6, filters=2, kernel_size=2, strides=[2,2], padding="SAME", name="Hidden6")

logit_lyr = tf.layers.dense(tf.layers.Flatten()(lyr7), n_outputs, name="outputs")

#Construct_Graph_Layers()

#lyr2.shape  #32x32
#lyr3.shape  #16x16
#lyr4.shape  #8x8
#lyr5.shape  #4x4
#lyr6.shape  #2x2
#lyr7.shape  #1x1

learning_rate = 0.01    
Construct_Graph_Ops()
        

# execute the graph
# =================
n_epochs = 50
batch_size = 100
nbr_batchez = math.ceil(y_train.shape[0] // batch_size)


t_start = time()
print('Starting graph construction:  ', t_start)
Execute_Graph__TrainingOp(n_epochs, nbr_batchez)
t_end = time()
print('Ending graph construction:  ', t_end)
print('Elapsed time:  ', t_end - t_start)
      
Execute_Graph__EvaluationOps(Result_Dict, 'seven layer')

Result_Dict['seven layer'].update({'run_time':  t_end - t_start})

logfl_wrtr.close()






# ========================================================
#   try a different graph  -- 7 layers to shrink the spatial dimension all the way down to 1x1
#            this time use a ReLU activation function
# ========================================================

reset_graph()




#define this up front so it can be used in layer lists
  #  !!  need to recreate if we reset the graph!
X = tf.placeholder(tf.float32, shape=(None, img_height, img_width, n_channels), name="X")
lyr2 = tf.layers.conv2d(X, filters=2, kernel_size=2, strides=[2,2], padding="SAME", activation=tf.nn.relu, name="Hidden1")
lyr3 = tf.nn.max_pool(lyr2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="Hidden2")
lyr4 = tf.layers.conv2d(lyr3, filters=2, kernel_size=2, strides=[2,2], padding="SAME", activation=tf.nn.relu, name="Hidden3")

lyr5 = tf.layers.conv2d(lyr4, filters=2, kernel_size=2, strides=[2,2], padding="SAME", activation=tf.nn.relu, name="Hidden4")
lyr6 = tf.nn.max_pool(lyr5, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="Hidden5")
lyr7 = tf.layers.conv2d(lyr6, filters=2, kernel_size=2, strides=[2,2], padding="SAME", activation=tf.nn.relu, name="Hidden6")

logit_lyr = tf.layers.dense(tf.layers.Flatten()(lyr7), n_outputs, name="outputs")

#Construct_Graph_Layers()

#lyr2.shape  #32x32
#lyr3.shape  #16x16
#lyr4.shape  #8x8
#lyr5.shape  #4x4
#lyr6.shape  #2x2
#lyr7.shape  #1x1

learning_rate = 0.01    
Construct_Graph_Ops()
        

# execute the graph
# =================
n_epochs = 50
batch_size = 100
nbr_batchez = math.ceil(y_train.shape[0] // batch_size)


t_start = time()
print('Starting graph construction:  ', t_start)
Execute_Graph__TrainingOp(n_epochs, nbr_batchez)
t_end = time()
print('Ending graph construction:  ', t_end)
print('Elapsed time:  ', t_end - t_start)
      
Execute_Graph__EvaluationOps(Result_Dict, 'seven layer_w_relu')

Result_Dict['seven layer_w_relu'].update({'run_time':  t_end - t_start})

logfl_wrtr.close()



# ========================================================
#   try a different graph - still seven layers, but with diff ordering of the pooling layers
# ========================================================

reset_graph()




#define this up front so it can be used in layer lists
  #  !!  need to recreate if we reset the graph!
X = tf.placeholder(tf.float32, shape=(None, img_height, img_width, n_channels), name="X")
lyr2 = tf.layers.conv2d(X, filters=2, kernel_size=2, strides=[2,2], padding="SAME", name="Hidden1")
lyr3 = tf.layers.conv2d(lyr2, filters=2, kernel_size=2, strides=[2,2], padding="SAME", name="Hidden3")
lyr4 = tf.nn.max_pool(lyr3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="Hidden2")

lyr5 = tf.layers.conv2d(lyr4, filters=2, kernel_size=2, strides=[2,2], padding="SAME", name="Hidden4")
lyr6 = tf.layers.conv2d(lyr5, filters=2, kernel_size=2, strides=[2,2], padding="SAME", name="Hidden6")
lyr7 = tf.nn.max_pool(lyr6, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="Hidden5")

logit_lyr = tf.layers.dense(tf.layers.Flatten()(lyr7), n_outputs, name="outputs")

#Construct_Graph_Layers()



learning_rate = 0.01    
Construct_Graph_Ops()
        

# execute the graph
# =================
n_epochs = 50
batch_size = 100
nbr_batchez = math.ceil(y_train.shape[0] // batch_size)


t_start = time()
print('Starting graph construction:  ', t_start)
Execute_Graph__TrainingOp(n_epochs, nbr_batchez)
t_end = time()
print('Ending graph construction:  ', t_end)
print('Elapsed time:  ', t_end - t_start)
      
Execute_Graph__EvaluationOps(Result_Dict, 'seven layer_reorged')

Result_Dict['seven layer_reorged'].update({'run_time':  t_end - t_start})

logfl_wrtr.close()






# ========================================================
#   try a different graph - still seven layers, but with diff ordering of the pooling layers
#            this time use a ReLU activation function
# ========================================================

reset_graph()




#define this up front so it can be used in layer lists
  #  !!  need to recreate if we reset the graph!
X = tf.placeholder(tf.float32, shape=(None, img_height, img_width, n_channels), name="X")
lyr2 = tf.layers.conv2d(X, filters=2, kernel_size=2, strides=[2,2], padding="SAME", activation=tf.nn.relu, name="Hidden1")
lyr3 = tf.layers.conv2d(lyr2, filters=2, kernel_size=2, strides=[2,2], padding="SAME", activation=tf.nn.relu, name="Hidden3")
lyr4 = tf.nn.max_pool(lyr3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="Hidden2")

lyr5 = tf.layers.conv2d(lyr4, filters=2, kernel_size=2, strides=[2,2], padding="SAME", activation=tf.nn.relu, name="Hidden4")
lyr6 = tf.layers.conv2d(lyr5, filters=2, kernel_size=2, strides=[2,2], padding="SAME", activation=tf.nn.relu, name="Hidden6")
lyr7 = tf.nn.max_pool(lyr6, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="Hidden5")

logit_lyr = tf.layers.dense(tf.layers.Flatten()(lyr7), n_outputs, name="outputs")

#Construct_Graph_Layers()



learning_rate = 0.01    
Construct_Graph_Ops()
        

# execute the graph
# =================
n_epochs = 50
batch_size = 100
nbr_batchez = math.ceil(y_train.shape[0] // batch_size)


t_start = time()
print('Starting graph construction:  ', t_start)
Execute_Graph__TrainingOp(n_epochs, nbr_batchez)
t_end = time()
print('Ending graph construction:  ', t_end)
print('Elapsed time:  ', t_end - t_start)
      
Execute_Graph__EvaluationOps(Result_Dict, 'seven layer_reorged_w_relu')

Result_Dict['seven layer_reorged_w_relu'].update({'run_time':  t_end - t_start})

logfl_wrtr.close()





































# try with diff filters

reset_graph()




#define this up front so it can be used in layer lists
  #  !!  need to recreate if we reset the graph!
X = tf.placeholder(tf.float32, shape=(None, img_height, img_width, n_channels), name="X")
lyr2 = tf.layers.conv2d(X, filters=4, kernel_size=2, strides=[2,2], padding="SAME", activation=tf.nn.relu, name="Hidden1")
lyr3 = tf.layers.conv2d(lyr2, filters=4, kernel_size=2, strides=[2,2], padding="SAME", activation=tf.nn.relu, name="Hidden3")
lyr4 = tf.nn.max_pool(lyr3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="Hidden2")

lyr5 = tf.layers.conv2d(lyr4, filters=3, kernel_size=2, strides=[2,2], padding="SAME", activation=tf.nn.relu, name="Hidden4")
lyr6 = tf.layers.conv2d(lyr5, filters=3, kernel_size=2, strides=[2,2], padding="SAME", activation=tf.nn.relu, name="Hidden6")
lyr7 = tf.nn.max_pool(lyr6, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="Hidden5")

logit_lyr = tf.layers.dense(tf.layers.Flatten()(lyr7), n_outputs, name="outputs")

#Construct_Graph_Layers()



learning_rate = 0.01    
Construct_Graph_Ops()
        

# execute the graph
# =================
n_epochs = 50
batch_size = 100
nbr_batchez = math.ceil(y_train.shape[0] // batch_size)


t_start = time()
print('Starting graph construction:  ', t_start)
Execute_Graph__TrainingOp(n_epochs, nbr_batchez)
t_end = time()
print('Ending graph construction:  ', t_end)
print('Elapsed time:  ', t_end - t_start)
      
Execute_Graph__EvaluationOps(Result_Dict, 'seven layer_reorged_w_relu_more_filters')

Result_Dict['seven layer_reorged_w_relu_more_filters'].update({'run_time':  t_end - t_start})


#view the results
Result_Dict


import pandas as pd
dfRslts = pd.DataFrame(Result_Dict)


##pass in a list of layers.  
##    - each item in the list should be a dict, with 2 keys:    first, the layer object, then the layer defn
##    - The list should be sorted in the proper order, and start with the first hidden layer
##        and end with the last hidden layer before the logit layer
#def Construct_Graph_Layers2(layer_list):
#    #input layer is placeholders that will be fed with values from each batch when the graph is executed
#    #was previous created
##    X = tf.placeholder(tf.float32, shape=(None, img_height, img_width, n_channels), name="X")
#    
#    for lyr in layer_list[:-1]:
#        lyr['lyr_object'] = lyr['lyr_defn']
#
#    #create an output layer that uses the last layer in the list
#    global logit_lyr
#    logit_lyr = lyr[-1:]['lyr_defn']
#
#
#
#layer_list2 = []
#
#lyr2_test = tf.Tensor; lyr3_test= tf.Tensor; lyr4_test = tf.Tensor
#
#layer_list2.append({'lyr_object':lyr2_test, 'lyr_defn': tf.layers.conv2d(X, filters=2, kernel_size=3, strides=[3,3], padding="SAME", name="Hidden1")})
#layer_list2.append({'lyr_object':lyr3_test, 'lyr_defn': tf.nn.max_pool(lyr2_test, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="Hidden2")})
#layer_list2.append({'lyr_object':lyr4_test, 'lyr_defn': tf.layers.conv2d(lyr3_test, filters=2, kernel_size=3, strides=[3,3], padding="SAME", name="Hidden3")})
#layer_list2.append({'lyr_object':'logit_lyr_test', 'lyr_defn':tf.layers.dense(tf.layers.Flatten()(lyr4_test), n_outputs, name="outputs")})
#
#
#
#
#Construct_Graph_Layers2(layer_list2)




import os
dir_final_submission = 'C:/Users/ashle/Documents/Personal Data/Northwestern/2018_4 FALL  PREDICT 422 Machine Learning/wk7 - deep learning for image classification'
os.listdir(dir_final_submission)




