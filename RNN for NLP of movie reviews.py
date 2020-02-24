# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 23:11:04 2018

@author: Ashley Wenger
"""

import pandas as pd

pd.set_option('display.max_columns', 40) 
pd.set_option('display.max_rows', 100) 
pd.set_option('display.width', 150)


#check which environment/Python interpreter is being used.
#import sys
#sys.executable
#when Spyder is started from command prompt w/ this folder\Scripts\sypder.exe,   I get:    Out[1]: 'c:\\users\\ashle\\AppData\\Local\\conda\\conda\\envs\\tf_gpu\\pythonw.exe'
#when Spyder is started from shortcut on desktop, I get the base folder.  
#sys.modules['itertools']


import os  # operating system functions
import os.path  # for manipulation of file path names
import numpy as np
import matplotlib.pyplot as plt
import re  # regular expressions
from collections import defaultdict
import nltk
from nltk.tokenize import TreebankWordTokenizer
import tensorflow as tf

# set parameters
RANDOM_SEED = 42
REMOVE_STOPWORDS = False  # no stopword removal.    
wkg_dir = 'C:/Users/ashle/Documents/Personal Data/Northwestern/2018_4 FALL  PREDICT 422 Machine Learning/wk8 - recurrent neural nets/'




# ==========================================================================
#    load the documents to be used in training and test
# ==========================================================================

# ------------------------------------------------------------
# code for working with movie reviews data 
# Source: Miller, T. W. (2016). Web and Network Data Science.
#    Upper Saddle River, N.J.: Pearson Education.
#    ISBN-13: 978-0-13-388644-3
# This original study used a simple bag-of-words approach
# to sentiment analysis, along with pre-defined lists of
# negative and positive words.        
# Code available at:  https://github.com/mtpa/wnds       
# ------------------------------------------------------------
# Utility function to get file names within a directory
def listdir_no_hidden(path):
    start_list = os.listdir(path)
    end_list = []
    for file in start_list:
        if (not file.startswith('.')):
            end_list.append(file)
    return(end_list)

# define list of codes to be dropped from document
# carriage-returns, line-feeds, tabs
codelist = ['\r', '\n', '\t']   


#code to remove stopwords, if desired
# !! We will not remove stopwords in this exercise because they are important to keeping sentences intact
def Create_StopWord_List():
    #print(nltk.corpus.stopwords.words('english'))

    # previous analysis of a list of top terms showed a number of words, along 
    # with contractions and other word strings to drop from further analysis, add
    # these to the usual English stopwords to be dropped from a document collection
    more_stop_words = ['cant','didnt','doesnt','dont','goes','isnt','hes',\
        'shes','thats','theres','theyre','wont','youll','youre','youve', 'br'\
        've', 're', 'vs'] 

    some_proper_nouns_to_remove = ['dick','ginger','hollywood','jack',\
        'jill','john','karloff','kudrow','orson','peter','tcm','tom',\
        'toni','welles','william','wolheim','nikita']

    # start with the initial list and add to it for movie text work 
    stoplist_ = nltk.corpus.stopwords.words('english') + more_stop_words +\
        some_proper_nouns_to_remove

    return stoplist_



# text parsing function for creating text documents 
# there is more we could do for data preparation 
# stemming... looking for contractions... possessives... 
# but we will work with what we have in this parsing function
# if we want to do stemming at a later time, we can use
#     porter = nltk.PorterStemmer()  
# in a construction like this
#     words_stemmed =  [porter.stem(word) for word in initial_words]  
def prepare_text_for_parsing(string):

    # replace non-alphabet with space 
    temp_string = re.sub('[^a-zA-Z]', '  ', string)    

    # replace codes with space
    for i in range(len(codelist)):
        stopstring = ' ' + codelist[i] + '  '
        temp_string = re.sub(stopstring, '  ', temp_string)      

    # replace single-character words with space
    temp_string = re.sub('\s.\s', ' ', temp_string)   

    # convert uppercase to lowercase
    temp_string = temp_string.lower()    

    if REMOVE_STOPWORDS:
        #get list of stop words
        stoplist = Create_StopWord_List()
        
        # replace selected character strings/stop-words with space
        for i in range(len(stoplist)):
            stopstring = ' ' + str(stoplist[i]) + ' '
            temp_string = re.sub(stopstring, ' ', temp_string)        

    # replace multiple blank characters with one blank character
    temp_string = re.sub('\s+', ' ', temp_string)    
    return(temp_string)    




# Function to read the contents of a file into memory
#     change it to lower case, clean it up and parse it into a list of words
def read_doc_and_parse_into_words(filename):
    """Returns a list of words.
    
        The input string is first converted to lower case and cleansed.
    """   
    with open(filename, encoding='utf-8') as f:
        data = tf.compat.as_str(f.read())
        data = data.lower()
        data = prepare_text_for_parsing(data)
        data = TreebankWordTokenizer().tokenize(data)  # The Penn Treebank

    return data




# -----------------------------------------------
# gather data for 500 negative movie reviews
# Data will be stored in a list of lists where the each list represents 
# a document and document is a list of words.
# We then break the text into words.
# -----------------------------------------------

#gather the filenames of the negative movie reviews
dir_name = wkg_dir + 'raw_data/movie-reviews-negative'  
filenames = listdir_no_hidden(path=dir_name)
num_files = len(filenames)

#check that each filename captured really is a file (vs. a directory or link??)
for i in range(len(filenames)):
    file_exists = os.path.isfile(os.path.join(dir_name, filenames[i]))
    assert file_exists
print('\nDirectory:',dir_name)    
print('%d files found' % len(filenames))


#read the NEGATIVE documents and their text into memory
negative_documents = []
print('\nProcessing document files under', dir_name)
for i in range(num_files):
    ## print(' ', filenames[i])
    words = read_doc_and_parse_into_words(os.path.join(dir_name, filenames[i]))
    negative_documents.append(words)
    # print('Data size (Characters) (Document %d) %d' %(i,len(words)))
    # print('Sample string (Document %d) %s'%(i,words[:50]))




# -----------------------------------------------
# gather data for 500 positive movie reviews
# -----------------------------------------------
#gather the filenames of the POSITIVE movie reviews
dir_name = wkg_dir + 'raw_data/movie-reviews-positive'
filenames = listdir_no_hidden(path=dir_name)
num_files = len(filenames)

#check that each filename captured really is a file (vs. a directory or link??)
for i in range(len(filenames)):
    file_exists = os.path.isfile(os.path.join(dir_name, filenames[i]))
    assert file_exists
print('\nDirectory:',dir_name)    
print('%d files found' % len(filenames))


#read the POSITIVE documents and their text into memory
positive_documents = []
print('\nProcessing document files under', dir_name)
for i in range(num_files):
    words = read_doc_and_parse_into_words(os.path.join(dir_name, filenames[i]))
    positive_documents.append(words)



##check out the min and max length of reviews
#max_review_length = 0  # initialize
#for doc in negative_documents:
#    max_review_length = max(max_review_length, len(doc))    
#for doc in positive_documents:
#    max_review_length = max(max_review_length, len(doc)) 
#print('max_review_length:', max_review_length) 
##longest review (after cleansing) is 1052 words
#
#min_review_length = max_review_length  # initialize
#for doc in negative_documents:
#    min_review_length = min(min_review_length, len(doc))    
#for doc in positive_documents:
#    min_review_length = min(min_review_length, len(doc)) 
#print('min_review_length:', min_review_length) 
##shortest review (after cleansing) is 22 words



# -----------------------------------------------------
# convert positive/negative documents into numpy array
# note that reviews vary from 22 to 1052 words   
# so we use the first 20 and last 20 words of each review 
# as our word sequences for analysis
# -----------------------------------------------------
# construct list of 1000 lists with 40 words in each list
from itertools import chain
documents = []
for doc in negative_documents:
    doc_begin = doc[0:20]
    doc_end = doc[len(doc) - 20: len(doc)]
    documents.append(list(chain(*[doc_begin, doc_end])))    
for doc in positive_documents:
    doc_begin = doc[0:20]
    doc_end = doc[len(doc) - 20: len(doc)]
    documents.append(list(chain(*[doc_begin, doc_end])))    



# Define the labels to be used 500 negative (0) and 500 positive (1)
thumbs_down_up = np.concatenate((np.zeros((500), dtype = np.int32), 
                                 np.ones((500), dtype = np.int32)), 
                                axis = 0)


# ====================
# end of document prep
# ====================
    







# ================================================================
#      gather pre-trained vectors that describe each word (aka embeddings)
#      these "libraries" of embeddings have been built on large sets of words, with
#                varying #s of dimensions per word dimensions.  Will facilitate analysis vs. onehot encoding
#             of words b/c it has been trained in a way that like words will have similar vectors (at least on 
#             some dimensions of similarity
#    
# ================================================================


#this util allows easy download of pre-trained word vectors, with a variety of languages, dimensions (vector lengths), vocabulary sizes, methods, corpuses, etc
import chakin
#returns a pandas dataframe-ish kind of result.  had no luck assigning the output to an object, like:   rslt = chakin.search,  but was able to see all columns
#    after setting pandas options for display width, max columns etc.   Otherwise got ... instead of the middle columns.
chakin.search(lang='English')


chakin.download(number=2, save_dir=wkg_dir)


#chakin.search(lang='English')
#                   Name  Dimension                     Corpus VocabularySize    Method Language    Author
#2          fastText(en)        300                  Wikipedia           2.5M  fastText  English  Facebook
#11         GloVe.6B.50d         50  Wikipedia+Gigaword 5 (6B)           400K     GloVe  English  Stanford
#12        GloVe.6B.100d        100  Wikipedia+Gigaword 5 (6B)           400K     GloVe  English  Stanford
#13        GloVe.6B.200d        200  Wikipedia+Gigaword 5 (6B)           400K     GloVe  English  Stanford
#14        GloVe.6B.300d        300  Wikipedia+Gigaword 5 (6B)           400K     GloVe  English  Stanford
#15       GloVe.42B.300d        300          Common Crawl(42B)           1.9M     GloVe  English  Stanford
#16      GloVe.840B.300d        300         Common Crawl(840B)           2.2M     GloVe  English  Stanford
#17    GloVe.Twitter.25d         25               Twitter(27B)           1.2M     GloVe  English  Stanford
#18    GloVe.Twitter.50d         50               Twitter(27B)           1.2M     GloVe  English  Stanford
#19   GloVe.Twitter.100d        100               Twitter(27B)           1.2M     GloVe  English  Stanford
#20   GloVe.Twitter.200d        200               Twitter(27B)           1.2M     GloVe  English  Stanford
#21  word2vec.GoogleNews        300          Google News(100B)           3.0M  word2vec  English    Google

# download several different vectors so we can compare performance with different vector lengths, vocabulary sizes, etc.
# =====================================================================================================================
# a) fastText vector.   This is big in both vocab (rows) and dimension (columns)
#                   Name  Dimension                     Corpus VocabularySize    Method Language    Author
#2          fastText(en)        300                  Wikipedia           2.5M  fastText  English  Facebook
chakin.download(number=2, save_dir=wkg_dir)
#Test: 100% ||                                      | Time:  0:26:00   4.0 MiB/s
#Out[5]: 'C:/Users/ashle/Documents/Personal Data/Northwestern/2018_4 FALL  PREDICT 422 Machine Learning/wk8 - recurrent neural nets/vector_files\\wiki.en.vec'


#tried downloading a list, but it wasn't set up for that, gave an error        chakin.download(number=[17, 18, 19, 20], save_dir=wkg_dir)

#B1)   try a set of GloVe ones w/ different dimensions (from fastText and from each other) but same vocab  so we can compare impact of dimension while holding constant the vocab size and corpus
#18    GloVe.Twitter.50d         50               Twitter(27B)           1.2M     GloVe  English  Stanford
#                   Name  Dimension                     Corpus VocabularySize    Method Language    Author
chakin.download(number=18, save_dir=wkg_dir)
#Test: 100% ||                                      | Time:  0:07:54   3.1 MiB/s
#Out[9]: 'C:/Users/ashle/Documents/Personal Data/Northwestern/2018_4 FALL  PREDICT 422 Machine Learning/wk8 - recurrent neural nets/vector_files\\glove.twitter.27B.zip'

#B2)   GloVe ones w/ 100 dimensions
#NOTE!   had to rename the output file or else a subsequent download from the GloVe Twitter family overwrote it with a new file of the same name (with no warning :( )
#19   GloVe.Twitter.100d        100               Twitter(27B)           1.2M     GloVe  English  Stanford
chakin.download(number=19, save_dir=wkg_dir)
#Test: 100% ||                                      | Time:  0:04:10   5.8 MiB/s
#Out[11]: 'C:/Users/ashle/Documents/Personal Data/Northwestern/2018_4 FALL  PREDICT 422 Machine Learning/wk8 - recurrent neural nets/vector_files\\glove.twitter.27B.zip'

#B3)   GloVe ones w/ 200 dimensions
#20   GloVe.Twitter.200d        200               Twitter(27B)           1.2M     GloVe  English  Stanford
chakin.download(number=20, save_dir=wkg_dir)
#Test: 100% ||                                      | Time:  0:07:39   3.2 MiB/s
#Out[10]: 'C:/Users/ashle/Documents/Personal Data/Northwestern/2018_4 FALL  PREDICT 422 Machine Learning/wk8 - recurrent neural nets/vector_files\\glove.twitter.27B.zip'

# ================================================================
#   END of   gather pre-trained vectors
# ================================================================















## ------------------------------------------------------------- 
## Select the pre-defined embeddings source        
## Define vocabulary size for the language model    
## Create a word_to_embedding_dict for GloVe.6B.50d
#embeddings_directory = 'embeddings/'
#filename = 'glove.twitter.27B.25d.txt'
#embeddings_filename = os.path.join(wkg_dir, embeddings_directory, filename)
## ------------------------------------------------------------- 
#
#
##check out the embeddings file format
#with open(embeddings_filename, 'r', encoding='utf-8') as embeddings_file:
#    #print(len(enumerate(embeddings_file)))
#    for i, lineval in enumerate(embeddings_file):
#        if i < 200:
#            print(i, lineval)
#        else:
#            pass
  
#interesting, many of the words are 
#    shapes like heart ❤  or star ★
#    or shorthand like '<lolface', '<heart>', <sadface>
#    or punctuation/symbols like '>>', '[', ')', &,  ω, ...
#   or French (je, le, ) or Portuguese (não)  or Spanish (tu, las, como, ..) or Kanji (笑
#   or abbreviations like lol, yg, 



#print('Nbr of lines total:  ', i+1)  #1,193,514
#print('Last line:  ', lineval)  #1,193,514
#Last line:   ﾟﾟﾟｵﾔｽﾐｰ -2.5807 -1.0965 -0.59056 1.1178 -0.30615 -0.44198 -1.377 -2.3494 2.0436 -0.15692 2.6962 1.033 0.81358 -1.7224 0.066939 -0.71714 1.0608 -0.43463 2.1178 0.65876 0.62825 -1.2018 1.7123 0.79867 0.32424    
    #strange that the "word" is a Japanese/Chinese char when this set is supposed to be English
    
    
#embeddings_file = open(embeddings_filename, 'r', encoding='utf-8')
#type(embeddings_file)
#embeddings_file.readlines(100)
#embeddings_file.close()






# Utility function for loading embeddings follows methods described in
# https://github.com/guillaume-chevalier/GloVe-as-a-TensorFlow-Embedding-Layer
# Creates the Python defaultdict dictionary word_to_embedding_dict
# for the requested pre-trained word embeddings
# 
# Note the use of defaultdict data structure from the Python Standard Library
# collections_defaultdict.py lets the caller specify a default value up front.
# The default value (0) will be returned if the key is not a known dictionary key.
# That is, unknown words are represented by a vector of zeros.
# For word embeddings, this default value is a vector of zeros
# Documentation for the Python standard library:
#   Hellmann, D. 2017. The Python 3 Standard Library by Example. Boston: 
#     Addison-Wesley. [ISBN-13: 978-0-13-429105-5]
def load_embedding_from_disks(embeddings_filename, with_indexes=True):
    """
    Read a embeddings txt file. If `with_indexes=True`, 
    we return a tuple of two dictionaries
    `(word_to_index_dict, index_to_embedding_array)`, 
    otherwise we return only a direct 
    `word_to_embedding_dict` dictionary mapping 
    from a string to a numpy array.
    """
    if with_indexes:
        word_to_index_dict = dict()
        index_to_embedding_array = []
  
    else:
        word_to_embedding_dict = dict()

    with open(embeddings_filename, 'r', encoding='utf-8') as embeddings_file:
        for (i, line) in enumerate(embeddings_file):

            split = line.split(' ')

            word = split[0]

            representation = split[1:]
            representation = np.array(
                [float(val) for val in representation]
            )

            if with_indexes:
                word_to_index_dict[word] = i
                index_to_embedding_array.append(representation)
            else:
                word_to_embedding_dict[word] = representation

    # Empty representation for unknown words.
    _WORD_NOT_FOUND = [0.0] * len(representation)
    if with_indexes:
        _LAST_INDEX = i + 1
        word_to_index_dict = defaultdict(
            lambda: _LAST_INDEX, word_to_index_dict)
        index_to_embedding_array = np.array(
            index_to_embedding_array + [_WORD_NOT_FOUND])
        return word_to_index_dict, index_to_embedding_array
    else:
        word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
        return word_to_embedding_dict


#print('\nLoading embeddings from', embeddings_filename)
#word_to_index, index_to_embedding = \
#    load_embedding_from_disks(embeddings_filename, with_indexes=True)
#print("Embedding loaded from disks.")





# Note: unknown words have representations with values [0, 0, ..., 0]

## Additional background code from
## https://github.com/guillaume-chevalier/GloVe-as-a-TensorFlow-Embedding-Layer
## shows the general structure of the data structures for word embeddings
## This code is modified for our purposes in language modeling 
#vocab_size, embedding_dim = index_to_embedding.shape
#print("Embedding is of shape: {}".format(index_to_embedding.shape))
#print("This means (number of words, number of dimensions per word)\n")
#print("The first words are words that tend occur more often.")
#
#print("Note: for unknown words, the representation is an empty vector,\n"
#      "and the index is the last one. The dictionary has a limit:")
#
##look up some sample words
#smpl_word = "the"
#idx_4_smpl_word = word_to_index[smpl_word]
#embd_4_smpl_word = list(index_to_embedding[idx_4_smpl_word])  # "int" for compact print only.
#print("    {} --> {} --> {}".format("A word", "Index in embedding", 
#      "Representation"))
#print("    {} --> {} --> {}".format(smpl_word, idx_4_smpl_word, embd_4_smpl_word))
#
#
#len(word_to_index)
#
#unknown_wrd = "worsdfkljsdf"  # a word obviously not in the vocabulary
#idx_4_unknown_wrds = word_to_index[unknown_wrd] # index for word obviously not in the vocabulary
#complete_vocabulary_size = idx_4_unknown_wrds 
#embd_4_unknown_wrds = list(np.array(index_to_embedding[idx_4_unknown_wrds], dtype=int)) # "int" compact print
#print("    {} --> {} --> {}".format("A word", "Index in embedding", 
#      "Representation"))
#print("    {} --> {} --> {}".format(unknown_wrd, idx_4_unknown_wrds, embd_4_unknown_wrds))
#
#
#
#
#
#
## Show how to use embeddings dictionaries with a test sentence
## This is a famous typing exercise with all letters of the alphabet
## https://en.wikipedia.org/wiki/The_quick_brown_fox_jumps_over_the_lazy_dog
#a_typing_test_sentence = 'The quick brown fox jumps over the lazy dog'
#print('\nTest sentence: ', a_typing_test_sentence, '\n')
#words_in_test_sentence = a_typing_test_sentence.split()
#
#print('Test sentence embeddings from complete vocabulary of', 
#      complete_vocabulary_size, 'words:\n')
#for word in words_in_test_sentence:
#    word_ = word.lower()
#    embedding = index_to_embedding[word_to_index[word_]]
#    print(word_ + ": ", embedding)
#
#
#
##
#first_ten_dict = {k: v for k, v in word_to_index.items() if v < 10}









# ------------------------------------------------------------- 
# Define vocabulary size for the language model    
# To reduce the size of the vocabulary to the n most frequently used words
def Create_Truncated_Vocabs(word_to_index_dict, index_to_embedding_list, new_size=10000):
    
    def default_factory_fn():
        return new_size  
    # if the word=>index dict is asked for a word it doesn't know the embedding for, it will return this value as the index for unknown words, 
    # and in the embeddings dict, this value will be the key for an all zero ("unknown word") vector

    #create a default dict with (vocab_size) most commonly used words
    #(apparently the words are indexed from 0 to N with the most commonly used words being first
    #the word=>index dict has the word as the key and the index as the value.  Hence the first 10k values of the dict are the 
        #10k most commonly used words
    truncated_word_to_index_dict = defaultdict(default_factory_fn, \
                                               {k: v for k, v in word_to_index_dict.items() if v < new_size})

    #look up words by their text, gives back the index of that word.  
        #If the given word is not in the dictionary, it returns EVOCABSIZE
    #limited_word_to_index['the']   #returns 13
    #limited_word_to_index['lol']   #returns 88
    #limited_word_to_index['friend']   #returns 531
    #
    #limited_word_to_index['asdfsdfasdf']   #returns 10000
    #limited_word_to_index['im not found']   #returns 10000
    #limited_word_to_index[12345]   #returns 10000
       #note that these words that weren't previously in the dict are now added (new keys), each with a val of 10000 as per the default_factory_fn
    #for k,v in limited_word_to_index.items():
    #    if v < 40 or v > 9960:
    #        print(v, k)
   

    #create the corresponding index_to_embedding dict, which, when given an index, gives back the corresponding embedding (a 1D array of 25 diff values)
    # ------------------------------------------------    
    # a) Select the first EVOCABSIZE rows of the index_to_embedding dict
    truncated_index_to_embedding_list = index_to_embedding_list[0:new_size,:]

    # b) for the unknown words, for which the word_to_index dict will give back an index of EVOCABSIZE,
    #    set the embeddings for key=EVOCABSIZE to be all zeros as previously discussed
    embedding_size = len(index_to_embedding_list[0])
    truncated_index_to_embedding_list = np.append(truncated_index_to_embedding_list, 
                                       np.zeros((1,embedding_size), dtype=np.float64),
                                       axis = 0)
                                       
 #                                      index_to_embedding[index_to_embedding.shape[0] - 1, :].reshape(1,embedding_dim), 
 #                                      axis = 0)
    return truncated_word_to_index_dict, truncated_index_to_embedding_list


#limited_index_to_embedding.shape
#10000 rows, 25 columns

#limited_word_to_index.items().shape
#now up to 10006 b/c we played with some test values above


# Verify the new vocabulary: should get same embeddings for test sentence
#print('\nTest sentence embeddings from vocabulary of', EVOCABSIZE, 'words:\n')
#for word in words_in_test_sentence:
#    word_ = word.lower()
#    embedding = limited_index_to_embedding[limited_word_to_index[word_]]
#    print(word_ + ": ", embedding)
# Note that "jumps" is not found due to small (due to truncation) set of vectors for which we have embeddings









# --------------------------------------------------------------------------      
# We use a very simple Recurrent Neural Network for this assignment
# Géron, A. 2017. Hands-On Machine Learning with Scikit-Learn & TensorFlow: 
#    Concepts, Tools, and Techniques to Build Intelligent Systems. 
#    Sebastopol, Calif.: O'Reilly. [ISBN-13 978-1-491-96229-9] 
#    Chapter 14 Recurrent Neural Networks, pages 390-391
#    Source code available at https://github.com/ageron/handson-ml
#    Jupyter notebook file 14_recurrent_neural_networks.ipynb
#    See section on Training an sequence Classifier, # In [34]:
#    which uses the MNIST case data...  we revise to accommodate
#    the movie review data in this assignment    
# --------------------------------------------------------------------------  

# To make output stable across runs
def reset_graph(seed= RANDOM_SEED):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)



#construct the graph
def Construct_Graph():
    n_steps = embeddings_array.shape[1]  # number of words per document 
    n_inputs = embeddings_array.shape[2]  # dimension of  pre-trained embeddings
    n_neurons = 20  # analyst specified number of neurons
    n_outputs = 2  # thumbs-down or thumbs-up

    learning_rate = 0.001

    global X
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    global y
    y = tf.placeholder(tf.int32, [None])

    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    logits = tf.layers.dense(states, n_outputs)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    global training_op
    training_op = optimizer.minimize(loss)
    correct = tf.nn.in_top_k(logits, y, 1)
    global accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    global init
    init = tf.global_variables_initializer()


#execute the graph
def Execute_Graph(mdl_name, n_epochs = 100):
    batch_size = 100

    accuracy_lst = []
    
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            print('\n  ---- Epoch ', epoch, ' ----\n')
            for iteration in range(y_train.shape[0] // batch_size):          
                X_batch = X_train[iteration*batch_size:(iteration + 1)*batch_size,:]
                y_batch = y_train[iteration*batch_size:(iteration + 1)*batch_size]
                print('  Batch ', iteration, ' training observations from ',  
                      iteration*batch_size, ' to ', (iteration + 1)*batch_size-1,)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
                print('\n  Train accuracy:', acc_train, 'Test accuracy:', acc_test)
            accuracy_lst.append( {'Model_Name': mdl_name, 'Epoch': epoch, 'Acc_Train': acc_train, 'Acc_Test': acc_test} ) 

    return accuracy_lst 

















from time import time

Accuracy_List = []
RunTime_List = []
embeddings_directory = wkg_dir + 'embeddings/'

# ==========================================================================================
#     test cell 1:    use the Glove.Twitter.27B.25d embedding, with a truncated vocab of the top ten thousand words
# ==========================================================================================
Model_Nm = 'GloVe.Twitter.25d__10K_wrds'
new_vocab_size = 10000
t_start = time()


#load the embedding from disk
filename = 'glove.twitter.27B.25d.txt'
embeddings_filename = os.path.join(embeddings_directory, filename)

word_to_index_dict__TestCell1, index_to_embedding_list__TestCell1 = load_embedding_from_disks(embeddings_filename, with_indexes=True)

#create the truncated embedding list
word_to_index_dict_SMALL__TestCell1, index_to_embedding_list_SMALL__TestCell1 = \
    Create_Truncated_Vocabs(word_to_index_dict__TestCell1, index_to_embedding_list__TestCell1, new_size=new_vocab_size)

# Delete large numpy array to clear some CPU RAM
del index_to_embedding_list__TestCell1


# set up the training and test sets
# ---------------------------------
# a.   create list of lists of lists for embeddings of the documents in our training+test collection
#             outermost list is documents
#                  next list is words in document
#                       innermost list is values of the embedding vector for that word
embeddings = []    
for doc in documents: 
    embedding_list_current_doc = []

    for word in doc:
        idx_current_wrd = word_to_index_dict_SMALL__TestCell1[word]
        embdding_curr_wrd = index_to_embedding_list_SMALL__TestCell1[idx_current_wrd] 
        embedding_list_current_doc.append(embdding_curr_wrd) 
        
    embeddings.append(embedding_list_current_doc)


#b.  make embeddings a numpy array for use in an RNN 
embeddings_array = np.array(embeddings)

#c.  create training and test sets with Scikit Learn
from sklearn.model_selection import train_test_split

# Random splitting of the data in to training (80%) and test (20%)  
X_train, X_test, y_train, y_test = \
    train_test_split(embeddings_array, thumbs_down_up, test_size=0.20, 
                     random_state = RANDOM_SEED)


# Train and evaluate the model
reset_graph()

Construct_Graph()


new_rslt = Execute_Graph(mdl_name=Model_Nm)
t_end = time()


Accuracy_List.append(new_rslt)
RunTime_List.append( {'Model_Name': Model_Nm, 'RunTime':t_end - t_start} ) 






# ==========================================================================================
#     test cell 2:    use the Glove.Twitter.27B.200d embedding, with a truncated vocab of the top ten thousand words
# ==========================================================================================
Model_Nm = 'GloVe.Twitter.200d__10K_wrds'
new_vocab_size = 10000
t_start = time()


#load the embedding from disk
filename = 'glove.twitter.27B.200d.txt'
embeddings_filename = os.path.join(embeddings_directory, filename)

word_to_index_dict__TestCell1, index_to_embedding_list__TestCell1 = load_embedding_from_disks(embeddings_filename, with_indexes=True)

#create the truncated embedding list
word_to_index_dict_SMALL__TestCell1, index_to_embedding_list_SMALL__TestCell1 = \
    Create_Truncated_Vocabs(word_to_index_dict__TestCell1, index_to_embedding_list__TestCell1, new_size=new_vocab_size)

# Delete large numpy array to clear some CPU RAM
del index_to_embedding_list__TestCell1


# set up the training and test sets
# ---------------------------------
# a.   create list of lists of lists for embeddings of the documents in our training+test collection
#             outermost list is documents
#                  next list is words in document
#                       innermost list is values of the embedding vector for that word
embeddings = []    
for doc in documents: 
    embedding_list_current_doc = []

    for word in doc:
        idx_current_wrd = word_to_index_dict_SMALL__TestCell1[word]
        embdding_curr_wrd = index_to_embedding_list_SMALL__TestCell1[idx_current_wrd] 
        embedding_list_current_doc.append(embdding_curr_wrd) 
        
    embeddings.append(embedding_list_current_doc)


#b.  make embeddings a numpy array for use in an RNN 
embeddings_array = np.array(embeddings)

#c.  create training and test sets with Scikit Learn
from sklearn.model_selection import train_test_split

# Random splitting of the data in to training (80%) and test (20%)  
X_train, X_test, y_train, y_test = \
    train_test_split(embeddings_array, thumbs_down_up, test_size=0.20, 
                     random_state = RANDOM_SEED)


# Train and evaluate the model
reset_graph()

Construct_Graph()

new_rslt = Execute_Graph(mdl_name=Model_Nm)
t_end = time()


Accuracy_List.append(new_rslt)
RunTime_List.append( {'Model_Name': Model_Nm, 'RunTime':t_end - t_start} ) 


import pickle
pkl_fl_accrcy_lst = wkg_dir + 'pckl1.dat'
pkl_fl_runtime_lst = wkg_dir + 'pckl2.dat'

with open(pkl_fl_accrcy_lst, 'wb') as fl:
    pickle.dump(Accuracy_List, fl)
with open(pkl_fl_runtime_lst, 'wb') as fl:
    pickle.dump(RunTime_List, fl)
    


#restart kernel to clear out memory use, then rerun the code above as necessary and reload from pickle for last 2
import pickle
pkl_fl_accrcy_lst = wkg_dir + 'pckl1.dat'
pkl_fl_runtime_lst = wkg_dir + 'pckl2.dat'

with open(pkl_fl_accrcy_lst, 'rb') as fl:
    Accuracy_List = pickle.load(fl)
with open(pkl_fl_runtime_lst, 'rb') as fl:
    RunTime_List = pickle.load(fl)
    


# ==========================================================================================
#     test cell 3:    use the Glove.Twitter.27B.25d embedding, with a truncated vocab of the top 100K words
# ==========================================================================================
Model_Nm = 'GloVe.Twitter.25d__100K_wrds'
new_vocab_size = 100000
t_start = time()


#load the embedding from disk
embeddings_directory = wkg_dir + 'embeddings/'
filename = 'glove.twitter.27B.25d.txt'
embeddings_filename = os.path.join(embeddings_directory, filename)

word_to_index_dict__TestCell1, index_to_embedding_list__TestCell1 = load_embedding_from_disks(embeddings_filename, with_indexes=True)

#create the truncated embedding list
word_to_index_dict_SMALL__TestCell1, index_to_embedding_list_SMALL__TestCell1 = \
    Create_Truncated_Vocabs(word_to_index_dict__TestCell1, index_to_embedding_list__TestCell1, new_size=new_vocab_size)

# Delete large numpy array to clear some CPU RAM
del index_to_embedding_list__TestCell1


# set up the training and test sets
# ---------------------------------
# a.   create list of lists of lists for embeddings of the documents in our training+test collection
#             outermost list is documents
#                  next list is words in document
#                       innermost list is values of the embedding vector for that word
embeddings = []    
for doc in documents: 
    embedding_list_current_doc = []

    for word in doc:
        idx_current_wrd = word_to_index_dict_SMALL__TestCell1[word]
        embdding_curr_wrd = index_to_embedding_list_SMALL__TestCell1[idx_current_wrd] 
        embedding_list_current_doc.append(embdding_curr_wrd) 
        
    embeddings.append(embedding_list_current_doc)


#b.  make embeddings a numpy array for use in an RNN 
embeddings_array = np.array(embeddings)

#c.  create training and test sets with Scikit Learn
from sklearn.model_selection import train_test_split

# Random splitting of the data in to training (80%) and test (20%)  
X_train, X_test, y_train, y_test = \
    train_test_split(embeddings_array, thumbs_down_up, test_size=0.20, 
                     random_state = RANDOM_SEED)


# Train and evaluate the model
reset_graph()

Construct_Graph()

new_rslt = Execute_Graph(mdl_name=Model_Nm)
t_end = time()


Accuracy_List.append(new_rslt)
RunTime_List.append( {'Model_Name': Model_Nm, 'RunTime':t_end - t_start} ) 





# ==========================================================================================
#     test cell 4:    use the Glove.Twitter.27B.200d embedding, with a truncated vocab of the top 100K words
# ==========================================================================================
Model_Nm = 'GloVe.Twitter.200d__100K_wrds'
new_vocab_size = 100000
t_start = time()


#load the embedding from disk
embeddings_directory = wkg_dir + 'embeddings/'
filename = 'glove.twitter.27B.200d.txt'
embeddings_filename = os.path.join(embeddings_directory, filename)

word_to_index_dict__TestCell1, index_to_embedding_list__TestCell1 = load_embedding_from_disks(embeddings_filename, with_indexes=True)

#create the truncated embedding list
word_to_index_dict_SMALL__TestCell1, index_to_embedding_list_SMALL__TestCell1 = \
    Create_Truncated_Vocabs(word_to_index_dict__TestCell1, index_to_embedding_list__TestCell1, new_size=new_vocab_size)

# Delete large numpy array to clear some CPU RAM
del index_to_embedding_list__TestCell1


# set up the training and test sets
# ---------------------------------
# a.   create list of lists of lists for embeddings of the documents in our training+test collection
#             outermost list is documents
#                  next list is words in document
#                       innermost list is values of the embedding vector for that word
embeddings = []    
for doc in documents: 
    embedding_list_current_doc = []

    for word in doc:
        idx_current_wrd = word_to_index_dict_SMALL__TestCell1[word]
        embdding_curr_wrd = index_to_embedding_list_SMALL__TestCell1[idx_current_wrd] 
        embedding_list_current_doc.append(embdding_curr_wrd) 
        
    embeddings.append(embedding_list_current_doc)


#b.  make embeddings a numpy array for use in an RNN 
embeddings_array = np.array(embeddings)

#c.  create training and test sets with Scikit Learn
from sklearn.model_selection import train_test_split

# Random splitting of the data in to training (80%) and test (20%)  
X_train, X_test, y_train, y_test = \
    train_test_split(embeddings_array, thumbs_down_up, test_size=0.20, 
                     random_state = RANDOM_SEED)


# Train and evaluate the model
reset_graph()

Construct_Graph()

new_rslt = Execute_Graph(mdl_name=Model_Nm)
t_end = time()


Accuracy_List.append(new_rslt)
RunTime_List.append( {'Model_Name': Model_Nm, 'RunTime':t_end - t_start} ) 





#count the # of words per document that are not found with 10k vocab vs. 100k vocab
wrds_not_fnd =   [{'Corpus':'GloVe.Twitter.25d', 'Wrds_Missing_10k': [], 'Wrds_Missing_100k': [], 'Wrds_Missing_1MM': []}, 
                       {'Corpus':'GloVe.Twitter.200d', 'Wrds_Missing_10k': [], 'Wrds_Missing_100k': [], 'Wrds_Missing_1MM': []}
                      ]
embeddings_directory = wkg_dir + 'embeddings/'

def Count_Missing_Wrds(filenm):
    #load the files into memory
    embeddings_filename = os.path.join(embeddings_directory, filenm)
    word_to_index_dict_, index_to_embedding_list_ = load_embedding_from_disks(embeddings_filename, with_indexes=True)
    if filenm == 'glove.twitter.27B.25d.txt': 
        rownbr = 0
    else:
        rownbr = 1

    #count the words not found when we truncate to 10k
    #-------------------------------------------------
    #create the truncated embedding list
    word_to_index_dict_SMALL, index_to_embedding_list_SMALL = \
        Create_Truncated_Vocabs(word_to_index_dict_, index_to_embedding_list_, new_size=10000)

    wrd_not_fnd_lst = []
    for doc in documents:
        for wrd in doc:
            if word_to_index_dict_SMALL[wrd] == 10000:
                wrd_not_fnd_lst.append(wrd)
            
    wrds_not_fnd[rownbr]['Wrds_Missing_10k'] = wrd_not_fnd_lst


    #repeat for 100k
    word_to_index_dict_SMALL, index_to_embedding_list_SMALL = \
        Create_Truncated_Vocabs(word_to_index_dict_, index_to_embedding_list_, new_size=100000)

    wrd_not_fnd_lst = []
    for doc in documents:
        for wrd in doc:
            if word_to_index_dict_SMALL[wrd] == 100000:
                wrd_not_fnd_lst.append(wrd)
            
    wrds_not_fnd[rownbr]['Wrds_Missing_100k'] = wrd_not_fnd_lst

    #repeat for 1MM
    word_to_index_dict_SMALL, index_to_embedding_list_SMALL = \
        Create_Truncated_Vocabs(word_to_index_dict_, index_to_embedding_list_, new_size=1000000)

    wrd_not_fnd_lst = []
    for doc in documents:
        for wrd in doc:
            if word_to_index_dict_SMALL[wrd] == 1000000:
                wrd_not_fnd_lst.append(wrd)
            
    wrds_not_fnd[rownbr]['Wrds_Missing_1MM'] = wrd_not_fnd_lst



Count_Missing_Wrds('glove.twitter.27B.25d.txt')

len(wrds_not_fnd[0]['Wrds_Missing_10k'])    #6105 not found when using top 10k
len(wrds_not_fnd[0]['Wrds_Missing_100k'])    #1710 not found when using top 100k
len(wrds_not_fnd[0]['Wrds_Missing_1MM'])    #372 not found when using top 100k

Count_Missing_Wrds('glove.twitter.27B.200d.txt')
len(wrds_not_fnd[1]['Wrds_Missing_10k'])    #6105 not found when using top 10k
len(wrds_not_fnd[1]['Wrds_Missing_100k'])    #1710 not found when using top 100k
len(wrds_not_fnd[1]['Wrds_Missing_1MM'])    #372
 not found when using top 100k


#I don't think the # of words in the truncated list will change the size of memory (or compute time??) required?
#    it will just replace the all-zero vector with more info.  Hence, going to go for 1mm vocab size and try to find all the words

import pickle
pkl_fl_accrcy_lst = wkg_dir + 'pckl1.dat'
pkl_fl_runtime_lst = wkg_dir + 'pckl2.dat'

with open(pkl_fl_accrcy_lst, 'wb') as fl:
    pickle.dump(Accuracy_List, fl)
with open(pkl_fl_runtime_lst, 'wb') as fl:
    pickle.dump(RunTime_List, fl)
    


#restart kernel to clear out memory use, then rerun the code above as necessary and reload from pickle for last 2
import pickle
pkl_fl_accrcy_lst = wkg_dir + 'pckl1.dat'
pkl_fl_runtime_lst = wkg_dir + 'pckl2.dat'

with open(pkl_fl_accrcy_lst, 'rb') as fl:
    Accuracy_List = pickle.load(fl)
with open(pkl_fl_runtime_lst, 'rb') as fl:
    RunTime_List = pickle.load(fl)





# ==========================================================================================
#     test cell 5:    use the Glove.Twitter.27B.25d embedding, with a truncated vocab of the top MILLION words
# ==========================================================================================
Model_Nm = 'GloVe.Twitter.25d__1MM_wrds'
new_vocab_size = 1000000
t_start = time()


#load the embedding from disk
filename = 'glove.twitter.27B.25d.txt'
embeddings_filename = os.path.join(embeddings_directory, filename)

word_to_index_dict__TestCell1, index_to_embedding_list__TestCell1 = load_embedding_from_disks(embeddings_filename, with_indexes=True)

#create the truncated embedding list
word_to_index_dict_SMALL__TestCell1, index_to_embedding_list_SMALL__TestCell1 = \
    Create_Truncated_Vocabs(word_to_index_dict__TestCell1, index_to_embedding_list__TestCell1, new_size=new_vocab_size)

# Delete large numpy array to clear some CPU RAM
del index_to_embedding_list__TestCell1


# set up the training and test sets
# ---------------------------------
# a.   create list of lists of lists for embeddings of the documents in our training+test collection
#             outermost list is documents
#                  next list is words in document
#                       innermost list is values of the embedding vector for that word
embeddings = []    
for doc in documents: 
    embedding_list_current_doc = []

    for word in doc:
        idx_current_wrd = word_to_index_dict_SMALL__TestCell1[word]
        embdding_curr_wrd = index_to_embedding_list_SMALL__TestCell1[idx_current_wrd] 
        embedding_list_current_doc.append(embdding_curr_wrd) 
        
    embeddings.append(embedding_list_current_doc)


#b.  make embeddings a numpy array for use in an RNN 
embeddings_array = np.array(embeddings)

#c.  create training and test sets with Scikit Learn
from sklearn.model_selection import train_test_split

# Random splitting of the data in to training (80%) and test (20%)  
X_train, X_test, y_train, y_test = \
    train_test_split(embeddings_array, thumbs_down_up, test_size=0.20, 
                     random_state = RANDOM_SEED)


# Train and evaluate the model
reset_graph()

Construct_Graph()


new_rslt = Execute_Graph(mdl_name=Model_Nm)
t_end = time()


Accuracy_List.append(new_rslt)
RunTime_List.append( {'Model_Name': Model_Nm, 'RunTime':t_end - t_start} ) 






# ==========================================================================================
#     test cell 6:    use the Glove.Twitter.27B.200d embedding, with a truncated vocab of the top MILLION words
# ==========================================================================================
Model_Nm = 'GloVe.Twitter.200d__1MM_wrds'
new_vocab_size = 1000000
t_start = time()


#load the embedding from disk
filename = 'glove.twitter.27B.200d.txt'
embeddings_filename = os.path.join(embeddings_directory, filename)

word_to_index_dict__TestCell1, index_to_embedding_list__TestCell1 = load_embedding_from_disks(embeddings_filename, with_indexes=True)

#create the truncated embedding list
word_to_index_dict_SMALL__TestCell1, index_to_embedding_list_SMALL__TestCell1 = \
    Create_Truncated_Vocabs(word_to_index_dict__TestCell1, index_to_embedding_list__TestCell1, new_size=new_vocab_size)

# Delete large numpy array to clear some CPU RAM
del index_to_embedding_list__TestCell1


# set up the training and test sets
# ---------------------------------
# a.   create list of lists of lists for embeddings of the documents in our training+test collection
#             outermost list is documents
#                  next list is words in document
#                       innermost list is values of the embedding vector for that word
embeddings = []    
for doc in documents: 
    embedding_list_current_doc = []

    for word in doc:
        idx_current_wrd = word_to_index_dict_SMALL__TestCell1[word]
        embdding_curr_wrd = index_to_embedding_list_SMALL__TestCell1[idx_current_wrd] 
        embedding_list_current_doc.append(embdding_curr_wrd) 
        
    embeddings.append(embedding_list_current_doc)


#b.  make embeddings a numpy array for use in an RNN 
embeddings_array = np.array(embeddings)

#c.  create training and test sets with Scikit Learn
from sklearn.model_selection import train_test_split

# Random splitting of the data in to training (80%) and test (20%)  
X_train, X_test, y_train, y_test = \
    train_test_split(embeddings_array, thumbs_down_up, test_size=0.20, 
                     random_state = RANDOM_SEED)


# Train and evaluate the model
reset_graph()

Construct_Graph()

new_rslt = Execute_Graph(mdl_name=Model_Nm)
t_end = time()


Accuracy_List.append(new_rslt)
RunTime_List.append( {'Model_Name': Model_Nm, 'RunTime':t_end - t_start} ) 












dfAccuracy1 = pd.DataFrame(Accuracy_List[0])
dfAccuracy2 = pd.DataFrame(Accuracy_List[1])
dfAccuracy3 = pd.DataFrame(Accuracy_List[2])
dfAccuracy4 = pd.DataFrame(Accuracy_List[3])
dfAccuracy5 = pd.DataFrame(Accuracy_List[4])
dfAccuracy6 = pd.DataFrame(Accuracy_List[5])

dfAccuracy = pd.concat([dfAccuracy1, dfAccuracy2, dfAccuracy3, dfAccuracy4, dfAccuracy5, dfAccuracy6], axis=0, ignore_index=True)


#plot the accuracy of training and test over epochs
i=0
fig = plt.figure(figsize=(20, 16))
for mdl_nm in dfAccuracy.Model_Name.unique():
    i += 1
    curr_ax = fig.add_subplot(3, 2, i)
    curr_ax.scatter( dfAccuracy[dfAccuracy.Model_Name == mdl_nm]['Epoch'], dfAccuracy[dfAccuracy.Model_Name == mdl_nm]['Acc_Train'])
    curr_ax.scatter( dfAccuracy[dfAccuracy.Model_Name == mdl_nm]['Epoch'], dfAccuracy[dfAccuracy.Model_Name == mdl_nm]['Acc_Test'])
    curr_ax.set_title(mdl_nm)
    curr_ax.set_xlabel('Epoch')
    curr_ax.set_ylim((0,1))
    curr_ax.set_ylabel('Accuracy')
    curr_ax.legend(frameon=False)
    
fig.savefig(wkg_dir + 'Scatterplots_of_Accuracy_100epochs.png', 
            bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
            orientation='portrait', papertype=None, format=None, 
            transparent=False, pad_inches=0.25, frameon=None)  
plt.show()




#plot the accuracy on the test dataset for the different models to compare them 
plt_dtls_dict = {'GloVe.Twitter.25d__10K_wrds':   {'color':'red', 'line_wgt': 1}, 
                 'GloVe.Twitter.25d__100K_wrds':  {'color':'red', 'line_wgt': 3}, 
                 'GloVe.Twitter.25d__1MM_wrds':  {'color':'red', 'line_wgt': 5}, 
                 'GloVe.Twitter.200d__10K_wrds':  {'color':'blue', 'line_wgt': 1}, 
                 'GloVe.Twitter.200d__100K_wrds': {'color':'blue', 'line_wgt': 3},   
                 'GloVe.Twitter.200d__1MM_wrds': {'color':'blue', 'line_wgt': 5}   }
fig = plt.figure(figsize=(18, 9))
curr_ax = fig.add_subplot(1, 2, 1)
for vocab_sz in ['10K_wrds', '100K_wrds', '1MM_wrds']:
    curr_ax.plot( dfAccuracy[dfAccuracy.Model_Name == 'GloVe.Twitter.25d__'+vocab_sz]['Epoch'], 
                  dfAccuracy[dfAccuracy.Model_Name == 'GloVe.Twitter.25d__'+vocab_sz]['Acc_Test'], 
                  label = vocab_sz, 
                  c=plt_dtls_dict['GloVe.Twitter.25d__'+vocab_sz]['color'], 
                  lw=plt_dtls_dict['GloVe.Twitter.25d__'+vocab_sz]['line_wgt']
                )
#plt.title('Accuracy on the Test Dataset')
curr_ax.set_title('GloVe.Twitter.25d')
curr_ax.set_xlabel('Epoch')
curr_ax.set_ylabel('Accuracy')
curr_ax.set_ylim((0,1))
curr_ax.legend(frameon=False)
#add the plot for the 200D embedding
curr_ax = fig.add_subplot(1, 2, 2)
for vocab_sz in ['10K_wrds', '100K_wrds', '1MM_wrds']:
    curr_ax.plot( dfAccuracy[dfAccuracy.Model_Name == 'GloVe.Twitter.200d__'+vocab_sz]['Epoch'], 
                  dfAccuracy[dfAccuracy.Model_Name == 'GloVe.Twitter.200d__'+vocab_sz]['Acc_Test'], 
                  label = vocab_sz, 
                  c=plt_dtls_dict['GloVe.Twitter.200d__'+vocab_sz]['color'], 
                  lw=plt_dtls_dict['GloVe.Twitter.200d__'+vocab_sz]['line_wgt']
                )
plt.suptitle('')
plt.title('Accuracy on the Test Dataset')
curr_ax.set_title('GloVe.Twitter.200d')
curr_ax.set_xlabel('Epoch')
curr_ax.set_ylabel('Accuracy')
curr_ax.set_ylim((0,1))
curr_ax.legend(frameon=False)
plt.suptitle('Accuracy on the Test Dataset')
    
fig.savefig(wkg_dir + 'Scatterplots_of_ModelAccuracy_Compared_100epochs.png', 
            bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
            orientation='portrait', papertype=None, format=None, 
            transparent=False, pad_inches=0.25, frameon=None)  
plt.show()


Accuracy_List[4]

print(RunTime_List[0])
print(RunTime_List[1])
print(RunTime_List[2])
print(RunTime_List[3])
print(RunTime_List[4])
print(RunTime_List[5])

#plt.suptitle('')
#plt.title('Boxplot of RMSE from evaluation folds')



#
#
## -----------------------------------------------------    
## Check on the embeddings list of list of lists 
## -----------------------------------------------------
## Show the first word in the first document
#test_word = documents[0][0]    
#print('\n\nFirst word in first document:', test_word)    
#print('Embedding for this word:\n', 
#      index_to_embedding_list_SMALL__TestCell1[word_to_index_dict_SMALL__TestCell1[test_word]])
#print('Corresponding embedding from embeddings list of list of lists\n',
#      embeddings[0][0][:])
#
## Show the tenth word in the seventh document  
#test_word = documents[6][9]    
#print('\n\nTenth word in seventh document:', test_word)    
#print('Embedding for this word:\n', 
#      index_to_embedding_list_SMALL__TestCell1[word_to_index_dict_SMALL__TestCell1[test_word]])
#print('Corresponding embedding from embeddings list of list of lists\n',
#      embeddings[6][9][:])
#
## Show the last word in the last document
#test_word = documents[999][39]    
#print('\n\nLast word in last document:', test_word)    
#print('Embedding for this word:\n', 
#      index_to_embedding_list_SMALL__TestCell1[word_to_index_dict_SMALL__TestCell1[test_word]])
#print('Corresponding embedding from embeddings list of list of lists\n',
#      embeddings[999][39][:])        
#








#get the list of files used
import os
dir_final_submission = 'C:/Users/ashle/Documents/Personal Data/Northwestern/2018_4 FALL  PREDICT 422 Machine Learning/wk8 - recurrent neural nets/'
filez = os.listdir(dir_final_submission)

for fl in filez:
    print(re.sub(',', '', fl))
    
