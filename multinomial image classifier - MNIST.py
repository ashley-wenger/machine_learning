# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 22:51:26 2018

@author: Ashley Wenger 
"""


##the original Scikit Learn program that obtained the data and created these comma-delimited text files
#from sklearn.datasets import fetch_mldata
#mnist = fetch_mldata('MNIST original')
#
#import pandas as pd
#
#mnist  # show structure of datasets Bunch object from Scikit Learn
#
## define arrays from the complete data set
#mnist_X, mnist_y = mnist['data'], mnist['target']
#
#mnist_X_df = pd.DataFrame(mnist_X)
#mnist_y_df = pd.DataFrame(mnist_y)
#
#mnist_X_df.to_csv('mnist_X.csv', sep = ',', header = True, index = False)
#mnist_y_df.to_csv('mnist_y.csv', sep = ',', header = True, index = False)





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 24

# although we standardize X and y variables on input,
# we will fit the intercept term in the models
# Expect fitted values to be close to zero
#SET_FIT_INTERCEPT = True


# modeling routines from Scikit Learn packages
from sklearn.metrics import mean_squared_error, r2_score  
from math import sqrt  # for root mean-squared error calculation
from sklearn.model_selection import train_test_split


pd.set_option('display.max_columns', 40) 
pd.set_option('display.max_rows', 100) 
pd.set_option('display.width', 100)


#set up time tracking
#---------------------
#set up a list to hold fn execution times, and a function to add new msgs to it.
log_list = []

def log_fn(msg):
    global log_list  #this tells the function to look for the "global" variable of this name, not to create a local one with this name. 
                        #global is w.r.t. the module, not all modules.
                    #if we didn't do this, when we wrote to this var, it wouldn't use the global var and the changes wouldn't be seen outside of the fn
    log_list.append(msg)


#create a decorator f'n that captures the start and end time of a fn that it decorates, and logs that  (in this case, adds an item to the log_list)
from time import time
from functools import wraps

def simple_time_tracker(log_fun):
    def _simple_time_tracker(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            start_time = time()

            try:
                result = fn(*args, **kwargs)
            finally:
                elapsed_time = time() - start_time

                # log the result
                log_fun({
                    'function_name': fn.__name__,
                    'total_time': elapsed_time,
                })
                
            return result

        return wrapped_fn
    return _simple_time_tracker

#end time tracking setup
    



#read in the predictor (X) vars
wkg_dir = 'C:/Users/ashle/Documents/Personal Data/Northwestern/2018_4 FALL  PREDICT 422 Machine Learning/wk5 - multinomial classifier/Wenger__MSDS422_Sec59_Assign5/'
dfMNIST_X = pd.read_csv(wkg_dir+'mnist_X.csv')
dfMNIST_y = pd.read_csv(wkg_dir+'mnist_y.csv')

dfMNIST_X.shape   #70,000 rows;   784 cols

dfMNIST_X.info()
#rows are indexed with #s 0 to 69,999
#cols are named (indexed) with #s 0 to 783

#look at the first 21 cols and the first 5 rows
#dfMNIST_X.iloc[:, 0:20].head(5)

#look at the first 21 rows, all cols
#dfMNIST_X[0:20]

#dfMNIST_X.columns.values

dfMNIST_y.shape
dfMNIST_y.info()
dfMNIST_y.describe()



# Utilize the first 60 thousand as a model development set and the final 10 thousand as a holdout test set. 
dfTrain_X, dfTest_X, dfTrain_y, dfTest_y = train_test_split(dfMNIST_X, dfMNIST_y, train_size=60000, shuffle=False)  
    #, random_state=RANDOM_SEED, stratify=???-- don't we want to have a good mix of each digit?)
    #I think this is the mistake!
    
dfTrain_X.shape  #( 60k rows as desired, all 784 cols)
dfTest_X.shape  #( 10k rows as desired, all 784 cols)
    
dfTrain_y.shape  #( 60k rows as desired, 1 col)
dfTest_y.shape  #( 10k rows as desired, 1 col)

#check the distros - a little worried that the lack of shuffling might give no 9's, etc
#dfTrain_y['0'].value_counts().sort_index()   # looks fine.   has all 10 digits, with roughly similar freq's
#dfTest_y['0'].value_counts().sort_index()  # looks fine.   has all 10 digits, with roughly similar freq's


#dfXval_descr = dfMNIST_X.describe()
#dfXval_descr.transpose().describe()
#the values in the predictor vars all range from 0 to 255 (global range, across all cols.   some cols are all 0s, some a mix, but all values overall are between 0 and 255
dfTrain_X = dfTrain_X.copy()
dfTrain_y = dfTrain_y.copy()
dfTest_X = dfTest_X.copy()
dfTest_y = dfTest_y.copy()

#give the y column in here a friendlier name that '0'
dfTrain_y.columns = ['y']
dfTest_y.columns = ['y']

dfTrain_y.info() 
dfTrain_y['y'].value_counts().sort_index()





# =========================================================================================================
#(1) Begin by fitting a random forest classifier using the full set of 784 explanatory variables and the model development set of 60,000 observations. 
#Record the time it takes to fit the model and evaluate the model on the holdout data. 
#Assess classification performance using the F1-score, which is the harmonic mean of precision and recall. 
# =========================================================================================================

from sklearn.ensemble import RandomForestClassifier

RndmForest_clf1 = RandomForestClassifier(n_estimators = 10, bootstrap=True, max_features='sqrt', random_state=RANDOM_SEED)

@simple_time_tracker(log_fn)
def fit_clf1():
    RndmForest_clf1.fit(dfTrain_X, dfTrain_y['y'])

#fit the function
for i in range(30):
    fit_clf1()

#review the time taken.
log_list


#dfTrain_y.head(50)
#dfTrain_y['correct_pred_clf1'] = ( dfTrain_y['y_pred_clf1'] == dfTrain_y['y'] ) 
#dfTrain_y[dfTrain_y.correct_pred_clf1 == True]  #59,938 of the predictions are correct.   serious overfitting???


#calculate precision manually
#============================
#precision = % of positive predictions that are actually correct.  = TP/(TP + FP)
    #for multinomial classification, I think this has to be calculated per class
dfTrain_y['y_pred_clf1'] = RndmForest_clf1.predict(dfTrain_X)

#get the precision, recall, and F1 score per class
#-------------------------------------------------
#precision and recall
precision_recall_list = []
for y_val in dfTrain_y.y.unique():
    tp = sum(  (dfTrain_y.y_pred_clf1 == y_val)  &  (dfTrain_y.y == y_val) )   #we predicted that value, and the prediction was right
    fp = sum(  (dfTrain_y.y_pred_clf1 == y_val)  &  (dfTrain_y.y != y_val) )   #we predicted that value, but the prediction was wrong
    fn = sum(  (dfTrain_y.y_pred_clf1 != y_val)  &  (dfTrain_y.y == y_val) )   #we didn't predicted that value, but we should have
    
    prcsn = tp / (tp + fp)   #how many of the positive predictions were correct
    recalll = tp / (tp + fn)   #how many of the true positives were correctly predicted
    
    precision_recall_list.append({'y_val':y_val, 'precision': prcsn, 'recall':recalll})


dfPrecision_Recall_by_class = pd.DataFrame(precision_recall_list)

#F1 score
def f1_score_akw(precision_, recall_):
    f1_score = 2 /  ( 1/precision_  +  1/recall_)
    return f1_score

dfPrecision_Recall_by_class['F1_score'] = dfPrecision_Recall_by_class.apply(lambda rw: f1_score_akw(rw['precision'], rw['recall']), axis=1) 

#check vs. algorithm
from sklearn.metrics import classification_report

clsfcn_rpt_dict = classification_report(dfTrain_y['y'], dfTrain_y['y_pred_clf1'], output_dict=True)
dfPrecision_Recall_by_class_chk = pd.DataFrame(clsfcn_rpt_dict)
dfrslts_4_graph_clf1_train = dfPrecision_Recall_by_class_chk.loc[['precision', 'recall', 'f1-score'], :].transpose()


#predict the results on the test set, and capture the time required
# =================================================================
#set up for timing
@simple_time_tracker(log_fn)
def predict_clf1():
    dfTest_y['y_pred_clf1'] = RndmForest_clf1.predict(dfTest_X)    

#predict the results
for i in range(30):
    predict_clf1()

#review the time
log_list



#review on the test set
clsfcn_rpt_dict__test = classification_report(dfTest_y['y'], dfTest_y['y_pred_clf1'], output_dict=True)
dfPrecision_Recall_by_class_test = pd.DataFrame(clsfcn_rpt_dict__test)

#review the key results
dfrslts_4_graph_clf1_test = dfPrecision_Recall_by_class_test.transpose()

#as compared to the training set, is a little less accurate but still very precise and very high recall.
#dfPrecision_Recall_by_class_chk.transpose()
    #worst on the 8's and 3's.   not surprising

markr_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

fig = plt.figure(figsize=(16, 6))
plt.title('F1 score on the test set, by class (i.e. digit)\nModel built on all features', size=14)
ax1 = fig.add_subplot(111)
ax1.bar(markr_list, dfrslts_4_graph_clf1_test.iloc[0:10, :]['f1-score'], color='blue') #, label='train', marker='$0$', s=70)
ax1.set_xlabel('Class (digit) to be identified', labelpad = 10, size=12)
ax1.set_ylabel('F1 score', labelpad = 10, size=12)
ax1.set_ylim(0,1)
plt.savefig(wkg_dir + '/F1_score__clf1.jpg', 
                bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                orientation='portrait', papertype=None, format=None, 
                transparent=False, pad_inches=0.25, frameon=None)  
plt.show()




#plot precision-recall scores of each class
#  use the digit as the symbol
#  color the training results one color, test results another color
fig = plt.figure(figsize=(16, 6))
plt.title('Precision vs Recall for each class (i.e. digit)\nModel built on all features', size=14)
ax1 = fig.add_subplot(111)
ax1.scatter(dfrslts_4_graph_clf1_train.loc['0.0', :]['recall'], dfrslts_4_graph_clf1_train.loc['0.0', :]['precision'], c='blue', label='train', marker='$0$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_train.loc['1.0', :]['recall'], dfrslts_4_graph_clf1_train.loc['1.0', :]['precision'], c='blue', label='train', marker='$1$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_train.loc['2.0', :]['recall'], dfrslts_4_graph_clf1_train.loc['2.0', :]['precision'], c='blue', label='train', marker='$2$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_train.loc['3.0', :]['recall'], dfrslts_4_graph_clf1_train.loc['3.0', :]['precision'], c='blue', label='train', marker='$3$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_train.loc['4.0', :]['recall'], dfrslts_4_graph_clf1_train.loc['4.0', :]['precision'], c='blue', label='train', marker='$4$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_train.loc['5.0', :]['recall'], dfrslts_4_graph_clf1_train.loc['5.0', :]['precision'], c='blue', label='train', marker='$5$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_train.loc['6.0', :]['recall'], dfrslts_4_graph_clf1_train.loc['6.0', :]['precision'], c='blue', label='train', marker='$6$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_train.loc['7.0', :]['recall'], dfrslts_4_graph_clf1_train.loc['7.0', :]['precision'], c='blue', label='train', marker='$7$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_train.loc['8.0', :]['recall'], dfrslts_4_graph_clf1_train.loc['8.0', :]['precision'], c='blue', label='train', marker='$8$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_train.loc['9.0', :]['recall'], dfrslts_4_graph_clf1_train.loc['9.0', :]['precision'], c='blue', label='train', marker='$9$', s=70)

ax1.scatter(dfrslts_4_graph_clf1_test.loc['0.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['0.0', :]['precision'], c='orange', label='test', marker='$0$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['1.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['1.0', :]['precision'], c='orange', label='test', marker='$1$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['2.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['2.0', :]['precision'], c='orange', label='test', marker='$2$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['3.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['3.0', :]['precision'], c='orange', label='test', marker='$3$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['4.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['4.0', :]['precision'], c='orange', label='test', marker='$4$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['5.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['5.0', :]['precision'], c='orange', label='test', marker='$5$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['6.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['6.0', :]['precision'], c='orange', label='test', marker='$6$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['7.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['7.0', :]['precision'], c='orange', label='test', marker='$7$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['8.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['8.0', :]['precision'], c='orange', label='test', marker='$8$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['9.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['9.0', :]['precision'], c='orange', label='test', marker='$9$', s=70)

ax1.set_xlabel('Recall', labelpad = 10, size=12)
ax1.set_ylabel('Precision', labelpad = 10, size=12)
fig.legend(loc='center right', frameon=False)
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
plt.savefig(wkg_dir + '/AccuracyComparedTestVsTrain_clf1.jpg', 
                bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                orientation='portrait', papertype=None, format=None, 
                transparent=False, pad_inches=0.25, frameon=None)  
plt.show()






#(2) Execute principal components analysis (PCA) on the full set of 70,000, generating principal components that represent 
#95 percent of the variability in the explanatory variables. The number of principal components in the solution should be 
#substantially fewer than the 784 explanatory variables. Record the time it takes to identify the principal components.
from sklearn.decomposition import PCA

@simple_time_tracker(log_fn)
def find_PCs():
    global prin_cmpnts
    prin_cmpnts = PCA(random_state=RANDOM_SEED)
    prin_cmpnts.fit(dfTrain_X)

#find the principal components
for i in range(30):
    find_PCs()

#review the time
log_list

#misc exploration of prin_cmpnts
#the one without the underscore returns the input param
#prin_cmpnts.n_components
#type(prin_cmpnts.n_components)   #type=NoneType ==> it is none, as expected from initialization call
#prin_cmpnts.n_features_
#prin_cmpnts.n_samples_  #60,000
#prin_cmpnts.n_components_   #784
#prin_cmpnts.explained_variance_ratio_
#prin_cmpnts.explained_variance_ratio_[0:9]
#END of misc exploration of prin_cmpnts




#get the # of prin components that cumulatively explain 95% of the variance in the original explanatory variables
# the prin components are ordered by descending explained_variance (PC1 is the new prin component that explains the most var; PC2 is the 2nd biggest, ...)
nbr_PCs_for_95pct = 0
cumttl_explnd_var = 0
for pc_nbr in range(prin_cmpnts.components_.shape[0]):
    cumttl_explnd_var += prin_cmpnts.explained_variance_ratio_[pc_nbr]
    if cumttl_explnd_var >= 0.95:
        print('The first ' + str(pc_nbr+1) + ' PCs explain ' + str(cumttl_explnd_var) + ' of the total variance.')
        nbr_PCs_for_95pct = pc_nbr
        break

#The first 154 PCs explain 0.9501960192613033 of the total variance.


#make a chart of var explained by PC and add a line at 95% cutoff level
fig = plt.figure(figsize=(16, 6))
plt.suptitle('Percent Variance Explained by Principal Component Number', size=14)
ax1 = fig.add_subplot(121)
ax1.scatter(range(1,len(prin_cmpnts.components_)+1), prin_cmpnts.explained_variance_ratio_)
ax1.set_xlabel('Principal Component Number', labelpad = 10, size=12)
ax1.set_ylabel('Percent of Variance Explained', labelpad = 10, size=12)

ax2 = fig.add_subplot(122)
ax2.scatter(range(1,len(prin_cmpnts.components_)+1), np.cumsum(prin_cmpnts.explained_variance_ratio_))
ax2.set_xlabel('Principal Component Number', labelpad = 10, size=12)
ax2.set_ylabel('Percent of Variance Explained, cumulative', labelpad = 10, size=12)
_xmin, _xmax = ax2.get_xlim()
_ymin, _ymax = ax2.get_ylim()
ax2.vlines(x=154, ymin=_ymin, ymax=_ymax, linestyle='dashed', color='grey')
ax2.hlines(y=0.95, xmin=_xmin, xmax=_xmax, linestyle='dashed', color='grey')
ax2.text(x=375, y=0.90, s='95% of variance explained') 
ax2.text(x=130, y=0.25, s='PC# = 154', rotation='vertical', verticalalignment='center') 
plt.savefig(wkg_dir + '/Explained_Var.jpg', 
                bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                orientation='portrait', papertype=None, format=None, 
                transparent=False, pad_inches=0.25, frameon=None)  
plt.show()


@simple_time_tracker(log_fn)
def transform_to_PCs():
    global dfTrain_X_PCA
    array_Train_X_PCA = prin_cmpnts.transform(dfTrain_X)
    dfTrain_X_PCA = pd.DataFrame(array_Train_X_PCA)

    #repeat for the test set
    global dfTest_X_PCA
    array_Test_X_PCA = prin_cmpnts.transform(dfTest_X)
    dfTest_X_PCA = pd.DataFrame(array_Test_X_PCA)

#transform to PC coordinates
for i in range(30):
    transform_to_PCs()

#review the time
log_list
    
#checks:
#dfTrain_X_PCA.info()
#check out the first and last 11 columns
#dfTrain_X_PCA.columns.values[0:10] , dfTrain_X_PCA.columns.values[-10:]



#(3) Using the identified principal components from step (2), use the first 60,000 observations to build another 
#random forest classifier. Record the time it takes to fit the model and to evaluate the model on the holdout data 
#(the last 10,000 observations). Assess classification performance using the F1-score, which is the harmonic mean of  precision and recall. 
RndmForest_clf2 = RandomForestClassifier(n_estimators = 10, bootstrap=True, max_features='sqrt', random_state=RANDOM_SEED)


@simple_time_tracker(log_fn)
def fit_clf2():
    RndmForest_clf2.fit( dfTrain_X_PCA.iloc[:, 0:nbr_PCs_for_95pct+1],    #use the first 154 columns of the transformed X vars (columns here are PC dimensions, not original dimensions)
                    dfTrain_y['y'])

#transform to PC coordinates
for i in range(30):
    fit_clf2()

#review the time
log_list



#predict the results for training set and test set
dfTrain_y['y_pred_pca'] = RndmForest_clf2.predict(  dfTrain_X_PCA.iloc[:, 0:nbr_PCs_for_95pct+1] )

@simple_time_tracker(log_fn)
def predict_clf2():
    dfTest_y['y_pred_pca'] = RndmForest_clf2.predict(  dfTest_X_PCA.iloc[:, 0:nbr_PCs_for_95pct+1]  )

for i in range(30):
    predict_clf2()

log_list


#get the F1 scores
#-----------------
#on the training set
clsfcn_rpt_dict__train_PCA = classification_report(dfTrain_y['y'], dfTrain_y['y_pred_pca'], output_dict=True)
dfScores_train_PCA = pd.DataFrame(clsfcn_rpt_dict__train_PCA)
dfrslts_4_graph_PCA_train = dfScores_train_PCA.transpose()
#dfrslts_4_graph_clf1_train

clsfcn_rpt_dict__test_PCA = classification_report(dfTest_y['y'], dfTest_y['y_pred_pca'], output_dict=True)
dfScores_test_PCA = pd.DataFrame(clsfcn_rpt_dict__test_PCA)
dfrslts_4_graph_PCA_test = dfScores_test_PCA.transpose()
#dfrslts_4_graph_clf1_test


#plot precision-recall scores of each class
#  use the digit as the symbol
#  color the training results one color, test results another color
fig = plt.figure(figsize=(16, 6))
plt.title('Precision vs Recall for each class (i.e. digit)\nModel built on PCs explaining 95% of total variance', size=14)
ax1 = fig.add_subplot(111)
ax1.scatter(dfrslts_4_graph_PCA_train.loc['0.0', :]['recall'], dfrslts_4_graph_PCA_train.loc['0.0', :]['precision'], c='blue', label='train', marker='$0$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_train.loc['1.0', :]['recall'], dfrslts_4_graph_PCA_train.loc['1.0', :]['precision'], c='blue', label='train', marker='$1$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_train.loc['2.0', :]['recall'], dfrslts_4_graph_PCA_train.loc['2.0', :]['precision'], c='blue', label='train', marker='$2$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_train.loc['3.0', :]['recall'], dfrslts_4_graph_PCA_train.loc['3.0', :]['precision'], c='blue', label='train', marker='$3$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_train.loc['4.0', :]['recall'], dfrslts_4_graph_PCA_train.loc['4.0', :]['precision'], c='blue', label='train', marker='$4$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_train.loc['5.0', :]['recall'], dfrslts_4_graph_PCA_train.loc['5.0', :]['precision'], c='blue', label='train', marker='$5$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_train.loc['6.0', :]['recall'], dfrslts_4_graph_PCA_train.loc['6.0', :]['precision'], c='blue', label='train', marker='$6$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_train.loc['7.0', :]['recall'], dfrslts_4_graph_PCA_train.loc['7.0', :]['precision'], c='blue', label='train', marker='$7$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_train.loc['8.0', :]['recall'], dfrslts_4_graph_PCA_train.loc['8.0', :]['precision'], c='blue', label='train', marker='$8$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_train.loc['9.0', :]['recall'], dfrslts_4_graph_PCA_train.loc['9.0', :]['precision'], c='blue', label='train', marker='$9$', s=70)

ax1.scatter(dfrslts_4_graph_PCA_test.loc['0.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['0.0', :]['precision'], c='orange', label='test', marker='$0$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['1.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['1.0', :]['precision'], c='orange', label='test', marker='$1$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['2.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['2.0', :]['precision'], c='orange', label='test', marker='$2$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['3.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['3.0', :]['precision'], c='orange', label='test', marker='$3$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['4.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['4.0', :]['precision'], c='orange', label='test', marker='$4$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['5.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['5.0', :]['precision'], c='orange', label='test', marker='$5$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['6.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['6.0', :]['precision'], c='orange', label='test', marker='$6$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['7.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['7.0', :]['precision'], c='orange', label='test', marker='$7$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['8.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['8.0', :]['precision'], c='orange', label='test', marker='$8$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['9.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['9.0', :]['precision'], c='orange', label='test', marker='$9$', s=70)

ax1.set_xlabel('Recall', labelpad = 10, size=12)
ax1.set_ylabel('Precision', labelpad = 10, size=12)
fig.legend(loc='center right', frameon=False)
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
plt.savefig(wkg_dir + '/AccuracyComparedTestVsTrain_PCA.jpg', 
                bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                orientation='portrait', papertype=None, format=None, 
                transparent=False, pad_inches=0.25, frameon=None)  
plt.show()



#(4) Compare test set performance across the two modeling approaches: original 784-variable model versus the 95-percent-PCA model. 
#Also evaluate the time required to perform (1) versus the time required to perform (2) and (3) together.  
#Ensure that accurate measures are made of the total time it takes to execute each of the modeling approaches in training the models. 
#Some guidance on the coding of benchmark studies may be found in the Python code under 


#plot precision-recall scores of each class, scoring on the test sets, for the "all features" model vs. the "PCA-top95%" model
#  use the digit as the symbol
#  color the training results one color, test results another color
fig = plt.figure(figsize=(13, 6))
plt.title('Precision vs Recall for each class (i.e. digit\nOn the test dataset)', size=14)
ax1 = fig.add_subplot(111)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['0.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['0.0', :]['precision'], c='blue', label='All_Features model', marker='$0$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['1.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['1.0', :]['precision'], c='blue', label='All_Features model', marker='$1$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['2.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['2.0', :]['precision'], c='blue', label='All_Features model', marker='$2$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['3.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['3.0', :]['precision'], c='blue', label='All_Features model', marker='$3$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['4.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['4.0', :]['precision'], c='blue', label='All_Features model', marker='$4$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['5.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['5.0', :]['precision'], c='blue', label='All_Features model', marker='$5$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['6.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['6.0', :]['precision'], c='blue', label='All_Features model', marker='$6$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['7.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['7.0', :]['precision'], c='blue', label='All_Features model', marker='$7$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['8.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['8.0', :]['precision'], c='blue', label='All_Features model', marker='$8$', s=70)
ax1.scatter(dfrslts_4_graph_clf1_test.loc['9.0', :]['recall'], dfrslts_4_graph_clf1_test.loc['9.0', :]['precision'], c='blue', label='All_Features model', marker='$9$', s=70)

ax1.scatter(dfrslts_4_graph_PCA_test.loc['0.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['0.0', :]['precision'], c='orange', label='PCA model', marker='$0$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['1.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['1.0', :]['precision'], c='orange', label='PCA model', marker='$1$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['2.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['2.0', :]['precision'], c='orange', label='PCA model', marker='$2$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['3.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['3.0', :]['precision'], c='orange', label='PCA model', marker='$3$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['4.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['4.0', :]['precision'], c='orange', label='PCA model', marker='$4$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['5.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['5.0', :]['precision'], c='orange', label='PCA model', marker='$5$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['6.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['6.0', :]['precision'], c='orange', label='PCA model', marker='$6$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['7.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['7.0', :]['precision'], c='orange', label='PCA model', marker='$7$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['8.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['8.0', :]['precision'], c='orange', label='PCA model', marker='$8$', s=70)
ax1.scatter(dfrslts_4_graph_PCA_test.loc['9.0', :]['recall'], dfrslts_4_graph_PCA_test.loc['9.0', :]['precision'], c='orange', label='PCA model', marker='$9$', s=70)

ax1.set_xlabel('Recall', labelpad = 10, size=12)
ax1.set_ylabel('Precision', labelpad = 10, size=12)
fig.legend(loc=(0.1, 0.15), frameon=False)
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
plt.savefig(wkg_dir + '/AccuracyCompared_byModel.jpg', 
                bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                orientation='portrait', papertype=None, format=None, 
                transparent=False, pad_inches=0.25, frameon=None)  
plt.show()









dfRunTimes = pd.DataFrame(log_list)
dfRunTimes['process'] = dfRunTimes.apply(lambda rww:  'AllFeatures' if rww['function_name'] in ['fit_clf1','predict_clf1'] else 'PCA_First154', axis=1)

RunTimes_list = []
#get the mean and st dev by process
    #should have recorded an iter #s to sum these up that way.  Don't want to re-run for time so reorganize to add that
#get the run time for all steps in the process1 iterations
for i, fit_time, prdct_time in zip(range(30),
                                   dfRunTimes[dfRunTimes.function_name == 'fit_clf1'].total_time,
                                   dfRunTimes[dfRunTimes.function_name == 'predict_clf1'].total_time):
    RunTimes_list.append({'process':'AllFeatures', 'runnbr':i, 'time_all_steps':fit_time + prdct_time})

#get the run time for all steps in the process2 iterations
for i, fnd_time, xform_time, fit_time, prdct_time in zip(range(30),
                                   dfRunTimes[dfRunTimes.function_name == 'find_PCs'].total_time,
                                   dfRunTimes[dfRunTimes.function_name == 'transform_to_PCs'].total_time,
                                   dfRunTimes[dfRunTimes.function_name == 'fit_clf2'].total_time,
                                   dfRunTimes[dfRunTimes.function_name == 'predict_clf2'].total_time):
    RunTimes_list.append({'process':'PCA_First154', 'runnbr':i, 'time_all_steps':fnd_time + xform_time + fit_time + prdct_time})

dfRunTimes2= pd.DataFrame(RunTimes_list)
dfRunTimes2[['process', 'time_all_steps']].groupby('process').aggregate(['mean', 'std'])

#             time_all_steps          
#                       mean       std
#process                              
#AllFeatures        3.996871  0.405803
#PCA_First154      18.809628  0.309036


#get the mean and st dev by step
dfRunTimes.groupby('function_name').aggregate(['mean', 'std'])


#                 total_time          
#                       mean       std
#function_name                        
#find_PCs           7.837609  0.276949
#fit_clf1           3.936477  0.405466
#fit_clf2           9.000915  0.143434
#predict_clf1       0.060394  0.006851
#predict_clf2       0.035408  0.007335
#transform_to_PCs   1.935695  0.100439


#(5) The experiment we have proposed has a design flaw. Identify the flaw. Fix it. 
#And rerun the experiment in a way that is consistent with a training-and-test regimen.

#a)  repeat for model 1 (all-features model)
dfTrain2_X, dfTest2_X, dfTrain2_y, dfTest2_y = train_test_split(dfMNIST_X, dfMNIST_y, train_size=60000, shuffle=True)  
dfTrain2_X = dfTrain2_X.copy()
dfTrain2_y = dfTrain2_y.copy()
dfTest2_X = dfTest2_X.copy()
dfTest2_y = dfTest2_y.copy()
dfTrain2_y.columns = ['y']
dfTest2_y.columns = ['y']

RndmForest_clf2 = RandomForestClassifier(n_estimators = 10, bootstrap=True, max_features='sqrt', random_state=RANDOM_SEED)
RndmForest_clf2.fit(dfTrain2_X, dfTrain2_y['y'])
dfTrain2_y['y_pred_clf1'] = RndmForest_clf2.predict(dfTrain2_X)
dfTest2_y['y_pred_clf1'] = RndmForest_clf2.predict(dfTest2_X)    

clsfcn_rpt_dict2 = classification_report(dfTrain2_y['y'], dfTrain2_y['y_pred_clf1'], output_dict=True)
dfPrecision_Recall_by_class2 = pd.DataFrame(clsfcn_rpt_dict2)
dfrslts_4_graph_clf2_train = dfPrecision_Recall_by_class2.loc[['precision', 'recall', 'f1-score'], :].transpose()

clsfcn_rpt_dict__test2 = classification_report(dfTest2_y['y'], dfTest2_y['y_pred_clf1'], output_dict=True)
dfPrecision_Recall_by_class_test2 = pd.DataFrame(clsfcn_rpt_dict__test2)
dfrslts_4_graph_clf2_test = dfPrecision_Recall_by_class_test2.transpose()




#b) PC model
prin_cmpnts2 = PCA(random_state=RANDOM_SEED)
prin_cmpnts2.fit(dfTrain2_X)

nbr_PCs_for_95pct2 = 0
cumttl_explnd_var2 = 0
for pc_nbr in range(prin_cmpnts2.components_.shape[0]):
    cumttl_explnd_var2 += prin_cmpnts2.explained_variance_ratio_[pc_nbr]
    if cumttl_explnd_var2 >= 0.95:
        print('The first ' + str(pc_nbr+1) + ' PCs explain ' + str(cumttl_explnd_var2) + ' of the total variance.')
        nbr_PCs_for_95pct2 = pc_nbr
        break
#The first 154 PCs explain 0.9504222464804126 of the total variance.
        
array_Train2_X_PCA = prin_cmpnts2.transform(dfTrain2_X)
dfTrain2_X_PCA = pd.DataFrame(array_Train2_X_PCA)

array_Test2_X_PCA = prin_cmpnts2.transform(dfTest2_X)
dfTest2_X_PCA = pd.DataFrame(array_Test2_X_PCA)


RndmForest_clf4 = RandomForestClassifier(n_estimators = 10, bootstrap=True, max_features='sqrt', random_state=RANDOM_SEED)
RndmForest_clf4.fit( dfTrain2_X_PCA.iloc[:, 0:nbr_PCs_for_95pct2+1],    #use the first 154 columns of the transformed X vars (columns here are PC dimensions, not original dimensions)
                    dfTrain2_y['y'])

dfTrain2_y['y_pred_pca'] = RndmForest_clf4.predict(  dfTrain2_X_PCA.iloc[:, 0:nbr_PCs_for_95pct2+1] )
dfTest2_y['y_pred_pca'] = RndmForest_clf4.predict(  dfTest2_X_PCA.iloc[:, 0:nbr_PCs_for_95pct2+1]  )


clsfcn_rpt_dict__train_PCA2 = classification_report(dfTrain2_y['y'], dfTrain2_y['y_pred_pca'], output_dict=True)
dfScores_train_PCA2 = pd.DataFrame(clsfcn_rpt_dict__train_PCA2)
dfrslts_4_graph_PCA_train2 = dfScores_train_PCA2.transpose()
#dfrslts_4_graph_clf1_train

clsfcn_rpt_dict__test_PCA2 = classification_report(dfTest2_y['y'], dfTest2_y['y_pred_pca'], output_dict=True)
dfScores_test_PCA2 = pd.DataFrame(clsfcn_rpt_dict__test_PCA2)
dfrslts_4_graph_PCA_test2 = dfScores_test_PCA2.transpose()



#create PCs using all 70k records
prin_cmpnts3 = PCA(random_state=RANDOM_SEED)
prin_cmpnts3.fit(dfMNIST_X)

nbr_PCs_for_95pct3 = 0
cumttl_explnd_var3 = 0
for pc_nbr in range(prin_cmpnts3.components_.shape[0]):
    cumttl_explnd_var3 += prin_cmpnts3.explained_variance_ratio_[pc_nbr]
    if cumttl_explnd_var3 >= 0.95:
        print('The first ' + str(pc_nbr+1) + ' PCs explain ' + str(cumttl_explnd_var3) + ' of the total variance.')
        nbr_PCs_for_95pct3 = pc_nbr
        break
#The first 154 PCs still explain 95.0%  (0.9503499702078614) of the total variance.






import os
dir_final_submission = 'C:/Users/ashle/Documents/Personal Data/Northwestern/2018_4 FALL  PREDICT 422 Machine Learning/wk5 - multinomial classifier/Wenger__MSDS422_Sec59_Assign5'
os.listdir(dir_final_submission)

