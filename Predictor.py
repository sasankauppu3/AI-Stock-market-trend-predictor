import numpy
import csv
import urllib
from sklearn import svm
from sklearn import linear_model
from sklearn import tree
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from yahoo_finance import Share
from pprint import pprint
import os
import warnings
warnings.filterwarnings("ignore")
    
def high(t, X):
    return max(X[:-t])

def low(t, X):
    return min(X[:-t])

def extract_features(data):
    
    data = data[:, [0, 1, 2, 3, 5]]
    features = data[:-1,:] - data[1:, :]
    Phigh = high(5, data[:, 1]) #highest of highs
    Plow = low(5, data[:, 2]) #lowest of lows
    vhigh = high(5, data[:, 4]) #highestclose
    vlow = low(5, data[:, 4]) #lowestclose
    
    openingprice_diff_by_highlow = features[:, 0]/ float(Phigh - Plow)
    closing_price_diff_by_highlow = features[:, 3]/float(Phigh - Plow)
    mov_avg_by_data = list()
    for i in range(len(features)):
        mov_avg_by_data.append(numpy.mean(data[:i+1, :], axis = 0)/data[i, :])
    mov_avg_by_data = numpy.array(mov_avg_by_data)
    features = data[:-1,[0]] - data[1:,[0]]
    features = preprocessing.normalize(features, norm='l2')

    if data.shape[1]==7:
        j_f=data[:,[6]]
        j_f=j_f[1:]
        j_f=numpy.array(j_f)
        features = numpy.column_stack((features, openingprice_diff_by_highlow, closing_price_diff_by_highlow, mov_avg_by_data,j_f))
    else:
        features = numpy.column_stack((features, openingprice_diff_by_highlow, closing_price_diff_by_highlow, mov_avg_by_data))
            
    return features



def move_days_forward(data, days):
    labels = ((data[days:, 3] - data[days:, 0]) > 0).astype(int)
    data = data[:-days, :]
    return data, labels
    
def predict(data):
    Accuracy_scores=list()
    data = numpy.array(data)    
    data = data.astype(float)
    Volume=data[:,5]
    labels = ((data[:, 3] - data[:, 0]) > 0).astype(int) #profit or loss label
    
    
    data,labels = move_days_forward(data,1) #we move the data by 1 day so that today's
                                            #stock is dependent on yesterday's value

    features = extract_features(data)

 
    ##adding extra features from talib adjust the hidden layers in Mlp in case of less accuracy
    ## uncomment the below code to aactivate the features
    '''
    import talib
    F_ADX=talib.ADX(numpy.asarray(data[:,1]),numpy.asarray(data[:,2]), numpy.asarray(data[:,5])  
    F_ADX = preprocessing.normalize(F_ADX, norm='l2')
    features.append(F_ADX)
    
    F_MFI=talib.MFI(np.asarray(data[:,1]),np.asarray(data[:,2]), np.asarray(data[:,5]),np.asarray(Volume))
    F_MFI = preprocessing.normalize(F_MFI, norm='l2')
    features.append(F_MFI)
    
    F_RSI=talib.RSI(np.asarray(data[:,5]))
    F_RSI = preprocessing.normalize(F_RSI, norm='l2')
    features.append(F_RSI)
    '''        
    ##500 for CSV's which is approx 25%data for testing###
    train_features = features[500:]
    test_features = features[:500]

    train_labels = labels[500:-1]
    test_labels = labels[:500]

    C_Val=[ 10e-1, 10e0, 10e1, 10e2]
    Gamma_Val =[ 10e-2, 10e-1, 10e0, 10e1]

    #################### SVM-RBF KERNEL ############################     
    max_acc=0
    max_c=0
    max_g=0
    for c in C_Val:
        for g in Gamma_Val:
            clf = svm.SVC(kernel = 'rbf', C = c, gamma = g).fit(train_features, train_labels) 
            curr_acc = accuracy_score(test_labels, clf.predict(test_features))
            if curr_acc>max_acc:
                max_acc=curr_acc
                max_g=g
                max_c=c
                
    
    clf = svm.SVC(kernel = 'rbf', C = max_c, gamma = max_g).fit(train_features, train_labels)
    best_predicted=clf.predict(test_features)
    best_acc = accuracy_score(test_labels,best_predicted)
    Accuracy_scores.append(best_acc)
    #OtherScores = precision_recall_fscore_support(test_labels, predicted)
    #print "RBF Accuracy: ",best_acc," for C=",max_c,"and G=",max_g          

    '''
    step = numpy.arange(0, len(test_labels))
    plt.subplot(211)
    plt.xlim(-1, len(test_labels) + 1)
    plt.ylim(-1, 2)
    plt.ylabel('Actual Values')
    plt.plot(step, test_labels, drawstyle = 'step')
    plt.subplot(212)
    plt.xlim(-1, len(test_labels) + 1)
    plt.ylim(-1, 2)
    plt.xlabel('Days')
    plt.ylabel('Predicted Values')
    plt.plot(step, best_predicted, drawstyle = 'step')
    plt.show()
    '''    

    ########################### Decision Trees #############################
    clf = tree.DecisionTreeClassifier(min_samples_split=80,min_impurity_split=1e-5)
    clf.fit(train_features, train_labels)

    predicted = clf.predict(test_features)
    Accuracy = accuracy_score(test_labels, predicted)
    #print "Decision tree Accuracy: ", Accuracy
    Accuracy_scores.append(Accuracy)
    '''
    step = numpy.arange(0, len(test_labels))
    plt.subplot(211)
    plt.xlim(-1, len(test_labels) + 1)
    plt.ylim(-1, 2)
    plt.ylabel('Actual Values')
    plt.plot(step, test_labels, drawstyle = 'step')
    plt.subplot(212)
    plt.xlim(-1, len(test_labels) + 1)
    plt.ylim(-1, 2)
    plt.xlabel('Days')
    plt.ylabel('Predicted Values')
    plt.plot(step, best_predicted, drawstyle = 'step')
    plt.show()
    '''
    ################################ Neural Net ################################# 
    clf = MLPClassifier(solver='lbfgs', activation='relu',alpha=1e-5,hidden_layer_sizes=(3,2,7),max_iter=200,random_state=1)
    clf.fit(train_features, train_labels)

    predicted = clf.predict(test_features)
    Accuracy = accuracy_score(test_labels, predicted)
    #print "MLP Accuracy: ", Accuracy
    Accuracy_scores.append(Accuracy)
    '''
    step = numpy.arange(0, len(test_labels))
    plt.subplot(211)
    plt.xlim(-1, len(test_labels) + 1)
    plt.ylim(-1, 2)
    plt.ylabel('Actual Values')
    plt.plot(step, test_labels, drawstyle = 'step')
    plt.subplot(212)
    plt.xlim(-1, len(test_labels) + 1)
    plt.ylim(-1, 2)
    plt.xlabel('Days')
    plt.ylabel('Predicted Values')
    plt.plot(step, predicted, drawstyle = 'step')
    plt.show()
    '''

    ################################ Logistic Regression ################################# 
    max_acc=0
    max_c=0
    max_g=0
    C_Val=[10e-7]
    for c in C_Val:
        clf = linear_model.LogisticRegression(C=c).fit(train_features, train_labels)
        curr_acc=accuracy_score(test_labels, clf.predict(test_features))
        if curr_acc>max_acc:
            max_acc=curr_acc
            max_c=c
    
    clf = linear_model.LogisticRegression(C=max_c).fit(train_features, train_labels)
    best_predicted=clf.predict(test_features)
    best_accuracy=accuracy_score(test_labels,best_predicted)
    #print "Logistic Regression Accuracy: ",best_accuracy," for C=",max_c          
    Accuracy_scores.append(best_accuracy)

    '''
    step = numpy.arange(0, len(test_labels))
    plt.subplot(211)
    plt.xlim(-1, len(test_labels) + 1)
    plt.ylim(-1, 2)
    plt.ylabel('Actual Values')
    plt.plot(step, test_labels, drawstyle = 'step')
    plt.subplot(212)
    plt.xlim(-1, len(test_labels) + 1)
    plt.ylim(-1, 2)
    plt.xlabel('Days')
    plt.ylabel('Predicted Values')
    plt.plot(step, best_predicted, drawstyle = 'step')
    plt.show()
    '''
    return Accuracy_scores



#################################################################################################################
#################################################################################################################
#################################################################################################################


############ USE IT FOR USING AVAILABLE DATA FROM CSV's ###############
urls = ['data/Googl.csv',
       'data/Aapl.csv',
       'data/Msft.csv',
       'data/AMZN.csv']
script_dir = os.path.dirname(__file__)


Net_Accuracy=list()
for url in urls:
    url = os.path.join(script_dir, url)
    data=list()
    #print (url[43:-4])
    with open(url, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    data = numpy.array(data)
    data = data[1:, 1:]
    Net_Accuracy.append(predict(data))
algos=["SVM-RBF", "Decision Trees", "MLP","Logistic-Regression"]
Net_Accuracy=numpy.array(Net_Accuracy)
Net_Accuracy=numpy.mean(Net_Accuracy, axis = 0)
for i in range(0,4):
    print algos[i],Net_Accuracy[i]

'''
################### USE IT FOR USING LIVE DATA #####################
stocks = [Share('Googl'),Share('Aapl'),Share('Msft'),Share('AMZN')]
Net_Accuracy=list()
for stock in stocks:
    data=list()
    for attr in stock.get_historical('2010-12-9', '2016-12-13'):
        row=list()
        row.append(attr['Open'])
        row.append(attr['High'])
        row.append(attr['Low'])
        row.append(attr['Close'])
        row.append(attr['Volume'])
        row.append(attr['Close'])
        data.append(row)
        
    Net_Accuracy.append(predict(data))
algos=["SVM-RBF", "Decision Trees", "MLP","Logistic-Regression"]
Net_Accuracy=numpy.array(Net_Accuracy)
Net_Accuracy=numpy.mean(Net_Accuracy, axis = 0)
for i in range(0,4):
    print algos[i],Net_Accuracy[i]

'''
