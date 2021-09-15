import json, csv,codecs
import pandas as pd
import numpy as np
import string 
import pickle
from pythainlp.corpus import thai_stopwords
from pythainlp import word_tokenize, Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import svm
from My_function import text_process,evaluation_classification,createSVMModel,TestingProcessSVM
import dill


data = pd.read_csv('DataSet_label.csv', sep=',', names=['message','sentiment'],header=None,encoding='utf-8')
data['message'] = data['message'].apply(text_process)

#!---------------------------------------------------------------------------------------------------------------

#!แบ่งข้อมูลเทรนเทส
x = data['message']
y = data['sentiment']

n_data = len(x)
n_fold = 1
size_fold = int(n_data/n_fold) 

y_data = len(y)
y_fold = 1
ysize_fold = int(y_data/y_fold) 

#!ใช้ข้อมูลทั้งหมดเทรน 

x_train = x[0:size_fold]
x_test = x[0:size_fold]

y_train = y[0:ysize_fold]
y_test = y[0:ysize_fold]


#!แบ่ง Fold ในการเทรน
##1
# x_test = x[0:size_fold]
# x_train = x[size_fold:5*size_fold]

# y_test = y[0:ysize_fold]
# y_train = y[ysize_fold:5*ysize_fold]

# #!-----------------------------------------------------------
##2
# x_test = x[size_fold:2*size_fold]
# x_train = x[0:size_fold]
# x_train = x_train.append(x[2*size_fold:5*size_fold])

# y_test = y[ysize_fold:2*ysize_fold]
# y_train = y[0:ysize_fold]
# y_train = y_train.append(y[2*ysize_fold:5*ysize_fold])

# #!-----------------------------------------------------------
# ##3
# x_test = x[2*size_fold:3*size_fold]
# x_train = x[0:2*size_fold]
# x_train = x_train.append(x[3*size_fold:5*size_fold])

# y_test = y[2*ysize_fold:3*ysize_fold]
# y_train = y[0:2*ysize_fold]
# y_train = y_train.append(y[3*ysize_fold:5*ysize_fold])

# #!-----------------------------------------------------------
# ##4
# x_test = x[3*size_fold:4*size_fold]
# x_train = x[0:3*size_fold]
# x_train = x_train.append(x[4*size_fold:5*size_fold])

# y_test = y[3*ysize_fold:4*ysize_fold]
# y_train = y[0:3*ysize_fold]
# y_train = y_train.append(y[4*ysize_fold:5*ysize_fold])

# #!-----------------------------------------------------------
# ##5
# x_test = x[4*size_fold:5*size_fold]
# x_train = x[0:4*size_fold] 

# y_test = y[4*ysize_fold:5*ysize_fold]
# y_train = y[0:4*ysize_fold] 

#!-----------------------------------------------------------

# #!ฝึกโมเดลด้วยSVM
# clf = svm.SVC()
# clf.fit(tfidf_vector_train,y_train)
# svm.SVC(C=0.1)

# model_svm,cvec,tfidf_transformer = createSVMModel(x_train,y_train,10)
# my_predictions = TestingProcessSVM(x_test,model_svm,cvec,tfidf_transformer)
# report,cfm = evaluation_classification(y_test,my_predictions)


# print(my_predictions)
# print(cfm)
# print(y_test)

#!--------------------------------------------------------------------------------------------------


#!SVM Model C=10
sentiment_model,cvec,tfidf_transformer = createSVMModel(x_train,y_train,10)
#!Save Model
pickle.dump(sentiment_model,open("sentiment_model.model",'wb'))
dill.dump(cvec,open("cvec_model.model",'wb'))
pickle.dump(tfidf_transformer,open("tfidf_transformer_model.model",'wb'))

my_predictions = TestingProcessSVM(x_test,sentiment_model,cvec,tfidf_transformer)

# print(report)














