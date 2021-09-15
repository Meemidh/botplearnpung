import json, csv,codecs
import pandas as pd
import numpy as np
import string 
from pythainlp.corpus import thai_stopwords
from pythainlp import word_tokenize, Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report


def text_process(message):
    list_stopwords = ['ค่ะ','ครับ','วันนี้','ใดๆ','ด้วย','คับ','ผม','นาย','เมื่อ',
                'เพราะ','จัด','เทอ','เธอ','จ้า','จัง','การ','กระ','ไง','แต่'
                'ค้าบ','บ้าง','อย่าง','อยู่','ค่า','คือ','ป่ะ','จริง','ยังไง','ขณะ',
                'นำ','กัน','ใต้','ตอนนี้','วันนั้น','อย่างไร','เกี่ยว',
                'หน่อย','เช่น','ขณะนี้','พวกเรา','พวกฉัน','ช่วง','ต่างๆ','ข้างๆ',
                'ใน','คุณ','แก่','นั้น','ฉัน','ชั้น','คะ','คำ','พรุ่งนี้','เมื่อวาน','มะรืน','วัน','นะ','ไหน','ไหม','อาหาร','เครื่องดื่ม']
    
    mess = "".join(u for u in message if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))  
    mess = word_tokenize(mess)
    mess = " ".join(u for u in mess)
    mess = " ".join(u for u in mess.split() 
                if u not in list_stopwords)    
    return mess



def createKnnModel(x_train,y_train,k):
    #!หาความถี่คำ
    cvec = CountVectorizer(analyzer=lambda x:x.split(' '))
    cvec.fit_transform(x_train) #vocab
    #print(cvec.vocabulary_)

    #!แสดงตารางความถี่
    train_bow = cvec.transform(x_train)
    bow = pd.DataFrame(train_bow.toarray(), columns=cvec.get_feature_names(), index=x_train)

    #!หาidf_weights
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
    tfidf_transformer.fit(bow)

    #!หาค่าTf-tdf
    count_vector=cvec.transform(x_train) 
    tfidf_vector_train=tfidf_transformer.transform(count_vector)
    feature_names = cvec.get_feature_names() 
    tfidf = pd.DataFrame(tfidf_vector_train.T.todense(), index=feature_names) 
    # print(tfidf)
    
    #!ฝึกโมเดลด้วยK-NN
    knn = KNeighborsClassifier(n_neighbors = k) 
    knn.fit(tfidf_vector_train,y_train)
    
    return knn,cvec,tfidf_transformer

#!---------------------------------------------------------------------####

def createSVMModel(x_train,y_train,c):
    #!หาความถี่คำ
    cvec = CountVectorizer(analyzer=lambda x:x.split(' '))
    cvec.fit_transform(x_train) #vocab
    #print(cvec.vocabulary_)

    #!แสดงตารางความถี่
    train_bow = cvec.transform(x_train)
    bow = pd.DataFrame(train_bow.toarray(), columns=cvec.get_feature_names(), index=x_train)

    #!หาidf_weights
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
    tfidf_transformer.fit(bow)

    #!หาค่าTf-tdf
    count_vector=cvec.transform(x_train) 
    tfidf_vector_train=tfidf_transformer.transform(count_vector)
    feature_names = cvec.get_feature_names() 
    tfidf = pd.DataFrame(tfidf_vector_train.T.todense(), index=feature_names) 

    #!ฝึกโมเดลด้วย SVM
    clf = svm.SVC(C = c)
    clf.fit(tfidf_vector_train,y_train)
    
    return clf,cvec,tfidf_transformer

#!----------------------------------------------------------------------------------------------------

#!testing process
def TestingProcess(x_test,knn,cvec,tfidf_transformer):
    count_vector_test = cvec.transform(x_test)
    tfidf_vector_test = tfidf_transformer.transform(count_vector_test)
    my_predictions = knn.predict(tfidf_vector_test)
    
    return my_predictions


def TestingProcessSVM(x_test,clf,cvec,tfidf_transformer):
    count_vector_test = cvec.transform(x_test)
    tfidf_vector_test = tfidf_transformer.transform(count_vector_test)
    my_predictions = clf.predict(tfidf_vector_test)

    return my_predictions


#!-------------------------------------------------------------------------------------------------


def evaluation_classification(y_test,my_predictions):
    y_test = np.array(y_test)
    cfm = confusion_matrix(y_test,my_predictions)
    report = classification_report(y_test,my_predictions)
    
    return report,cfm





# np.sum(y_test == 1)
