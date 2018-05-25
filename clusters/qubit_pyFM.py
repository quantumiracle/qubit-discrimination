#coding=utf-8
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
import pickle
import gzip
import numpy as np
f =gzip.open('./DetectionBinsData_pickle_clean.gzip','rb') 

'''classification model'''
#clf = SGDClassifier(max_iter=10000)
#clf = svm.SVC()
#model = LogisticRegression()   # %98.5%
#model = GaussianNB()   # 95.7%
#model = KNeighborsClassifier()  %92.37
#model = DecisionTreeClassifier()  %97.33
fm = pylibfm.FM(num_factors=50, num_iter=40, verbose=True, task="classification", initial_learning_rate=0.01, learning_rate_schedule="optimal")

#learn step
X=[]
Y=[]
batch_num=600
for i in range (batch_num):   #training times
    print(i)
    data=pickle.load(f)
    d=data[:,1:101]
    b=data[:,102:]
    batch=[]
    batch.append(d)
    batch.append(b)
    #print(batch)
    train_batch=np.array(batch).reshape(200,100)
    #print(train_batch) 
    train_Y=100*[0]+100*[1]
    #print(Y)
    X.append(train_batch)
    Y.append(train_Y)
    
    
X=np.array(X).reshape(batch_num*200,100)
Y=np.array(Y).reshape(batch_num*200)
#pyFM
#https://github.com/coreylynch/pyFM
X_train = [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in X]
v = DictVectorizer()
X_train = v.fit_transform(X_train)
fm.fit(X_train, Y)  

#test step
error_d=0
error_b=0
for i in range(100):
    test=pickle.load(f)
    d=test[:,1:101]
    b=test[:,102:]
    d_test = [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in d]
    b_test = [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in b]
    d_test=v.transform(d_test)
    b_test=v.transform(b_test)  
    result_d = fm.predict(d_test) # 0 for dark; 1 for bright
    result_b = fm.predict(b_test) # 0 for dark; 1 for bright
    #print(result_d, '\n', result_b)
    
    for j in range(100):
        if result_d[j]>0.5:
            error_d+=1
        if result_b[j]<=0.5:
            error_b+=1
print('dark error counts:', error_d)
print('bright error counts:', error_b)
accuracy_rate=1-(float)(error_b+error_d)/20000.0
print('total accuracy rate:',accuracy_rate)   #95.9%
f.close()