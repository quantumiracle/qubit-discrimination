#coding=utf-8
from sklearn import svm
from sklearn.linear_model import SGDClassifier
import pickle
import gzip
import numpy as np
f =gzip.open('./DetectionBinsData_pickle_compact.gzip','rb') 
#clf = SGDClassifier(max_iter=10000)
clf = svm.SVC()

#learn step
X=[]
Y=[]
batch_num=6
num_bins=10
for i in range (batch_num):   #training times
    print(i)
    data=np.array(pickle.load(f))
    d=data[:,:num_bins]
    b=data[:,num_bins:]
    print(d)
    batch=[]
    batch.append(d)
    batch.append(b)
    #print(batch)
    train_batch=np.array(batch).reshape(2*num_bins,100)
    #print(train_batch) 
    train_Y=100*[0]+100*[1]
    #print(Y)
    X.append(train_batch)
    Y.append(train_Y)
    
    
X=np.array(X).reshape(batch_num*200,num_bins)
Y=np.array(Y).reshape(batch_num*200)
#http://scikit-learn.org/stable/modules/scaling_strategies.html#incremental-learning
clf.fit(X, Y)  

#test step
error_d=0
error_b=0
samples=30
for i in range(samples*100):
    test=np.array(pickle.load(f))
    d=test[:,:num_bins]
    b=test[:,num_bins:]
    print(b)  
    result_d = clf.predict(d) # 0 for dark; 1 for bright
    result_b = clf.predict(b) # 0 for dark; 1 for bright
    #print(result_d, '\n', result_b)
    
    for j in range(100):
        if result_d[j]!=0:
        #if d[j]>=2:
            error_d+=1
        if result_b[j]!=1:
        #if b[j]<2:  
            error_b+=1
print('dark error counts:', error_d)
print('bright error counts:', error_b)
accuracy_rate=1-(float)(error_b+error_d)/(samples*20000.0)
print('total accuracy rate:',accuracy_rate)   #bins 1: 98.56%    10: 98.56%
f.close()