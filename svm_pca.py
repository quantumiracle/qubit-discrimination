#coding=utf-8
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
import pickle
import gzip
import numpy as np
f =gzip.open('./DetectionBinsData_pickle_clean.gzip','rb') 
#clf = SGDClassifier(max_iter=10000)
clf = svm.SVC()
num_components=10
pca = PCA(n_components=num_components)
#learn step
X=[]
Y=[]
batch_num=600
for i in range (batch_num):   #training times
    print(i)
    data=pickle.load(f)
    d=data[:,1:101]
    b=data[:,102:]
    '''
    pca.fit(b)
    print('variance ratio=')
    print(pca.explained_variance_ratio_)
    time.sleep()
    print(pca.transform(b))
    '''
    pca_d=pca.fit_transform(d)
    pca_b=pca.fit_transform(b)
    batch=[]
    batch.append(pca_d)
    batch.append(pca_b)
    #print(batch)
    train_batch=np.array(batch).reshape(2*100,num_components)
    #print(train_batch) 
    train_Y=100*[0]+100*[1]
    #print(Y)
    X.append(train_batch)
    Y.append(train_Y)
    
    
X=np.array(X).reshape(batch_num*200,num_components)
Y=np.array(Y).reshape(batch_num*200)
#http://scikit-learn.org/stable/modules/scaling_strategies.html#incremental-learning
clf.fit(X, Y)  

#test step
error_d=0
error_b=0
for i in range(100):
    test=pickle.load(f)
    d=test[:,1:101]
    b=test[:,102:]
    pca_d=pca.fit_transform(d)
    pca_b=pca.fit_transform(b)  
    result_d = clf.predict(pca_d) # 0 for dark; 1 for bright
    result_b = clf.predict(pca_b) # 0 for dark; 1 for bright
    #print(result_d, '\n', result_b)
    
    for j in range(100):
        if result_d[j]!=0:
            error_d+=1
        if result_b[j]!=1:
            error_b+=1
print('dark error counts:', error_d)
print('bright error counts:', error_b)
accuracy_rate=1-(float)(error_b+error_d)/20000.0
print('total accuracy rate:',accuracy_rate)   #94.44%
f.close()