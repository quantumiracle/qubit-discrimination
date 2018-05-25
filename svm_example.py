#coding=utf-8
from sklearn import svm  
  
X = [[0, 0], [1, 1], [1, 0]]  # training samples   
y = [0, 1, 1]  # training target  
clf = svm.SVC()  # class   
clf.fit(X, y)  # training the svc model  
  
result = clf.predict([[2, 2]]) # predict the target of testing samples 
print(result)