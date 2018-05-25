#maximum likelihood method wothout bright-dark process
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle

f =gzip.open('./DetectionBinsData_pickle_clean.gzip','rb')
#print([2][10*[0]])
N=20  #max cnts per bin
num_bins=100
print(1/2)
#initialize
dis_b=[]
dis_d=[]
for i in range(num_bins):
    dis_b.append(N*[0])
    dis_d.append(N*[0])
prob_dis_b=[]
prob_dis_d=[]
for i in range(num_bins):
    prob_dis_b.append(N*[0])
    prob_dis_d.append(N*[0])

for i in range (50):
    print(i)
    data=pickle.load(f)
    d=data[:,1:num_bins+1]
    b=data[:,num_bins+2:]
    for j in range (100):
        for k in range(num_bins):
            for m in range(N):
                if d[j][k]==m:    #j for jth measurement, k for kth bin
                    dis_d[k][m]+=1  #k for kth bin,  m for this bin cnt to be m
                    break
    for j in range (100):
        for k in range(num_bins):
            for m in range(N):
                if b[j][k]==m:
                    dis_b[k][m]+=1
                    break
for i in range (num_bins):
    prob_dis_b[i]=(dis_b[i])/np.sum(dis_b[i])   # probability distribution of cnts [0,N] for all bins
    prob_dis_d[i]=(dis_d[i])/np.sum(dis_d[i])
                    
#test
err_d=0
err_b=0
samples=100
for n in range(samples):
    data=pickle.load(f)
    d=data[:,1:num_bins+1]
    b=data[:,num_bins+2:]
    #d test
    for m in range (100):
        p_d=1
        for i in range (num_bins):
            for j in range(N):
                if d[m][i]== j:  #m for mth measurement, i for ith bin, j for cnts of bin
                    p_d*=prob_dis_d[i][j]  # i for ith bin, j for cnts of bin
                    #print(p_d)
                    break
        print(p_d)

        p_b=1
        for i in range (num_bins):
            for j in range(N):
                if d[m][i]== j:  #m for mth measurement, i for ith bin, j for cnts of bin
                    p_b*=prob_dis_b[i][j]  # i for ith bin, j for cnts of bin
                    #print(p_d)
                    break
        print(p_b)
        
        if p_d<p_b:
            err_d+=1    

    #b test
    for m in range (100):
        p_d=1
        for i in range (num_bins):
            for j in range(N):
                if b[m][i]== j:  #m for mth measurement, i for ith bin, j for cnts of bin
                    p_d*=prob_dis_d[i][j]  # i for ith bin, j for cnts of bin
                    #print(p_d)
                    break
        print(p_d)

        p_b=1
        for i in range (num_bins):
            for j in range(N):
                if b[m][i]== j:  #m for mth measurement, i for ith bin, j for cnts of bin
                    p_b*=prob_dis_b[i][j]  # i for ith bin, j for cnts of bin
                    #print(p_d)
                    break
        print(p_b)
        if p_d>p_b:
            err_b+=1  
print(err_b,err_d)
acc=1-(err_b+err_d)/(samples*2*100)
print("acc:", acc)