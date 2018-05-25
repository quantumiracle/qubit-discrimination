#coding=utf-8
#bins 2进制转10进制
import numpy as np
import gzip
import math

f1name = 'DetectionBinsData_pickle_clean.gzip'  #去除0列或101列的cooling photons cnt<10的整个100 measurements 的sample
f2name = 'DetectionBinsData_pickle_compact2-10.gzip'
import pickle

f1 = gzip.open(f1name, 'rb')
f2 = gzip.open(f2name, 'wb')

num=10 #num of bins after compact    
i=1
while(i<4000):
    try:
        print(i)
        data=pickle.load(f1)
        d=data[:,1:101]
        b=data[:,102:]
        compact=[]
        for j in range (100):
            d_com=[]
            b_com=[]
            cnt_d=0
            cnt_b=0

            for k in range(num):
                #print(k*100/num,(k+1)*100/num)
                dc=0
                bc=0
                for m in range(100/num):
                    dc+=d[j][k*100/num+m]*pow(2,m)
                d_com.append(dc)
                for m in range(100/num):
                    bc+=b[j][k*100/num+m]*pow(2,m)
                b_com.append(bc)
            #print(b_com) 
            compact.append(np.concatenate([d_com, b_com]))
        #print(compact)
        pickle.dump(compact,f2)
        i +=1
    except EOFError:
        break

f1.close()
f2.close()