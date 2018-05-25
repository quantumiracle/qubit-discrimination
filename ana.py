import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
f =gzip.open('./DetectionBinsData0_pickle.gzip','rb')
t=np.linspace(1,100,100)
i=1
total_curve_b=100*[0]
total_curve_d=100*[0]
while(i<3000):
    try:
        print(i)
        #print(pickle.load(f))
        data=pickle.load(f)
        i +=1
        #bright state
        plt.subplot(221)
        plt.scatter(t,np.sum(data[:,102:],axis=0))  #photon bins distribution
        total_curve_b+=np.sum(data[:,102:],axis=0)  #sum of bins distribution
        #dark state
        plt.subplot(223)
        plt.scatter(t,np.sum(data[:,1:101],axis=0))
        total_curve_d+=np.sum(data[:,1:101],axis=0)
    except EOFError:
        break

f.close()
plt.subplot(221)
plt.title("Bright State")
plt.plot(np.sum(data[:,102:],axis=0))  #single example distribution, 100 mearsure*100 bins
plt.subplot(223)
plt.plot(total_curve_b)
plt.subplot(222)
plt.title("Dark State")
plt.plot(np.sum(data[:,1:101],axis=0))
plt.subplot(224)
plt.plot(total_curve_d)
plt.show()