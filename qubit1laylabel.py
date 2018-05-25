


from __future__ import print_function
from __future__ import division
import tensorflow as tf

import numpy as np
import random
import cPickle as pickle
import matplotlib.pyplot as plt
import argparse
import math
import gzip
save_file='./model1lay.ckpt'

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)


args = parser.parse_args()
#print(tf.reduce_sum([[1,2],[3,4]],reduction_indices=[0,1]))

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    
    error=tf.reduce_sum((abs(y_pre-v_ys)))

    
    result1 = sess.run(error, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})

    return result1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,1,1], strides=[1,2,1,1], padding='SAME')

def leakyrelu(x, alpha=0.3, max_value=None):  #alpha need set
    '''ReLU.

    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x


num_bins=100
xs = tf.placeholder(tf.float32, [None, num_bins])   # 28x28
ys = tf.placeholder(tf.float32, [None, 2])  #num_p add 1 om
keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)


W_fc1 = weight_variable([num_bins, 2])
b_fc1 = bias_variable([2])


saver = tf.train.Saver()  #define saver of the check point

prediction = tf.nn.softmax(tf.matmul(tf.reshape(xs,[-1,num_bins]), W_fc1) + b_fc1)
loss = tf.reduce_mean(tf.reduce_sum(abs(ys - prediction),
                                        reduction_indices=[1])) 
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)



fig = plt.figure()
f =gzip.open('./DetectionBinsData_pickle_clean.gzip','rb')  #49664*100 times measurement
if args.train:
    N=20  #max cnts per bin
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


    te_dd=pickle.load(f)[:,1:101]
    te_bb=pickle.load(f)[:,102:]
    fig_x=[]
    fig_y=[]
    for i in range (2500):  #training times
        try:
            #first row is cooling photon counts
            #d = pickle.load(f)[:,:101] #dark state data
            #b = pickle.load(f)[:,101:] #bright state data
            #without first row
            data=pickle.load(f)
            dd=data[:,1:101]
            bb=data[:,102:]
            y_d=[]
            y_b=[]
            #d
            for m in range (100):
                p_d=1
                for p in range (num_bins):
                    for q in range(N):
                        if d[m][p]== q:  #m for mth measurement, i for ith bin, j for cnts of bin
                            p_d*=prob_dis_d[p][q]  # i for ith bin, j for cnts of bin
                            #print(p_d)
                            break


                p_b=1
                for p in range (num_bins):
                    for q in range(N):
                        if d[m][p]== q:  #m for mth measurement, i for ith bin, j for cnts of bin
                            p_b*=prob_dis_b[p][q]  # i for ith bin, j for cnts of bin
                            #print(p_d)
                            break

                y_d.append([p_d,p_b])

            #b
            for m in range (100):
                p_d=1
                for p in range (num_bins):
                    for q in range(N):
                        if b[m][p]== q:  #m for mth measurement, i for ith bin, j for cnts of bin
                            p_d*=prob_dis_d[p][q]  # i for ith bin, j for cnts of bin
                            #print(p_d)
                            break


                p_b=1
                for p in range (num_bins):
                    for q in range(N):
                        if b[m][p]== q:  #m for mth measurement, i for ith bin, j for cnts of bin
                            p_b*=prob_dis_b[p][q]  # i for ith bin, j for cnts of bin
                            #print(p_d)
                            break

                y_b.append([p_d,p_b])
            y_d=np.array(y_d)
            y_b=np.array(y_b)
            print(y_b)
            if i<1000:
                _, curr_loss = sess.run([train_step,loss], feed_dict={xs: dd, ys:y_d, keep_prob: 0.5, lr:0.00005})
                _, curr_loss = sess.run([train_step,loss], feed_dict={xs: bb, ys:y_b, keep_prob: 0.5, lr:0.00005})
            elif i<2000:
                _, curr_loss = sess.run([train_step,loss], feed_dict={xs: dd, ys:y_d, keep_prob: 0.5, lr:0.001})
                _, curr_loss = sess.run([train_step,loss], feed_dict={xs: bb, ys:y_b, keep_prob: 0.5, lr:0.001})
            else:
                _, curr_loss = sess.run([train_step,loss], feed_dict={xs: dd, ys:y_d, keep_prob: 0.5, lr:0.0005})
                _, curr_loss = sess.run([train_step,loss], feed_dict={xs: bb, ys:y_b, keep_prob: 0.5, lr:0.0005})
            if i % 5 == 0:
                print(i)
                #error=compute_accuracy(te_dd,y_d)
                #error+=compute_accuracy(te_bb,y_b)
                curr_loss=100*curr_loss  #scale
                print(curr_loss)
                #print(sess.run(loss),feed_dict={xs: dd, ys:y_d, keep_prob: 0.5, lr:0.0001})
                fig_x.append(i)
                #fig_y.append(error)
                fig_y.append(curr_loss)
                plt.ylim(0,10)
                plt.plot(fig_x, fig_y, color='blue')
                #plt.pause(0.1)
        except EOFError:
            break
    saver.save(sess, save_file)
    #plt.ylim(0,50)
    plt.savefig('com.png')
    plt.show()
if args.test:
    '''
    #show the predict results
    dd=pickle.load(f)[:10,1:101]
    y_d=10*[[1,0]]
    bb=pickle.load(f)[:10,102:]
    y_b=10*[[0,1]]
    saver.restore(sess, save_file)
    predict_d = sess.run(prediction, feed_dict={xs: dd, keep_prob: 1})
    print(predict_d)
    predict_b = sess.run(prediction, feed_dict={xs: bb, keep_prob: 1})
    print(predict_b)
    '''

    #show the error/accuracy rate
    error_cnt_b=0
    error_cnt_d=0
    saver.restore(sess, save_file)
    test_samples=3   #test_samples * 100 *100 = test num
    for k in range (test_samples):
        for i in range (100):
            data=pickle.load(f)
            dd=data[:,1:101]
            bb=data[:,102:]
            predict_d = sess.run(prediction, feed_dict={xs: dd, keep_prob: 1})
            for j in range(100):
                if predict_d[j][0]<=predict_d[j][1]:
                    error_cnt_d+=1
            predict_b = sess.run(prediction, feed_dict={xs: bb, keep_prob: 1})
            for j in range(100):
                if predict_b[j][0]>=predict_b[j][1]:
                    error_cnt_b+=1
    error_rate_d=(float)(error_cnt_d/(test_samples*10000.0))
    error_rate_b=(float)(error_cnt_b/(test_samples*10000.0))
    accuracy_rate=1-(float)(error_cnt_b+error_cnt_d)/(test_samples*20000.0)
    print('error_dark sate:',error_cnt_d,error_rate_d)
    print('error_bright sate:',error_cnt_b,error_rate_b)
    print('total accuracy rate:',accuracy_rate)    
f.close()