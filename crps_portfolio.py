# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 16:45:24 2020

@author: UOS
"""
import os
import numpy as np

import tensorflow as tf

os.chdir("C:/Users/UOS/Desktop/portfolio/dataset")
#%%
x = np.load('return_mat.npy')[0:100,0:40]
#%%
def q_fun(gamma, bl, quantile, oml): 
    q = gamma + tf.reduce_sum(bl * tf.clip_by_value(quantile - tf.math.cumsum(oml),
                                                          clip_value_min=0, clip_value_max=float("inf")))
    return q    

def find_tilde_a(x, beta, gamma, bl, oml):
    q = tf.map_fn(lambda x: q_fun(gamma, bl, x, oml), tf.math.cumsum(oml)) 
    xbeta = x @ beta[:, tf.newaxis]
    l0 = np.zeros(x.shape[0], dtype=np.dtype(np.int64))
    
    for i in range(len(q)-1):
        l0[tf.squeeze((q[i] < xbeta) & (xbeta < q[i+1]))] = i 
    
    tilde_a = np.zeros(x.shape[0])
    
    for j in range(x.shape[0]):
        tilde_a[j] = tf.clip_by_value((xbeta[j] - gamma + tf.math.cumsum(bl*tf.math.cumsum(oml))[l0[j]])/tf.math.cumsum(bl)[l0[j]], clip_value_min=0, clip_value_max=1)
        
    return tilde_a
# tf.clip_by_value((xbeta - gamma + sum(bl[:l0] * tf.math.cumsum(oml)[:l0]))/sum(bl[:l0]), clip_value_min=0, clip_value_max=1)    
#%%
def loss(tilde_a, x, beta, gamma, bl, oml, tmu):
    xbeta = x @ beta[:, tf.newaxis]
    
    return (2*tf.reduce_sum((tilde_a - 1)*xbeta) + tf.reduce_sum((1-2*tilde_a)*gamma) +
            x.shape[0]*tf.reduce_sum(bl*((1-tf.pow(tf.math.cumsum(oml),3))/3 - tf.math.cumsum(oml))) -
            tf.reduce_sum(bl * tf.pow(tf.math.maximum(tilde_a[:, tf.newaxis], tf.math.cumsum(oml2)), 2)) +
            tf.reduce_sum(bl * tf.math.maximum(tilde_a[:, tf.newaxis], tf.math.cumsum(oml))*tf.math.cumsum(oml)) +
            1000000*tf.abs(tf.reduce_mean(xbeta) - tmu) + 1000000*tf.abs(tf.reduce_sum(beta) - 1))
#%%
beta = tf.Variable(tf.constant(np.repeat(1, x.shape[1])/x.shape[1], dtype=tf.dtypes.float32))
bl = tf.Variable(tf.random.uniform([5], minval=0, maxval=0.1, dtype=tf.dtypes.float32))
oml = tf.Variable(tf.constant(np.repeat(1, 5)/5, dtype=tf.dtypes.float32))    
gamma = tf.Variable(tf.random.normal([1], mean=0, stddev=0.01, dtype=tf.dtypes.float32))
tmu = tf.constant([0.0007])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# loss(tilde_a, xbeta, gamma, bl, dl)
for i in range(1000):
    with tf.GradientTape() as tape:
        bl2 = tf.clip_by_value(bl, clip_value_min=0, clip_value_max=float("inf"))
        oml2 = tf.math.exp(oml)/tf.reduce_sum(tf.math.exp(oml))
        tilde_a = find_tilde_a(x, beta, gamma, bl2, oml2)
        tmp_loss = loss(tilde_a, x, beta, gamma, bl2, oml2, tmu)
        if (i+1) % 10 == 0:
            print(tmp_loss)
    grads = tape.gradient(tmp_loss, [beta, gamma, bl, oml])
    optimizer.apply_gradients(zip(grads, [beta, gamma, bl, oml]))   

#%% check
tf.reduce_mean(tf.reduce_mean(x, axis=0))
xbeta = x @ beta[:, tf.newaxis]
gamma

# loss
(2*tf.reduce_sum((tilde_a - 1)*xbeta) + tf.reduce_sum((1-2*tilde_a)*gamma) +
    x.shape[0]*tf.reduce_sum(bl2*((1-tf.pow(tf.math.cumsum(oml2),3))/3 - tf.math.cumsum(oml2))) -
    tf.reduce_sum(bl2 * tf.pow(tf.math.maximum(tilde_a[:, tf.newaxis], tf.math.cumsum(oml2)), 2)) +
    tf.reduce_sum(bl2 * tf.math.maximum(tilde_a[:, tf.newaxis], tf.math.cumsum(oml2))*tf.math.cumsum(oml2)))

### constraint
1-tf.reduce_sum(beta)
tf.abs(tf.reduce_mean(xbeta) - tmu)

# quantile estimate
tf.map_fn(lambda x: q_fun(gamma, bl, x, oml2), tf.math.cumsum(oml2)) 
