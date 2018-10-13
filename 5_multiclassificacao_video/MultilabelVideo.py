
# coding: utf-8

# # Classificação de multi etiquetas de vídeo
# 
# 
# 

# In[1]:


import tensorflow as tf
import numpy as np

import os 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from read_batch import *
from evaluate import *


# In[2]:


def build_deep_fully_network(num_features, num_classes):
    
    # Placeholders
    X_ = tf.placeholder(dtype=tf.float32, shape=[None, num_features])
    Y_ = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])
    
    kernel_initializer = tf.contrib.layers.xavier_initializer()
    
    layer1 = tf.layers.dense(X_, 2000, activation=tf.nn.relu, kernel_initializer=kernel_initializer)
    
    layer2 = tf.layers.dense(layer1, 3200, activation=tf.nn.relu, kernel_initializer=kernel_initializer)  
    
    logits = tf.layers.dense(layer2, num_classes, activation=None, kernel_initializer=kernel_initializer)
    
    #Loss function
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y_))

    #Optimizer
    opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    
    output = tf.nn.sigmoid(logits)
    
    #Acc
    correct_prediction = tf.equal(tf.round(output), Y_)
    acc = evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return opt, cost, acc, output, X_, Y_


# In[3]:


#Iniciando
sess = tf.InteractiveSession()

num_features = 1152
num_classes = 3862

#construindo o modelo de rede
opt, cost, acc, output, X_, Y_ = build_deep_fully_network(num_features, num_classes)

# inicializando as variveis do tensorflow
sess.run(tf.global_variables_initializer())


# In[4]:


num_epochs = 20
best_gap = 0
for epoch in range(num_epochs):
    
    #training loop
    while True:
        batch_num, labels, features = get_next_train_batch()
        # if batch_num is lower than 0, so this means that the batch ends,  a new epoch must starts
        if batch_num < 0:
            break
                
        feed_dict={X_: features, Y_: labels}
        _, loss = sess.run([opt,cost], feed_dict=feed_dict)
        
        #print("epoch", epoch,"batch id", batch_num,"loss:", loss)
        
    #test loop
    while True:
        batch_num, labels, features = get_next_val_batch()
        if batch_num < 0:
            break

        feed_dict={X_: features, Y_: labels}
        result = sess.run(output, feed_dict=feed_dict)
        register_batch_evaluation(labels, num_classes, result)
    
    gap, hit1, hit5, hit20 = get_global_evaluation_result()
    print("validation: epoch", epoch, "- gap", gap, "top1", hit1, "top5", hit5, "top20", hit20)
    
    clear_registered_evaluations()
    
    if gap > best_gap:
        best_gap = gap
        save_path = saver.save(sess, "save/train_"+str(best_gap)+".ckpt")
        print("new best saved at ...", save_path)
    
        

