import wavio
import numpy as np
import pandas as pd

a=["%02d" % (x+1) for x in range(37)]
p1_final = []
p2_final = []
t1_final = []
t2_final = []

for z in range(28):
    y = z
    f = "E:/Neural Nets/Project Data/Sound Files/HS_D"+a[y]+".wav"
    wav1 = wavio.read(f)
    print(wav1.rate)
    print(wav1.sampwidth)
    s1 = wav1.data[:,0]/32767
    s2 = wav1.data[:,1]/32767
    rate = int(wav1.rate/50)
    p1 = []
    i = 0
    j = rate
    while j<=s1.shape[0]:
        p1.append(s1[i:j])
        i = i+int(rate/2)
        j = j+int(rate/2)
    
    
    p2 = []
    i = 0
    j = rate
    while j<=s2.shape[0]:
        p2.append(s2[i:j])
        i = i+int(rate/2)
        j = j+int(rate/2) 
    
    d1 = pd.read_csv("E:/Neural Nets/Project Data/CSV_Files_Final/HS_D"+a[y]+"_Spk1.csv")
    d1['tmi0'] = ((d1['tmi0']*100).round()).astype('int')
    d1['tmax'] = ((d1['tmax']*100).round()).astype('int')
    
    
    d2 = pd.read_csv("E:/Neural Nets/Project Data/CSV_Files_Final/HS_D"+a[y]+"_Spk2.csv")
    d2['tmi0'] = ((d2['tmi0']*100).round()).astype('int')
    d2['tmax'] = ((d2['tmax']*100).round()).astype('int')
    
    t1=[]
    t2=[]
    x = []
    for i in range(len(d1)):
        t1 = t1 + [d1.iloc[i][2]]*(d1.iloc[i][1] - d1.iloc[i][0])
        x.append([d1.iloc[i][1],len(t1)])
    for i in range(len(d2)):
        t2 = t2 + [d2.iloc[i][2]]*(d2.iloc[i][1] - d2.iloc[i][0])
        
      
            
    for i in range(len(t1)-1):
        t1[i] = min(t1[i],t1[i+1])
    for i in range(len(t2)-1):
        t2[i] = min(t2[i],t2[i+1])
        
    t1 = t1[0:len(p1)]
    t2 = t2[0:len(p2)]
    
    p1_final = p1_final + p1
    p2_final = p2_final + p2
    t1_final = t1_final + t1
    t2_final = t2_final + t2
    
print(len(p1_final))
print(len(p2_final))
print(len(t1_final))
print(len(t2_final))

p_final = p1_final+p2_final
t_final = t1_final+t2_final 

for i in range(len(t_final)):
    if t_final[i] == '1`' or t_final[i]==12:
        t_final[i] = 1

print("start")
p_final = np.asarray(p_final).astype(np.float32)
t_final = np.asarray(t_final).astype(np.float32)
print("end")

loss_g = []

import tensorflow as tf

learning_rate = 0.001
num_steps = 60000
batch_size = 100

def cnn_model_fn(features,reuse=False,training=True):
    with tf.variable_scope('neural_net_model', reuse=reuse):
        input_layer = tf.reshape(features['images'], shape=[-1, 21, 21, 2])
        conv1 = tf.layers.conv2d(input_layer, 32, 5, activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
        pool2_flat = tf.contrib.layers.flatten(pool2)
        dense = tf.layers.dense(pool2_flat, 1024)
        dropout = tf.layers.dropout(dense, rate=0.4, training=training)
        logits = tf.layers.dense(dropout, 2)
    return logits


def model_fn(features, labels, mode):

    logits_train = cnn_model_fn(features)
    logits_test = cnn_model_fn(features, reuse=True,training=False)

    predictions = tf.argmax(logits_test, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss,global_step=tf.train.get_global_step())
    acc_op = tf.metrics.accuracy(labels=labels, predictions=predictions)
    loss_g.append(loss)
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

p1_final = []
p2_final = []
t1_final = []
t2_final = []

for z in range(1):
    y = 17
    f = "E:/Neural Nets/Project Data/Sound Files/HS_D"+a[y]+".wav"
    wav1 = wavio.read(f)
    print(wav1.rate)
    print(wav1.sampwidth)
    s1 = wav1.data[:,0]/32767
    s2 = wav1.data[:,1]/32767
    rate = int(wav1.rate/50)
    p1 = []
    i = 0
    j = rate
    while j<=s1.shape[0]:
        p1.append(s1[i:j])
        i = i+int(rate/2)
        j = j+int(rate/2)
    
    
    p2 = []
    i = 0
    j = rate
    while j<=s2.shape[0]:
        p2.append(s2[i:j])
        i = i+int(rate/2)
        j = j+int(rate/2) 
    
    d1 = pd.read_csv("E:/Neural Nets/Project Data/CSV_Files_Final/HS_D"+a[y]+"_Spk1.csv")
    d1['tmi0'] = ((d1['tmi0']*100).round()).astype('int')
    d1['tmax'] = ((d1['tmax']*100).round()).astype('int')
    
    
    d2 = pd.read_csv("E:/Neural Nets/Project Data/CSV_Files_Final/HS_D"+a[y]+"_Spk2.csv")
    d2['tmi0'] = ((d2['tmi0']*100).round()).astype('int')
    d2['tmax'] = ((d2['tmax']*100).round()).astype('int')
    
    t1=[]
    t2=[]
    x = []
    for i in range(len(d1)):
        t1 = t1 + [d1.iloc[i][2]]*(d1.iloc[i][1] - d1.iloc[i][0])
        x.append([d1.iloc[i][1],len(t1)])
    for i in range(len(d2)):
        t2 = t2 + [d2.iloc[i][2]]*(d2.iloc[i][1] - d2.iloc[i][0])
    
    for i in range(len(t1)-1):
        t1[i] = min(t1[i],t1[i+1])
    for i in range(len(t2)-1):
        t2[i] = min(t2[i],t2[i+1])
        
    t1 = t1[0:len(p1)]
    t2 = t2[0:len(p2)]
    
    p1_final = p1_final + p1
    p2_final = p2_final + p2
    t1_final = t1_final + t1
    t2_final = t2_final + t2




p1_final = np.asarray(p1).astype(np.float32)
t1_final = np.asarray(t1).astype(np.float32)
p2_final = np.asarray(p2).astype(np.float32)
t2_final = np.asarray(t2).astype(np.float32)

model = tf.estimator.Estimator(model_fn)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': p_final}, y=t_final,
    batch_size=batch_size, num_epochs=None, shuffle=True)

model.train(input_fn, steps=num_steps)



input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': p1_final}, y=t1_final,
    batch_size=batch_size, shuffle=False)

e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])

a1= list(model.predict(input_fn))

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': p2_final}, y=t2_final,
    batch_size=batch_size, shuffle=False)

e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])

a2= list(model.predict(input_fn))

#a1= t1_final
#a2= t2_final
p = a1[0]
count = 0
f1 = []
for i in range(len(a1)):
    if a1[i] == p:
        count = count+1
    else:
        f1.append([a1[i-1],count])
        count = 1
        p = a1[i]
        
f1.append([a1[-1],count])


p = a2[0]
count = 0
f2 = []
for i in range(len(a2)):
    if a2[i] == p:
        count = count+1
    else:
        f2.append([a2[i-1],count])
        count = 1
        p = a2[i]
        
f2.append([a2[-1],count])

c1 = []
g = 0
for i in range(len(f1)):
    c1.append([g,g+f1[i][1]/10.0,f1[i][0],1])
    g = g+f1[i][1]/10