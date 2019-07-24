#!/usr/bin/env python
# coding: utf-8

# In[1]:


#保存文件路径用库
import pathlib
#PLT 
import matplotlib.pyplot as plt 
#panadas 数据处理用库
import pandas as pd
#多个表格联合显示
import seaborn as sns
import numpy as np
#TENSORFLOW KERAS LAYER常用库
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#查看TF版本，一定要2.0新版本才行
print(tf.__version__)


# In[2]:


#给pandas列名
column_names = ['open','high','close','low','volume','price_change','p_change','ma5','ma10','ma20','v_ma5','v_ma10','v_ma20'] 
#将data格式数据读入内存 并加上每一列的名字
raw_dataset=pd.read_csv("D:/000875.CSV",
                        names=column_names,
                        na_values="0",
                        comment="\t",
                        sep=",",skipinitialspace=True)
#复制数据集
dataset=raw_dataset.copy()


# In[3]:


#统计有哪些数据不全
dataset.isna().sum()
#去掉数据中不全的行
dataset=dataset.dropna()


# In[4]:


ad=dataset['close']-dataset['open']
aod=[]
for i in range(len(ad)):
    if ad[i] >= 0 :
        aod.append(1)
    else:
        aod.append(0)


# In[5]:


del(aod[0])


# In[6]:


dataset


# In[7]:


dataset.drop(dataset.index[-1], axis=0)


# In[8]:


dataset.drop(dataset.index[-1], axis=0)


# In[9]:


#分割训练数据 测试数据
# train_dataset = dataset.sample(frac=0.8,random_state=0)
# test_dataset = dataset.drop(train_dataset.index)
train_dataset = dataset.drop(dataset.index[-1], axis=0)
# test_dataset  = dataset[400:]
aod1 = aod
# aod2 = aod[400:]


# In[11]:


#统计各种数据 并且旋转易于观察

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
train_stats

#制作归一化函数 这里用的是Zscore方法
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
# normed_test_data = norm(test_dataset)


# In[12]:


def build_model():
    model = keras.Sequential([
        layers.Dense(9, activation=tf.nn.tanh, input_shape=[len(normed_train_data.keys())],kernel_regularizer = keras.regularizers.l1(0.001)),
        #         kernel_regularizer = keras.regularizers.l1(0.001),
#                 layers.Dropout(0.2),
#                 layers.Dense(32, activation=tf.nn.relu),
#                 layers.Dropout(0.2),
        #         layers.Dense(9, activation=tf.nn.tanh),
        layers.Dense(1,activation=tf.nn.softmax)
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
model = build_model()
model.summary()


# In[13]:


#回调函数编写
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 1 == 0: print('')
        print('.', end='')

EPOCHS = 100

history = model.fit(
      normed_train_data, aod,
      epochs=EPOCHS,
      validation_split = 0.2, verbose=1,
      callbacks=[PrintDot()])
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.head()


# In[20]:


#打印出损失值，观察有没有过拟合或者欠拟合
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist['loss']= history.losses

#     plt.figure()
#     plt.xlabel('Epoch')
#     plt.ylabel('Cross Entropy [label]')
#     plt.plot(hist['epoch'], hist['loss'],
#            label='Train Error')
#     plt.plot(hist['epoch'], hist['val_loss'],
#            label = 'Val Error')
# #     plt.plot(hist['epoch'], hist['accuracy'],
# #            label = 'acc')
# #     plt.xlabel('Epoch')
# #     plt.ylabel('ACCURACY]')
# #     plt.plot(hist['epoch'], hist['accuracy],
# #            label='Train Accuracy')
# #     plt.plot(hist['epoch'], hist['val_accuracy'],
# #            label = 'Val Accuracy')
#     plt.ylim([0,10])
#     plt.legend()
#     plt.show()
# plot_history(history)
# import matplotlib.pyplot as plt
# history_dict = history.history
# history_dict.keys()
# acc = history_dict['accuracy']
# val_acc = history_dict['val_accuracy']
# loss = history_dict['loss']
# val_loss = history_dict['val_loss']
# epochs = range(1, len(acc)+1)

plt.plot(hist['epoch'], hist['loss'],label='train loss')
plt.plot(hist['epoch'], hist['val_loss'],label='val loss')
plt.title('Train and val loss')
plt.xlabel('Epochs')
plt.xlabel('loss')
plt.legend()
plt.show()

plt.plot(hist['epoch'], hist['accuracy'], label='Training acc')
plt.plot(hist['epoch'], hist['val_accuracy'], label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[21]:


test_predictions = model.predict(normed_test_data)
test_predictions


# In[ ]:


weights = model.get_weights()


# In[ ]:


weights


# In[ ]:




