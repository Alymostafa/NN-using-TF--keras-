#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[4]:


from tensorflow import keras as k


# In[5]:


import numpy as np


# In[6]:


import matplotlib.pyplot as plt


# In[8]:




print(tf.__version__)


# In[12]:


fashion_mnist=keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[14]:


class_names=['t-shirt/top','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankleboot']

train_images.shape
# In[15]:


train_images.shape


# In[16]:


test_images.shape


# In[17]:


len(train_labels)


# In[18]:


train_labels.shape


# In[19]:


train_labels


# In[20]:


test_labels.shape


# In[24]:


#plt.figure()
plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)


# In[25]:


train_images=train_images/255.0
test_images=test_images/255.0


# In[54]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    #plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
    #plt.ylabel(class_names[train_labels[i]])


# In[86]:


model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])


#  

# In[87]:


model.compile(optimizer=tf.train.AdamOptimizer()
              ,loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[89]:


model.fit(train_images, train_labels, epochs=5)


# In[93]:


test_loss,test_acc=model.evaluate(test_images,test_labels)
print('test accuracy',test_acc)


# In[94]:


predictions=model.predict(test_images)


# In[95]:


predictions[0]


# In[96]:


np.argmax(predictions[0])


# In[97]:


test_labels[0]


# In[98]:


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# In[99]:


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)


# In[102]:


i = 25
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)


# In[103]:


# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)


# In[104]:


img=test_images[0]
print(img.shape)


# In[105]:


img = (np.expand_dims(img,0))
print(img.shape)


# In[107]:


prediction_single=model.predict(img)
print(prediction_single)


# In[116]:


plot_value_array(0, prediction_single, test_labels)
_ = plt.xticks(range(10), class_names,rotation=30)


# In[117]:


np.argmax(prediction_single[0])


# In[ ]:




