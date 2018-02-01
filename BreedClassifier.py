import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

train = pd.read_csv('labels.csv')
train_path = '/Users/shashvatkedia/Desktop/folder/train/'
test_path = '/Users/shashvatkedia/Desktop/folder/test'

from scipy.misc import imresize

train_img = []
for i in range(len(train)):
    img = image.load_img(train_path+train['id'][i]+'.jpg',target_size=(224,224))
    img = image.img_to_array(img)
    train_img.append(img)
train_img = np.array(train_img)
train_img = preprocess_input(train_img)

test_img = []
for file in os.listdir(test_path):
    if file.endswith('.jpg'):
        img = image.load_img(os.path.join(test_path,file),target_size=(224,224))
        img = image.img_to_array(img)
        test_img.append(img)
test_img = np.array(test_img)
test_img = preprocess_input(test_img)

from keras import applications
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, BatchNormalization, Dropout, Flatten
from keras import backend as k
from keras.callbacks import LearningRateScheduler

model = applications.VGG19(weights = 'imagenet',include_top = False,input_shape = (224,224,3))

for layer in model.layers[:8]:
    layer.trainable = False

output = model.output
output = Flatten()(output)
output = Dense(output_dim = 2048,activation = 'relu',init = 'uniform')(output)
output = BatchNormalization()(output)
output = Dropout(0.2)(output)
output = Dense(output_dim = 2048,activation = 'relu',init = 'uniform')(output)
output = BatchNormalization()(output)
output = Dropout(0.1)(output)
predict = Dense(output_dim = 120,activation = 'softmax')(output)

model_f = Model(input = model.input,output = predict)

adam = optimizers.Adam(lr = 0.0001)

model_f.compile(optimizer = adam,loss = 'categorical_crossentropy',metrics = ['accuracy'])

lr = 0.0001
def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

train_y = np.asarray(train['breed'])
train_y = pd.get_dummies(train_y)
train_y = np.array(train_y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train_img,train_y,random_state = 0,test_size = 0.2)

model_f.fit(X_train,y_train,batch_size = 100,epochs = 10,validation_data = (X_test,y_test),callbacks = [LearningRateScheduler(lt_schedule)])

pred = model.predict(X_test,batch_size = 100)

from sklearn.metrics import log_loss
score = log_loss(y_test,pred)
print(score)
