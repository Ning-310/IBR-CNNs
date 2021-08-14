import pandas as pd
import numpy as np
from skimage import io
import os, sys
import random
import keras
from skimage import transform
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import tensorflow as tf
import sys
from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.metrics import roc_curve, roc_auc_score
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D,Conv1D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras import backend as K

from sklearn.metrics import roc_auc_score

class roc_callback(keras.callbacks.Callback):
    def __init__(self,training_data, validation_data):

        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)

        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)

        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
class RocAucMetricCallback(keras.callbacks.Callback):
    def __init__(self, predict_batch_size=1024, include_on_batch=False):
        super(RocAucMetricCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['auc_acc'] = float('-inf')
            if (self.validation_data):
                logs['auc_acc'] = roc_auc_score(self.validation_data[1],
                                                    self.model.predict(self.validation_data[0] ))

    def on_train_begin(self, logs={}):
        if not ('auc_acc' in self.params['metrics']):
            self.params['metrics'].append('auc_acc')

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        logs['auc_acc'] = float('-inf')
        if (self.validation_data):
            logs['auc_acc'] = roc_auc_score(self.validation_data[1],
                                                self.model.predict(self.validation_data[0]))
    def auc(y_true, y_pred):
        ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
        pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
        pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
        binSizes = -(pfas[1:]-pfas[:-1])
        s = ptas*binSizes
        return K.sum(s, axis=0)

    def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
        y_pred = K.cast(y_pred >= threshold, 'float32')

        N = K.sum(1 - y_true)

        FP = K.sum(y_pred - y_pred * y_true)
        return FP/N

    def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
        y_pred = K.cast(y_pred >= threshold, 'float32')
        P = K.sum(y_true)
        TP = K.sum(y_pred * y_true)
        return TP/P

model = Sequential()
print(x_train.shape[1:])
model.add(Conv2D(16,  (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=5e-3),
            metrics=['accuracy'])
model.summary()
batch_size = 128
epochs =20

history = model.fit(x_train, y_train, batch_size=batch_size,
    nb_epoch=epochs,validation_data=(x_test, y_test1),class_weight = 'auto',
    callbacks=[RocAucMetricCallback(),ModelCheckpoint('model.model', monitor='auc_acc', save_best_only=True)])
model=load_model('model.model')
predict = model.predict(x_test)
pindx = np.argmax(predict, axis=1)
predict = [predict[i][1] for i in range(len(y_test))]
auc = roc_auc_score(y_test[:], predict)
fpr, tpr, threshold = roc_curve(y_test[:], predict, pos_label=1)

label = [k + ' (AUC = ' + str(round(auc, 3)) + ')']
xmajorLocator = MultipleLocator(0.1)
xmajorFormatter = FormatStrFormatter('%1.1f')
ymajorLocator = MultipleLocator(0.1)
ymajorFormatter = FormatStrFormatter('%1.1f')

ax = subplot(111)

ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_major_formatter(xmajorFormatter)

ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_major_formatter(ymajorFormatter)

ax.xaxis.grid(True, which='major')
ax.yaxis.grid(True, which='major')
plt.plot(fpr, tpr, 'r')

plt.legend(label, loc=4)
plt.savefig('roc.png', dpi=300)
plt.clf()
