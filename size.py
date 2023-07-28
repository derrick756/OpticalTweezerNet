# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:35:28 2019

@author: admin
"""
from tensorflow import keras
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.pyplot.switch_backend('agg')
import spectra_process.subpys as subpys
import scipy.optimize as optimize
import os
import time

# Set default decvice: GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # if set '-1', python runs on CPU, '0' uses 1060 6GB
# In[] CNN preprocessing: 2-2
source_data_path = './data/EV/'
save_train_model_path = './TweezerNet/EV/'

# load Resnet or CNN model
X_mean = np.load(save_train_model_path+'X_scale_mean.npy')
X_std = np.load(save_train_model_path+'X_scale_std.npy')
model = keras.models.load_model(save_train_model_path+'regression_model.h5')
print('1. Finish loading CNN model!')
# load experimental dataset
X = np.load(source_data_path+'X_train.npy')

Y = np.load(source_data_path+'Y_train.npy')

print('2. Finish loading test dataset!') 
X = (X - X_mean)/X_std
YPredict = model.predict(np.reshape(X, (X.shape[0], X.shape[1], 1)))
print(YPredict.shape)
print('3. Finish predicting!') 

rmse = np.sqrt(((Y-YPredict)**2).mean())
r1= np.sqrt(((Y-YPredict)**2).mean())
rmse=r1
fontsize_val = 19
plt.figure(figsize=(8, 8))
    

print (rmse)

x=np.array(Y)
y=np.array(YPredict)
a,b=np.polyfit(x,y,1)

plt.scatter(x,y)
#print(x)
#print(y)
plt.plot(x,a*x+b, color='red')
dataframe= pd.DataFrame(y)
dataframe.to_csv('./output/train_status/binary_train/data1.csv')

plt.xlabel('True size', fontsize=fontsize_val)
plt.ylabel(' Predicted size', fontsize=fontsize_val)


plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize_val)
plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize_val)
#plt.text(0.85, 0.05, 'RMSE = %f'%r1, fontsize=fontsize_val)
plt.setp(plt.gca().get_lines(), linewidth=4)
plt.title('Network vs. Actual size ', fontsize=fontsize_val)

#CSV_Path = './output/train_status/binary_train/'
#lr = pd.read_csv(CSV_Path+'run-train-tag-epoch_lr.csv').values


plt.savefig('./output/prediction/2-2.png', dpi=300)
plt.show()



