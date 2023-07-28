import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, Add, \
                         Activation, ZeroPadding1D, BatchNormalization, \
                         AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
from sklearn import preprocessing
import pandas as pd
import os
#from sklearn.metrics import mean_squared_log_error


#Architecture based off of the paper: 'Deep Residual Learning for Image Recognition'.

# Set default decvice: GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'




def idn_block(input_tensor):
    """
    Implementation of the identity block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main 
         path
    filters -- python list of integers, defining the number of filters in the 
               CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in 
             the network
    block -- string/character, used to name the layers, depending on their 
             position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    #conv_name_base = 'res' + str(stage) + block + '_branch'
    #bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    #F1, F2, F3 = filters
    
    # Save the input value to later add back to the main path. 
    X_shortcut = input_tensor
    
    # First component of main path
    X = Conv1D(filters=64, kernel_size=5, padding='same')(input_tensor)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv1D(filters=64, kernel_size=5, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv1D(filters=64, kernel_size=5, padding='same')(X)
    X = BatchNormalization()(X)

    # Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
   
    return X
    

def conv_block(input_tensor):
    """
    Implementation of the convolutional block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main 
         path
    filters -- python list of integers, defining the number of filters in the 
               CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in 
             the network
    block -- string/character, used to name the layers, depending on their 
             position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    #conv_name_base = 'res' + str(stage) + block + '_branch'
    #bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    #F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = input_tensor

    # First component of main path 
    X = Conv1D(filters=64, kernel_size=5, padding='same')(input_tensor)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv1D(filters=64, kernel_size=5, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv1D(filters=64, kernel_size=5, padding='same')(X)
    X = BatchNormalization()(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv1D(filters=64, kernel_size=5, padding='same')(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)

    # Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def step_decay(epoch):
    lr = 5e-5 # 1e-3
    drop_factor = 0.1
    drop_period = 10 # 20
    lrate = lr*np.math.pow(drop_factor, np.math.floor((1+epoch)/drop_period))
#    decay_rate.append(lrate)
    return lrate

    

def ResNet50(input_tensor, output_shape=1, dropout_rate=0.8, learning_rate=5e-5):
    """
    Implementation of the popular ResNet50 the following architecture:
    
    """    
    # Stage 1
    X = Conv1D(filters=64, kernel_size=5, padding='same')(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling1D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = conv_block(X)
    X = idn_block(X)
    X = idn_block(X)

    # Stage 3
    X = conv_block(X) 
    X = idn_block(X)
    X = idn_block(X)
    X = idn_block(X)

    # Stage 4
    X = conv_block(X) 
    X = idn_block(X)
    X = idn_block(X)
    X = idn_block(X)
    X = idn_block(X)
    X = idn_block(X)

    # Stage 5
    X = conv_block(X) 
    X = idn_block(X)
    X = idn_block(X)

    # AVGPOOL 
    X = AveragePooling1D((2,2), name='avg_pool')(X)
    
    # Flatten
    X = Flatten()(X)
    
    # Add extra dense layers
    #if extra_layers is not None:
        #assert len(extra_layers) == len(dropouts), \
               # "Arguments do Not match in length: extra_layers, dropouts."
        #for i, layer, dpout in (zip(range(len(extra_layers)), extra_layers, dropouts)):
           # X = Dense(layer, name='fc_'+str(i)+'_'+str(layer), activation='relu',
               #       kernel_initializer=glorot_uniform(seed=0))(X)
            #X = Dropout(dpout, seed=0, name='dropout_'+str(i)+'_'+str(dpout))(X)

    # Output 
    X = Dense(1, activation='relu')(X)

    
    # Create model
    model = Model(inputs = X_input, outputs = X)
    
    # Compile model
    learning_rate = 5e-5
    optim = Adam(lr=learning_rate, epsilon=1e-8)
    model.compile(loss=keras.losses.mean_squared_error, optimizer='adam',
                  metrics=['acc'])

    
    return model

max_epoch = 50
validation_ratio=0.1
batch_size = 32
learning_rate = 5e-5
dropout_rate = 0.8 # 0.2
source_data_path = './data/EV/'
save_train_model_path = './TweezerNet/EV/'
# read in train datasets and get sizes
print("Step 1: read in train and test data")
X = np.load(source_data_path+'X_train.npy')
Y = np.load(source_data_path+'Y_train.npy')
split_marker = np.int64(np.round((1-validation_ratio)*Y.shape[0]))

# normalization: zero-mean along column, with_std=False
X_scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(X[:split_marker, :])
X[:split_marker, :] = X_scaler.transform(X[:split_marker, :])
X[split_marker:, :] = X_scaler.transform(X[split_marker:, :])
np.save(save_train_model_path+'X_scale_mean.npy', X_scaler.mean_)
np.save(save_train_model_path+'X_scale_std.npy', X_scaler.scale_)

x_train = X[:split_marker, ]
y_train = Y[:split_marker, ]
x_test  = X[split_marker:,]
y_test  = Y[split_marker:, ]

    
# reshape train data
X = np.reshape(X, [X.shape[0], X.shape[1], 1])
    
# define train size 
input_shape = X.shape[1:]
output_shape = 1

print(input_shape)
print(X.shape)
print(X[:split_marker, :].shape)
print(X[split_marker:, :].shape)



if os.path.exists(save_train_model_path+'regression_model.h5'):
        model = keras.models.load_model(save_train_model_path+'regression_model.h5')
        model.summary()
        print(model.summary)
        print('Load saved model and train again!!!')
        print("###################################################################")
        print("Step 3: train saved Resnet model")
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=save_train_model_path+'regression_model.h5', verbose=1, save_best_only=True)
        ##################################################################################################################
        tbCallBack = keras.callbacks.TensorBoard(histogram_freq=0, write_graph=True, write_images=True)
        model.fit(X[:split_marker, ], Y[:split_marker, ], batch_size=batch_size, epochs=max_epoch, 
                  validation_data=(X[split_marker:,], Y[split_marker:, ]), callbacks=[checkpointer, tbCallBack])


else:

        model = ResNet50(input_shape, output_shape=1, dropout_rate=0.8, learning_rate=5e-5)

        lrate = keras.callbacks.LearningRateScheduler(step_decay)
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=save_train_model_path+'regression_model.h5', verbose=1, save_best_only=True)
        tbCallBack = keras.callbacks.TensorBoard(histogram_freq=0, write_graph=True, write_images=True)
        history = model.fit(X[:split_marker, ], Y[:split_marker, ], batch_size=batch_size, epochs=max_epoch, 
                  validation_data=(X[split_marker:, ], Y[split_marker:, ]), callbacks=[lrate, checkpointer, tbCallBack])
        train_loss = history.history['loss']
        train_acc = history.history['acc']
        val_loss = history.history['val_loss']
        val_acc = history.history['val_acc']
        train_lr = history.history['lr']
        train_loss = np.array(train_loss)
        train_acc = np.array(train_acc)
        val_loss = np.array(val_loss)
        val_acc = np.array(val_acc)
        train_lr = np.array(train_lr)
        epoch = np.zeros(max_epoch)
        for i in range(max_epoch):
            epoch[i] = i
        df_train_loss = pd.DataFrame({'Step':epoch,'Value':train_loss})
        df_train_loss.to_csv("run-train-tag-epoch_loss.csv",index=False,sep=',')
        df_train_acc = pd.DataFrame({'Step':epoch,'Value':train_acc})
        df_train_acc.to_csv("run-train-tag-epoch_accuracy.csv",index=False,sep=',')
        df_val_loss = pd.DataFrame({'Step':epoch,'Value':val_loss})
        df_val_loss.to_csv("run-validation-tag-epoch_loss.csv",index=False,sep=',')
        df_val_acc = pd.DataFrame({'Step':epoch,'Value':val_acc})
        df_val_acc.to_csv("run-validation-tag-epoch_accuracy.csv",index=False,sep=',')
        df_train_lr = pd.DataFrame({'Step':epoch,'Value':train_lr})
        df_train_lr.to_csv("run-train-tag-epoch_lr.csv",index=False,sep=',')   
               


 
               
print('Finish training!!!')
