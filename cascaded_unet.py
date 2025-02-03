# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:15:05 2022
@author: mkazemzadeh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### seems like sklearn is now deprecated; use scikit-learn
from sklearn.model_selection import train_test_split

# okay so here we're importing modules required for U-Net model - which is 1D in this case
import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Activation, ReLU, Dense, Reshape, Multiply, Normalization
from tensorflow.keras.layers import BatchNormalization, Conv1DTranspose, Concatenate,Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# hmmmmmm I suppose the raspberry Pi will require cleaned data.... do we need to add a processing step here?
# or can we asume data will be in a simpler format?

X=np.load('sigd.npy')[:120000]  # input signals from generated spectra
y1=np.load('sigc.npy')[:120000] # first target
y=np.load('sigi.npy')[:120000] # second target

print('finito')

X=X.reshape(X.shape[0],X.shape[1],1) # reshaped to have single-channel dimension
y1=y1.reshape(y1.shape[0],y1.shape[1],1)
y=y.reshape(y.shape[0],y.shape[1],1)

# data split into test & training  ----- if using raw data here, might need to do a clean first??? Not artificial spectra?
# or is the denoising from 1D U-Net sufficient?
X_train,X_val,y1_train,y1_val,y_train,y_val=train_test_split(X,y1,y) 

# building U-Net
# so this function is applying two 1D convolutional layers with batch normalization and ReLU activation

# Note to self: 1D convolutional layer applies set of kernels (learnable filters) to input tensor to extract local patterns
# using this to detect local features in Raman data - like peaks, trends etc....


def convolution_operation(entered_input, filters=64):
    # Taking first input and implementing the first conv block
    conv1 = Conv1D(filters, kernel_size = (3), padding = "same", kernel_regularizer=regularizers.L2(1e-4))(entered_input)
    batch_norm1 = BatchNormalization()(conv1)
    act1 = ReLU()(batch_norm1) # adds non-linearity
    
    # act1 = attention(act1)  ------ why do you remove the attention blocks???
    
    # Taking first input and implementing the second conv block
    conv2 = Conv1D(filters, kernel_size = (3), padding = "same", kernel_regularizer=regularizers.L2(1e-4))(act1)
    batch_norm2 = BatchNormalization()(conv2)
    act2 = ReLU()(batch_norm2)
    
    # act2 = attention(act2)
    return act2

# apply the net / convolution, and then max pooling for downsampling
def encoder(entered_input, filters=64):
    # Collect the start and end of each sub-block for normal pass and skip connections
    enc1 = convolution_operation(entered_input, filters)
    MaxPool1 = MaxPooling1D(strides = (2))(enc1)
    return enc1, MaxPool1

# upsample and concatenate skip connections
def decoder(entered_input, skip, filters=64):
    # Upsampling and concatenating the essential features
    Upsample = Conv1DTranspose(filters, 2, strides=2, padding="same"
                               , kernel_regularizer=regularizers.L2(1e-4))(entered_input)
    Connect_Skip = Concatenate()([Upsample, skip])
    out = convolution_operation(Connect_Skip, filters)
    return out

Sz1=16
Sz2=16
Sz3=16


# first stage processes input to generate output (out0) --an intermediate output
# second stage produces output1 - like a refined version
### FINAL model -> takes X and predicts two outputs [out0, out1]

def U_Net(Image_Size):
    # Take the image size and shape
    input1 = Input(Image_Size)
    
    # Construct the encoder blocks
    skip1_1, encoder_1 = encoder(input1, Sz1)
    skip2_1, encoder_2 = encoder(encoder_1, Sz1*2)
    skip3_1, encoder_3 = encoder(encoder_2, Sz1*4)
    skip4_1, encoder_4 = encoder(encoder_3, Sz1*8)
    
    # Preparing the next block
    conv_block = convolution_operation(encoder_4, Sz1*16)
    
    # Construct the decoder blocks
    decoder_1 = decoder(conv_block, skip4_1, Sz1*8)
    decoder_2 = decoder(decoder_1, skip3_1, Sz1*4)
    decoder_3 = decoder(decoder_2, skip2_1, Sz1*2)
    decoder_4 = decoder(decoder_3, skip1_1, Sz1)
    
    out0 = Conv1D(1, 1, padding="same")(decoder_4)
    
    
    skip1, encoder_1 = encoder(out0, Sz2)
    skip2, encoder_2 = encoder(encoder_1, Sz2*2)
    skip3, encoder_3 = encoder(encoder_2, Sz2*4)
    skip4, encoder_4 = encoder(encoder_3, Sz2*8)
    
    # Preparing the next block
    conv_block = convolution_operation(encoder_4, Sz2*16)
    
    # Construct the decoder blocks
    decoder_1 = decoder(conv_block, skip4, Sz2*8)
    decoder_2 = decoder(decoder_1, skip3, Sz2*4)
    decoder_3 = decoder(decoder_2, skip2, Sz2*2)
    decoder_4 = decoder(decoder_3, skip1, Sz2)
    
    out1 = Conv1D(1, 1, padding="same")(decoder_4)

    model = Model(input1, [out0,out1])
    return model

model=U_Net((1024, 1))
model.summary()



class TestCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch%1==0:
            at=model.predict(X_val[:10])[-1]
            fig, ax =plt.subplots(4,5,figsize=(12,10), sharex=True, sharey='row')
            for j in range(5):
                ax[0,j].plot(X_val[j])
                ax[1,j].plot(y_val[j])
                ax[2,j].plot(at[j])
                ax[3,j].plot(at[j]-y_val[j])

            for i in range(4):
                for j in range(5):
                    ax[i,j].grid()
                    
            plt.tight_layout()
            plt.show()

### need a .keras file suffix in this following line
checkpoint_filepath = 'g 2.h5.keras'

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='low',
    save_best_only=True)


# okay so the loss function is mean absolute error and that's for both outputs 
#Adam is an optimizer and learning rate is 1e-3

# model trained on 200 epochs with batch size 256 ---- so it learns to map input spectra to two output signals

model.compile(loss=['mae','mae'], optimizer=Adam(learning_rate=1e-3,),loss_weights=[1,1])
his=model.fit(X_train, [y1_train,y_train], validation_data=(X_val, [y1_val,y_val]),
              batch_size=256, callbacks=[model_checkpoint_callback,TestCallback()],
                            epochs=200)


