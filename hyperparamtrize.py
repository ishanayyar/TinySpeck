import os
import numpy as np
import tensorflow as tf
from itertools import product
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# okay so here we're importing modules required for U-Net model - which is 1D in this case
import tensorflow as tf  
import tensorflow.keras as keras # type: ignore
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Activation, ReLU, Dense, Reshape, Multiply, Normalization
from tensorflow.keras.layers import BatchNormalization, Conv1DTranspose, Concatenate,Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
X=np.load('sigd.npy')[:120000]  # input signals from generated spectra
y1=np.load('sigc.npy')[:120000] # first target
y=np.load('sigi.npy')[:120000] # second target

print('finito')

# need to ask Mohammad which one is best to use in terms of downsampling
def downsample_subsampling(data, factor=2):
    """Downsample by selecting every nth point (subsampling)."""
    return data[:, ::factor]

def downsample_averaging(data, window_size=2):
    """Downsample by averaging adjacent points."""
    kernel = np.ones((1, window_size)) / window_size
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel.ravel(), mode='valid'), axis=1, arr=data)
    return smoothed[:, ::window_size]  # Take every nth point after smoothing

def downsample_pca(data, n_components=512):
    """Downsample using PCA to extract principal components."""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

# Load original data
X = np.load('sigd.npy')[:120000]  
y1 = np.load('sigc.npy')[:120000]  
y = np.load('sigi.npy')[:120000]  

# Ensure correct original size
original_size = X.shape[1]
if original_size not in [1024, 2048]:
    raise ValueError("Unexpected input size. Expected 1024 or 2048.")

target_size = 512  

# Downsample correctly
X = downsample_subsampling(X, factor=original_size // target_size)
y1 = downsample_subsampling(y1, factor=original_size // target_size)
y = downsample_subsampling(y, factor=original_size // target_size)

# Reshape for Conv1D (batch, time_steps, channels)
X = X.reshape(X.shape[0], X.shape[1], 1)
y = y.reshape(y.shape[0], y.shape[1], 1)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=7)

print("X_train shape:", X_train.shape)  # Should be (batch_size, 512, 1)
print("y_train shape:", y_train.shape)  # Should be (batch_size, 512, 1)

#ALL the hyperparameters 
param_grid = {
    'num_blocks': [2, 3, 4],  # Number of encoder-decoder blocks
    'max_pool_stride': [2, 3, 4],  # Max pooling strides
    'batch_size': [8, 32, 64, 128],  # Batch sizes
    'learning_rate': [1e-4, 1e-3, 1e-2]  # Learning rates
}

def U_Net(input_shape, num_blocks, max_pool_stride):
    inputs = tf.keras.layers.Input(shape=input_shape)

    #encoder
    x = inputs
    skips = []
    for i in range(num_blocks):

        # Use batch normalization here AND THEN activation --> else output becomes less than 0
        # Dead neurons
        x = tf.keras.layers.Conv1D(64 * (2 ** i), 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)  # Add batch norm
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv1D(64 * (2 ** i), 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)  # Add batch norm
        x = tf.keras.layers.Activation('relu')(x)
        
        skips.append(x)  #Save the skip connection
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=max_pool_stride)(x)

    
    x = tf.keras.layers.Conv1D(1024, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv1D(1024, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    #decoder
    for i in range(num_blocks):
        x = tf.keras.layers.UpSampling1D(size=2)(x)
        x = tf.keras.layers.Concatenate()([x, skips[-(i + 1)]]) 
        
        x = tf.keras.layers.Conv1D(64 * (2 ** (num_blocks - i - 1)), 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv1D(64 * (2 ** (num_blocks - i - 1)), 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

    #output layer
    outputs = tf.keras.layers.Conv1D(1,1)(x)

    model = tf.keras.models.Model(inputs, outputs)
    return model

for num_blocks, max_pool_stride, batch_size in product(param_grid['num_blocks'], 
                                                       param_grid['max_pool_stride'], 
                                                       param_grid['batch_size']):
    print(f"Training model: Blocks={num_blocks}, PoolStride={max_pool_stride}, Batch={batch_size}")
    model = U_Net(input_shape=(X_train.shape[1],1),  # Downsampled input shape
                  num_blocks=num_blocks, 
                  max_pool_stride=max_pool_stride)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  
    model.compile(optimizer=optimizer, loss=tf.keras.losses.LogCosh(), metrics=['mae'])

    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val), 
                        epochs=50,  
                        batch_size=batch_size)

    model_filename = f"unet_blocks{num_blocks}_pool{max_pool_stride}_batch{batch_size}.keras"
    model.save(model_filename)
    print(f"Model saved as {model_filename}")