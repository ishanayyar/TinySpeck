import os
import numpy as np
import tensorflow as tf
from itertools import product
from sklearn.model_selection import train_test_split

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
from tensorflow.keras.callbacks import TensorBoard
import datetime

# set  GPU as the device
if tf.config.list_physical_devices('GPU'):
#    tf.config.set_visible_devices([], 'CPU')  # THIS ISN"TY WORKING!!!! Disable any CPU devices
    print("TensorFlow Metal is active!")
else:
    print("GPU not detected, TensorFlow Metal is not active.")
    
X=np.load('sigd.npy')[:120000]  # input signals from generated spectra
y1=np.load('sigc.npy')[:120000] # first target
y=np.load('sigi.npy')[:120000] # second target

# Ensure correct original size
original_size = X.shape[1]
if original_size not in [1024, 2048]:
    raise ValueError("Not expected input size. I expected 1024 or 2048.")


# norm data to [0, 1] range - adding epsilon to avoid div by zero
epsilon = 1e-8
X = X / (np.max(X, axis=1, keepdims=True) + epsilon)
y = y / (np.max(y, axis=1, keepdims=True) + epsilon)

### y1 = y1 / (np.max(y1, axis=1, keepdims=True) + 1e-8)  # Normalize y1 too!
print(X, y)
X= 100 * X
y = 100 * y

# Downsample data
def downsample_subsampling(data, factor=2):
    return data[:, ::factor]


### linear scaling --> just subsampling
### add callback at the bottom
### normalize the intensity of input spectra --> multiply by a number to bring maximum to same level
### bring it down to 0-1 range.... int14?

## downsample to 512 points then get max value then divide to get 0-1 range
## model checkpoint callback

## search plateau function so it stops epochs when it reaches a plateau
## lower learning rate = learns more slowly but more accurately
## reduce learning rate on plateau callback
'''
# need to ask Mohammad which one is best to use in terms of downsampling
def downsample_subsampling(data, factor=2):
## take every nth pt
    return data[:, ::factor]

def downsample_averaging(data, window_size=2):
## avg adjacent pts
    kernel = np.ones((1, window_size)) / window_size
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel.ravel(), mode='valid'), axis=1, arr=data)
    return smoothed[:, ::window_size]  # Take every nth point after smoothing
'''
'''def downsample_pca(data, n_components=512):
# downsample using PCA
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)'''


target_size = 512  

# Downsample correctly
X = downsample_subsampling(X, factor=original_size // target_size)
#y1 = downsample_subsampling(y1, factor=original_size // target_size)
y = downsample_subsampling(y, factor=original_size // target_size)
X = X.reshape(X.shape[0], X.shape[1], 1)
y = y.reshape(y.shape[0], y.shape[1], 1)



## just use y_train ---> not y1 ---> it doesn't need the noise
# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=7)



print("X_train shape:", X_train.shape)  # Should be (batch_size, 512, 1)
print("y_train shape:", y_train.shape)  # Should be (batch_size, 512, 1)
'''
# need to reshape becuase sometimes it breaks at ~epoch 64?
X_train = X_train.reshape(X_train.shape[0], 512, 1)
y_train = y_train.reshape(y_train.shape[0], 512, 1)
'''


#ALL the hyperparameters 
param_grid = {
    'num_blocks': [1, 2, 3, 4],  # Number of encoder-decoder blocks
    'max_pool_stride': [ 4],  # Max pooling strides   # 2 done and 3 was having issues with indivisibility errors, so removing
    'batch_size': [32, 64, 128],  # Batch sizes
    # failing at lower batch sizes
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


# my custom callback for plotting predictions and residuals
class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, run_name):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.run_name = run_name  # need a unique identifier for each hyperparameter set

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0:
            at = self.model.predict(self.X_val[:10])[-1]
            fig, ax = plt.subplots(4, 5, figsize=(12, 10), sharex=True, sharey='row')

            for j in range(5):
                ax[0, j].plot(self.X_val[j])
                ax[1, j].plot(self.y_val[j])
                ax[2, j].plot(at[j])
                ax[3, j].plot(at[j] - self.y_val[j])

            for i in range(4):
                for j in range(5):
                    ax[i, j].grid()

            plt.tight_layout()
            save_path = f"plots/{self.run_name}_epoch{epoch}.png"
            os.makedirs("plots", exist_ok=True)  # ensure directory exists
            plt.savefig(save_path)
            plt.close()



# Training loop with hyperparameter tuning
for num_blocks, max_pool_stride, batch_size in product(param_grid['num_blocks'], 
                                                   param_grid['max_pool_stride'], 
                                                   param_grid['batch_size']):
    print(f"Training model: Blocks={num_blocks}, PoolStride={max_pool_stride}, Batch={batch_size}")
    
    # Create model instance
    model = U_Net(input_shape=(X_train.shape[1], 1),
                 num_blocks=num_blocks,
                 max_pool_stride=max_pool_stride)
    
    # Define optimizer and callbacks
    optimizer = Adam(learning_rate=1e-4)
    
    # Model checkpoint callback
    checkpoint_filepath = 'best.keras'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    # Create a timestamped log directory
    log_dir = (
        f"logs/unet_experiments/" 
        f"blocks_{num_blocks}_"
        f"stride_{max_pool_stride}_"
        f"bs_{batch_size}_"
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,  # Log weight histograms every epoch
        write_graph=True,
        write_images=True,
        profile_batch=(500, 520),
        update_freq='epoch',# Profile batches 100-110
    )
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    # Create TestCallback with required arguments
    run_name = f"blocks{num_blocks}_pool{max_pool_stride}_batch{batch_size}"
    test_callback = TestCallback(X_val, y_val, run_name)
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=tf.keras.losses.LogCosh(), metrics=['mae'])

    # Train the model
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=100,
        batch_size=batch_size,
        callbacks=[
            model_checkpoint_callback, 
            early_stopping_callback, 
            reduce_lr_callback, 
            test_callback,
            tensorboard_callback
        ]
    )

    model_filename = f"new_unet_blocks{num_blocks}_pool{max_pool_stride}_batch{batch_size}.keras"
    model.save(model_filename)
    print(f"Model saved as {model_filename}")

### plot at end of each epoch --> need a callback at end of epoch

## train model that doesn't produce noise --> just cleans
## install tensorflow --> metal 
## normalise and x 100
