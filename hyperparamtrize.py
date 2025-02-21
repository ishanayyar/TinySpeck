import os
import numpy as np
import tensorflow as tf
from itertools import product
from sklearn.model_selection import train_test_split

# Load your dataset (replace with your paths)
data_dir = "data/"
X = np.load(os.path.join(data_dir, "X.npy"))  # Input images
Y = np.load(os.path.join(data_dir, "Y.npy"))  # Ground truth masks

# Normalize images and split data into training and validation sets
X = X / 255.0  # Normalize to [0, 1]
Y = Y / 255.0  # Normalize to [0, 1]

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define hyperparameter grid
param_grid = {
    'num_blocks': [1, 2, 3, 4],  # Number of encoder-decoder blocks
    'max_pool_stride': [2, 3, 4],  # Max pooling strides
    'batch_size': [8, 32, 64, 128]  # Batch sizes
}

# Define U-Net model
def U_Net(input_shape, num_blocks, max_pool_stride):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    x = inputs
    skips = []
    for i in range(num_blocks):
        x = tf.keras.layers.Conv2D(64 * (2 ** i), (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(64 * (2 ** i), (3, 3), activation='relu', padding='same')(x)
        skips.append(x)  # Save the skip connection
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=max_pool_stride)(x)

    # Bottleneck
    x = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    for i in range(num_blocks):
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Concatenate()([x, skips[-(i + 1)]])  # Add skip connection
        x = tf.keras.layers.Conv2D(64 * (2 ** (num_blocks - i - 1)), (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(64 * (2 ** (num_blocks - i - 1)), (3, 3), activation='relu', padding='same')(x)

    # Output layer
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs, outputs)
    return model

# Iterate over all combinations of hyperparameters
for num_blocks, max_pool_stride, batch_size in product(param_grid['num_blocks'], 
                                                       param_grid['max_pool_stride'], 
                                                       param_grid['batch_size']):
    print(f"Training model with Blocks={num_blocks}, PoolStride={max_pool_stride}, Batch={batch_size}")
    
    # Build the U-Net model
    model = U_Net(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), 
                  num_blocks=num_blocks, 
                  max_pool_stride=max_pool_stride)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # Fixed learning rate
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, Y_train, 
                        validation_data=(X_val, Y_val), 
                        epochs=50,  # Adjust epochs as needed
                        batch_size=batch_size)

    # Save the model with a unique filename
    model_filename = f"unet_blocks{num_blocks}_pool{max_pool_stride}_batch{batch_size}.keras"
    model.save(model_filename)
    print(f"Model saved as {model_filename}")