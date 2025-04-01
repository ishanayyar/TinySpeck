import tensorflow as tf

# check TensorFlow version and GPU availability

print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Check if the device is using Metal
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("Metal plugin detected with GPU:", gpu_devices)
else:
    print("No Metal plugin GPU detected.")
    
    
    ### YESSSSS GPU NOW WORKING IN MACOS WITH METAL PLUGIN!
    #### REMEMBER TO ACTIVATE TF_METAL_PLUGIN ENVIRONMENT VARIABLE :) 