import tensorflow as tf

# needed to test if GPU is being used; NEED /Users/ishanayyar/miniconda3/envs/tf_metal/bin/python /Users/ishanayyar/Downloads/TinySpeck/test_metal_plugin.py
'''
Output: (tf_metal) ishanayyar@dhcp-10-105-177-159 TinySpeck % /Users/ishanayyar/miniconda3/envs/tf_metal/bin/python /Users/ishanayyar/Downloads/TinySpeck/test_metal_plugin.py
GPU is set to be used!!!
2025-04-02 11:17:14.582451: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M1 Pro
2025-04-02 11:17:14.582487: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 32.00 GB
2025-04-02 11:17:14.582497: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 10.67 GB
2025-04-02 11:17:14.582539: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2025-04-02 11:17:14.582553: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
Available devices: [LogicalDevice(name='/device:CPU:0', device_type='CPU'), LogicalDevice(name='/device:GPU:0', device_type='GPU')]
Computation result: [[1. 3.]
 [3. 7.]]
'''
# Force TensorFlow to use GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU is set to be used!!!")
    except RuntimeError as e:
        print(e)

print("Available devices:", tf.config.list_logical_devices())

with tf.device('/device:GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)

print("Computation result:", c.numpy())