import matplotlib.pyplot as plt
# okay so here we're importing modules required for U-Net model - which is 1D in this case
import tensorflow as tf  
import tensorflow.keras as keras # type: ignore
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Activation, ReLU, Dense, Reshape, Multiply, Normalization
from tensorflow.keras.layers import BatchNormalization, Conv1DTranspose, Concatenate,Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# Predict using best model
best_model =  tf.keras.models.load_model('best.keras')
predictions = best_model.predict(X_val[:10])  

# Plot results
fig, ax = plt.subplots(4, 5, figsize=(12, 10), sharex=True, sharey='row')

for j in range(5):
    ax[0, j].plot(X_val[j], label="Input Spectrum")
    ax[1, j].plot(y_val[j], label="True Spectrum")
    ax[2, j].plot(predictions[j], label="Predicted Spectrum")
    ax[3, j].plot(predictions[j] - y_val[j], label="Residuals")  # Error difference

for i in range(4):
    for j in range(5):
        ax[i, j].grid()
        ax[i, j].legend()

plt.tight_layout()
plt.show()