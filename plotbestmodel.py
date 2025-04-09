from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import PIL.Image

# Load your Keras model from file

# Save the model plot as an image
plot_path = "model_plot.png"
plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)

# Display the plot
img = PIL.Image.open(plot_path)
plt.imshow(img)
plt.axis("off")
plt.show()