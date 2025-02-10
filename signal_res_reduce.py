import numpy as np
import matplotlib.pyplot as plt

# Load original data
sigi = np.load('sigc.npy')[:120000]  # Load full dataset

# Select a single sample spectrum to visualize (e.g., the first one)
original_spectrum = sigi[0]

# Apply downsampling (subsampling method)
factor = original_spectrum.shape[0] // 512
downsampled_spectrum = original_spectrum[::factor]

# Plot original vs. downsampled spectrum
plt.figure(figsize=(10, 5))
plt.plot(original_spectrum, label="Original (1024/2048 points)", alpha=0.7)
plt.plot(np.linspace(0, len(original_spectrum)-1, len(downsampled_spectrum)), 
         downsampled_spectrum, label="Downsampled (512 points)", marker='o', linestyle='dashed')

plt.xlabel("Data Points")
plt.ylabel("Intensity")
plt.legend()
plt.title("Before and After Downsampling")
plt.grid()
plt.savefig('downsampled_spectrum3.png', dpi=300)
plt.show()
