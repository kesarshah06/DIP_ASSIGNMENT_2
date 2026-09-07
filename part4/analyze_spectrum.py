import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Read image
filename = 'input.jpeg'
print("Loaded:", filename)

img = Image.open(filename).convert("L")

# Resize so long axis = 1024 pixels
max_dim = 1024
scale = max_dim / max(img.size)

if scale < 1:
    new_size = (round(img.size[0] * scale),
                round(img.size[1] * scale))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

f = np.asarray(img, dtype=np.float64) / 255.0

print("Image size:", f.shape)

# Fourier transform of blurred image
F_blurred = np.fft.fft2(f)
F_blurred_shifted = np.fft.fftshift(F_blurred)

# Log spectrum for visualization
image_spectrum = np.log1p(np.abs(F_blurred_shifted))

# Save image spectrum to analyze
plt.figure(figsize=(10, 10))
plt.imshow(image_spectrum, cmap="gray")
plt.axis("off")
plt.savefig("spectrum.png", bbox_inches='tight', pad_inches=0)
print("Saved spectrum.png")
