import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate

# Read image
filename = 'input.jpeg'
img = Image.open(filename).convert("L")

max_dim = 1024
scale = max_dim / max(img.size)

if scale < 1:
    new_size = (round(img.size[0] * scale),
                round(img.size[1] * scale))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

f = np.asarray(img, dtype=np.float64) / 255.0

# Fourier transform
F_blurred = np.fft.fft2(f)
F_blurred_shifted = np.fft.fftshift(F_blurred)
S = np.log1p(np.abs(F_blurred_shifted))

M, N = S.shape

# Zero out the DC component and low frequencies
Y, X = np.ogrid[:M, :N]
R = np.sqrt((X - N/2)**2 + (Y - M/2)**2)
S_filtered = S.copy()
S_filtered[R < 20] = 0

# Radon transform-like approach: rotate and project
angles = np.arange(0, 180, 1)
variances = []

for angle in angles:
    rotated = rotate(S_filtered, angle, reshape=False)
    # project along columns
    proj = np.sum(rotated, axis=0)
    variances.append(np.var(proj))

best_angle = angles[np.argmax(variances)]
print(f"Estimated angle of blur stripes: {best_angle} degrees")

# The blur direction is perpendicular to the stripes
blur_angle = (best_angle + 90) % 180
print(f"Estimated blur angle: {blur_angle} degrees")

# Rotate to align stripes vertically, then the blur is horizontal
rotated_S = rotate(S_filtered, best_angle, reshape=False)
proj = np.sum(rotated_S, axis=0)

# Find peaks/troughs in the projection to find the frequency of zero crossings
from scipy.signal import find_peaks
troughs, _ = find_peaks(-proj, distance=10)
print(f"Troughs indices: {troughs}")
if len(troughs) > 1:
    avg_dist = np.mean(np.diff(troughs))
    print(f"Average distance between dark bands: {avg_dist}")
    
    # If the distance is D in frequency domain (pixels), the length in spatial domain is N / D
    # Wait, the projection is of length N.
    # blur_length = N / D
    # Let's be careful about N vs M. The rotated image has size M x N roughly.
    blur_length = N / avg_dist
    print(f"Estimated blur length: {blur_length}")
    
    a_est = blur_length * np.cos(np.deg2rad(blur_angle))
    b_est = blur_length * np.sin(np.deg2rad(blur_angle))
    print(f"Estimated blur vector (a, b): ({a_est}, {b_est})")
else:
    print("Could not find enough troughs.")

