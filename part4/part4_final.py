import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil

#..................................................reading the input image..................................................
filename = 'input.jpeg'
print("Loaded:", filename)

img = Image.open(filename).convert("L")

# ...........................................Resize so long axis = 1024 pixels................................................
max_dim = 1024
scale = max_dim / max(img.size)

if scale < 1:
    new_size = (round(img.size[0] * scale),
                round(img.size[1] * scale))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

f = np.asarray(img, dtype=np.float64) / 255.0

print("Image size:", f.shape)


a = 30.0                                                                             # estimated blur vector: (a, b) = (30.0, 0.0) pixels
b = 0.0
print(f"Estimated blur vector: (a, b) = ({a}, {b}) pixels")

M, N = f.shape
u = np.fft.fftfreq(N)                                                                # u is the frequency variable corresponding to the horizontal axis (N) of the image
v = np.fft.fftfreq(M)                                                                # v is the frequency variable corresponding to the vertical axis (M) of the image
U, V = np.meshgrid(u, v)

z = a * U + b * V                                                                    # motion blur transfer function H(u,v) = exp(-j*pi*(a*u + b*v)) * sinc(a*u + b*v)
H = np.exp(-1j * np.pi * z) * np.sinc(z)

#........................................................inverse filtering.......................................................
G = np.fft.fft2(f)                                                                   # G is the Fourier transform of the blurred image f, which represents the frequency domain representation of the image. It is used in the inverse filtering process to recover the original image from the blurred version by dividing it by the transfer function H.
epsilon = 1e-3                                                                       # small value to avoid division by zero
H_safe = H.copy()
H_safe[np.abs(H_safe) < epsilon] = epsilon                                           # H_safe is a modified version of the transfer function H, where any values that are too close to zero (less than epsilon) are replaced with epsilon. This is done to prevent division by zero when performing inverse filtering, ensuring numerical stability in the calculations.
F_inverse = G / H_safe                                                               # F_inverse is the result of the inverse filtering process, obtained by dividing the Fourier transform of the blurred image G by the modified transfer function H_safe. This operation aims to recover the original image in the frequency domain by compensating for the effects of the motion blur represented by H.
f_inverse = np.real(np.fft.ifft2(F_inverse))
f_inverse_display = np.clip(f_inverse, 0, 1)

#........................................................WIENER DECONVOLUTION.......................................................
def wiener_deconvolution(G, H, K):
    F_hat = (np.conj(H) * G) / (np.abs(H)**2 + K)
    result = np.real(np.fft.ifft2(F_hat))
    result = np.clip(result, 0, 1)
    return result

#...................................................tuning the K parameter for Wiener deconvolution................................................
K_best = 0.005
wiener_best = wiener_deconvolution(G, H, K_best)
wiener_low = wiener_deconvolution(G, H, 0.1 * K_best)
wiener_high = wiener_deconvolution(G, H, 10 * K_best)

#...................................................REGULARIZED DECONVOLUTION.......................................................
P = 4 - 2 * np.cos(2 * np.pi * U) - 2 * np.cos(2 * np.pi * V)                        # P(u,v) = |Px|^2 + |Py|^2 = 4 - 2*cos(2*pi*U) - 2*cos(2*pi*V)

def regularized_deconvolution(G, H, P, lam):
    F_hat = (np.conj(H) * G) / (np.abs(H)**2 + lam * P + 1e-8)
    result = np.real(np.fft.ifft2(F_hat))
    result = np.clip(result, 0, 1)
    return result

#......................................tuning the lambda parameter for regularized deconvolution................................................
lam_best = 0.05 
reg_best = regularized_deconvolution(G, H, P, lam_best)
reg_low = regularized_deconvolution(G, H, P, 0.1 * lam_best)
reg_high = regularized_deconvolution(G, H, P, 10 * lam_best)

#.....................................................saving the results......................................................
os.makedirs("part4_results", exist_ok=True)

def save_image(filename, image):
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(image_uint8).save("part4_results/" + filename)

H_mag_shifted = np.fft.fftshift(np.abs(H))

save_image("01_input.png", f)
save_image("02_H_magnitude.png", H_mag_shifted / H_mag_shifted.max())
save_image("03_inverse_filter.png", f_inverse_display)
save_image("04_wiener_0.1K.png", wiener_low)
save_image("05_wiener_best.png", wiener_best)
save_image("06_wiener_10K.png", wiener_high)
save_image("07_reg_0.1lam.png", reg_low)
save_image("08_reg_best.png", reg_best)
save_image("09_reg_10lam.png", reg_high)

print("Saved results in: part4_results/")
