import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
filename = "input.jpeg"

#.....................................................processing the input image..............................................
img = Image.open(filename).convert('L')                             

max_size = 1024                                                                    # making sure the image is not too large for processing

scale = min(1.0, max_size / max(img.size))
new_size = (                                                                       # making the image 1024*1024 or smaller if it is already smaller than that
    int(img.size[0] * scale),
    int(img.size[1] * scale)
)

img = img.resize(new_size)                                                         # resizing the image itself

image = np.asarray(img, dtype=np.float64)                                          # converting the image to a floating point array

#........................................FFT and IFFT implementations : COMPLEXITY (O(N log N))..............................................
def fft1d(x):

    #.............................................using radix-2 Cooley--Tukey FFT algorithm.....................................
    x = np.asarray(x, dtype=complex)                                               # converting the input to a complex array
    N = len(x)

    if N == 1:                                                                     # Base case: return the input array if its length is 1
        return x.copy()

    if N & (N - 1):                                                                # Power of 2 check: if N is not a power of 2, raise an error
        raise ValueError("Length of input must be a power of 2")

    X_even = fft1d(x[::2])                                                         # Recursively compute the FFT of the even-indexed elements
    X_odd = fft1d(x[1::2])

    k = np.arange(N // 2)                                                          # Twiddle factor: compute the complex exponential factors for combining the even and odd FFTs

    W = np.exp(-2j * np.pi * k / N)                                                # Compute the twiddle factors for combining the even and odd FFTs

    X = np.zeros(N, dtype=complex)                                                 # Initialize the output array for the FFT result

    X[:N//2] = X_even + W * X_odd                                                  # Combine the even and odd FFTs to compute the first half of the FFT result
    X[N//2:] = X_even - W * X_odd                                                  # Combine the even and odd FFTs to compute the second half of the FFT result

    return X

#.............................................1D inverse FFT implementation using the FFT above..............................................
def ifft1d(X):

    X = np.asarray(X, dtype=complex)                                              # converting the input to a complex array
    N = len(X)                                                                    

    return np.conj(fft1d(np.conj(X))) / N                                         # Compute the inverse FFT by taking the conjugate of the input, applying the FFT, taking the conjugate of the result, and dividing by N

#........................................2D FFT and IFFT implementations using the 1D FFT and IFFT above..............................................
def fft2d(image):

    image = np.asarray(image, dtype=complex)                                      # converting the input to a complex array

    rows, cols = image.shape

    temp = np.zeros((rows, cols), dtype=complex)                                  # Initialize a temporary array to store the FFT results along the rows

    for i in range(rows):
        temp[i, :] = fft1d(image[i, :])

    result = np.zeros((rows, cols), dtype=complex)                                # Initialize the final result array

    for j in range(cols):
        result[:, j] = fft1d(temp[:, j])

    return result

#........................................2D inverse FFT implementation using the 1D inverse FFT above..............................................
def ifft2d(F):

    F = np.asarray(F, dtype=complex)                                             # converting the input to a complex array

    rows, cols = F.shape

    temp = np.zeros((rows, cols), dtype=complex)                                 # Initialize a temporary array to store the IFFT results along the rows

    for i in range(rows):
        temp[i, :] = ifft1d(F[i, :])

    result = np.zeros((rows, cols), dtype=complex)                               # Initialize the final result array

    for j in range(cols):
        result[:, j] = ifft1d(temp[:, j])

    return result

#...........................................Padding the image to the next power of 2 for FFT..............................................
def next_power_of_two(n):
    return 1 if n == 1 else 2 ** int(np.ceil(np.log2(n)))                        # Compute the next power of two greater than or equal to n using logarithm and exponentiation

#...........................................Padding the image to the next power of 2 for FFT..............................................
def pad_to_power_of_two(image):
    rows, cols = image.shape

    new_rows = next_power_of_two(rows)                                           # Compute the next power of two for the number of rows
    new_cols = next_power_of_two(cols)

    padded = np.zeros((new_rows, new_cols), dtype=float)                         # pad the image so to make it a square image with dimensions that are powers of two

    padded[:rows, :cols] = image

    return padded

#...........................................Testing the FFT and IFFT implementations..............................................
padded_image = pad_to_power_of_two(image)

print("Original shape:", image.shape)
print("Padded shape:", padded_image.shape)

F = fft2d(padded_image)

reconstructed = ifft2d(F)

reconstructed = np.real(reconstructed)

error = np.max(np.abs(padded_image - reconstructed))                             # error is the maximum absolute difference between the original padded image and the reconstructed image

print("Maximum reconstruction error:", error)                      

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(padded_image, cmap='gray')
plt.title("Original / Padded Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed, cmap='gray')
plt.title("IFFT(FFT(Image))")
plt.axis("off")

plt.savefig("reconstruction.png")                                               # Save the plot as a PNG file
plt.close()

#....................................................Gaussian kernel generation.................................................
def gaussian_kernel(size, sigma=None):

    if sigma is None:
        sigma = size / 6.0

    ax = np.arange(-(size // 2), size // 2 + 1)

    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    kernel /= np.sum(kernel)

    return kernel

#....................................................Spatial convolution implementation.................................................
def spatial_convolution(image, kernel):

    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)

    kh, kw = kernel.shape

    pad_h = kh // 2                                                             # Compute the amount of padding needed for the height of the kernel
    pad_w = kw // 2

    padded = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode='constant'
    )

    kernel_flipped = np.flipud(np.fliplr(kernel))                               # Flip the kernel both vertically and horizontally to prepare for convolution

    output = np.zeros_like(image)

    for i in range(image.shape[0]):                                             # Loop over each pixel in the image   
        for j in range(image.shape[1]):

            region = padded[
                i:i+kh,
                j:j+kw
            ]

            output[i, j] = np.sum(region * kernel_flipped)

    return output

#.................................................Frequency convolution implementation.................................................
def prepare_kernel(kernel, shape):

    kh, kw = kernel.shape                                                    

    padded_kernel = np.zeros(shape, dtype=float)

    padded_kernel[:kh, :kw] = kernel

    padded_kernel = np.roll(
        padded_kernel,
        -kh // 2,
        axis=0
    )

    padded_kernel = np.roll(
        padded_kernel,
        -kw // 2,
        axis=1
    )

    return padded_kernel

#.................................................Frequency convolution implementation.................................................
def frequency_convolution(image, kernel):

    image_padded = pad_to_power_of_two(image)                                  # Pad the image to the next power of two for efficient FFT computation

    kernel_padded = prepare_kernel(
        kernel,
        image_padded.shape
    )

    F = fft2d(image_padded)                                                    # Compute the FFT of the padded image

    H = fft2d(kernel_padded)                                                   # Compute the FFT of the padded kernel

    G = F * H                                                                  # Perform element-wise multiplication in the frequency domain to apply the convolution

    result = ifft2d(G)                                                         # Compute the inverse FFT to obtain the convolved image in the spatial domain

    result = np.real(result)

    result = result[:image.shape[0], :image.shape[1]]

    return result, F, H, G

kernel_size = 31                                                               # Define the size of the Gaussian kernel to be used for blurring the image

kernel = gaussian_kernel(kernel_size)                                          # Generate the Gaussian kernel using the specified size

blurred, F, H, G = frequency_convolution(                                      # Apply frequency-domain convolution to the original image using the generated Gaussian kernel
    image,
    kernel
)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(blurred, cmap='gray')
plt.title("Frequency-Domain Blurred Image")
plt.axis("off")

plt.savefig("blurred_comparison.png")
plt.close()

def spectrum(F):
    return np.log1p(np.abs(np.fft.fftshift(F)))

plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("1. Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(kernel, cmap='gray')
plt.title("2. Blur Kernel")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(spectrum(F), cmap='gray')
plt.title("3. Image Spectrum")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(spectrum(H), cmap='gray')
plt.title("4. Kernel Spectrum")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(spectrum(G), cmap='gray')
plt.title("5. Product Spectrum")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(blurred, cmap='gray')
plt.title("6. Final Blurred Image")
plt.axis("off")

plt.tight_layout()
plt.savefig("spectra_steps.png")
plt.close()

#...............................................Timing comparison between spatial and frequency filtering.................................................
timing_size = 512

timing_image = Image.open(filename).convert('L')

scale = min(
    1.0,
    timing_size / max(timing_image.size)
)

timing_image = timing_image.resize(  
    (
        int(timing_image.size[0] * scale),
        int(timing_image.size[1] * scale)
    )
)

timing_image = np.asarray(
    timing_image,
    dtype=np.float64
)

print("Timing image shape:", timing_image.shape)

#............................................Timing comparison between spatial and frequency filtering.................................................
kernel_sizes = [3, 31, 71, 121, 201, 301, 401, 501]

spatial_times = []
frequency_times = []

#...............................................Timing comparison between spatial and frequency filtering.................................................
for size in kernel_sizes:

    print("Kernel size:", size)                                                  # Generate a Gaussian kernel of the specified size for convolution

    kernel = gaussian_kernel(size)                                               # applying gaussian kernel                      

    start = time.perf_counter()                                                  # Measure the start time for spatial convolution

    spatial_convolution(                                                         # applying spatial convolution to the timing image using the generated Gaussian kernel
        timing_image,
        kernel
    )

    end = time.perf_counter()                                                    # Measure the end time for spatial convolution

    spatial_time = end - start                                                   # Calculate the elapsed time for spatial convolution and store it in the list of spatial times

    spatial_times.append(spatial_time)

    start = time.perf_counter()

    frequency_convolution(                                                       # applying frequency convolution to the timing image using the generated Gaussian kernel
        timing_image,
        kernel
    )

    end = time.perf_counter()

    frequency_time = end - start                                                 # Calculate the elapsed time for frequency convolution and store it in the list of frequency times

    frequency_times.append(frequency_time)

    print(
        f"Spatial: {spatial_time:.4f}s | "
        f"Frequency: {frequency_time:.4f}s"
    )

plt.figure(figsize=(10, 6))

plt.plot(
    kernel_sizes,
    spatial_times,
    marker='o',
    label='Spatial Filtering'
)

plt.plot(
    kernel_sizes,
    frequency_times,
    marker='o',
    label='Frequency-Domain Filtering'
)

plt.xlabel("Kernel Size")
plt.ylabel("Computation Time (seconds)")
plt.title("Spatial vs Frequency-Domain Filtering")
plt.legend()
plt.grid(True)

plt.savefig("time_comparison_linear.png")
plt.close()

plt.figure(figsize=(10, 6))

plt.loglog(
    kernel_sizes,
    spatial_times,
    marker='o',
    label='Spatial Filtering'
)

plt.loglog(
    kernel_sizes,
    frequency_times,
    marker='o',
    label='Frequency-Domain Filtering'
)

plt.xlabel("Kernel Size")
plt.ylabel("Computation Time (seconds)")
plt.title("Computation Time vs Kernel Size")
plt.legend()
plt.grid(True, which="both")

plt.savefig("time_comparison_loglog.png")
plt.close()

