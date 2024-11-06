import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, std=25):
    """
    Add Gaussian noise to an image.
    
    Parameters:
    - image: Input image
    - mean: Mean of the Gaussian noise
    - std: Standard deviation of the Gaussian noise
    
    Returns:
    - Noisy image
    """
    gaussian_noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + gaussian_noise

    # Clip the values to be in the valid range [0, 255] and convert to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

# Load an image using OpenCV
image = cv2.imread("C:/Users/Zeynep/Image-Processing-2024/original-image.jpg", cv2.IMREAD_GRAYSCALE)  # Load in grayscale

# Add Gaussian noise
mean = 0        # Adjust mean of the noise
std = 25        # Adjust standard deviation of the noise for intensity
noisy_image = add_gaussian_noise(image, mean, std)

# Save the noisy image instead of displaying it
cv2.imwrite("gaussian-noise-image.jpg", noisy_image)
print("Noisy image saved as gaussian_noisy_image.jpg")