import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess an image: resize, convert to grayscale, apply CLAHE, and enhance features."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)

    # Apply Canny Edge Detection
    edges = cv2.Canny(image, 50, 150)

    # Apply Fourier Transform (FFT)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    
    # Resize and Normalize
    image = cv2.resize(magnitude_spectrum, target_size)
    image = image.astype("float32") / 255.0
    image = img_to_array(image)
    
    return image