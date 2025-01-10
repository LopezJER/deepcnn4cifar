import tensorflow as tf
import numpy as np


def preprocess_image_for_vgg16(image, target_size):
    """
    Preprocess a single image for VGG16:
    - Reshape, normalize, and resize.

    Args:
        image: Flat array of shape (3072,).
               The input image is a 1D array representing a 32x32 RGB image
               where the first 1024 values are the red channel, the next 1024
               are the green channel, and the final 1024 are the blue channel.
        target_size: Tuple specifying the desired image resolution (e.g., (224, 224)).

    Returns:
        Preprocessed image of shape (target_size[0], target_size[1], 3).
    """
    # Constants for ImageNet normalization
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Step 1: Reshape the flat array to (32, 32, 3)
    reshaped_image = image.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.float32)

    # Step 2: Normalize pixel values to [0, 1]
    normalized_image = reshaped_image / 255.0

    # Step 3: Resize the image to the target size (e.g., 224x224)
    resized_image = tf.image.resize(normalized_image, target_size)

    # Step 4: Apply ImageNet normalization (subtract mean and divide by std)
    preprocessed_image = (resized_image - IMAGENET_MEAN) / IMAGENET_STD

    return preprocessed_image
