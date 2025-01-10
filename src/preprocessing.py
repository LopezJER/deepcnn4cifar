import tensorflow as tf
import numpy as np


def preprocess_image_for_vgg16(image, target_size):
    """
    Preprocess a single image for VGG16:
    - Reshape, normalize, and resize.

    Args:
        image: Flat array of shape (3072,).
        target_size: Tuple specifying the desired image resolution.

    Returns:
        Preprocessed image of shape target_size + (3,).
    """
    reshaped = image.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.float32) / 255.0
    resized = tf.image.resize(reshaped, target_size)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (resized - mean) / std
