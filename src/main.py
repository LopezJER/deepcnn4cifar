import os
import sys
import tensorflow as tf

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import custom modules
from src.utils import unpickle
from src.preprocessing import preprocess_image_for_vgg16
from src.config import DATA_DIR, BATCH_SIZE, TARGET_SIZE


def load_cifar10_data(batch_type="train"):
    """
    Load CIFAR-10 data directly from disk in batches.

    Args:
        batch_type (str): Type of dataset to load ("train" or "test").

    Returns:
        generator: A generator yielding tuples of (images, labels).
    """
    # Determine file paths based on dataset type
    batch_files = (
        [
            os.path.join(DATA_DIR, f"data_batch_{i}") for i in range(1, 6)
        ]  # Training batches
        if batch_type == "train"
        else [os.path.join(DATA_DIR, "test_batch")]  # Test batch
    )

    # Load data from each file
    for file_path in batch_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Batch file not found: {file_path}")

        batch = unpickle(file_path)
        images = batch[b"data"]  # Raw image data
        labels = batch[b"labels"]  # Corresponding labels
        yield images, labels


def create_tf_dataset(batch_type="train"):
    """
    Create a TensorFlow dataset for CIFAR-10 with on-the-fly preprocessing.

    Args:
        batch_type (str): Type of dataset to load ("train" or "test").

    Returns:
        tf.data.Dataset: A TensorFlow dataset object.
    """

    def generator():
        """
        Generator function to preprocess images and yield (image, label) pairs.
        """
        for images, labels in load_cifar10_data(batch_type):
            for img, label in zip(images, labels):
                preprocessed_img = preprocess_image_for_vgg16(img, TARGET_SIZE)
                yield preprocessed_img, label

    # Create a TensorFlow dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),  # Preprocessed image
            tf.TensorSpec(shape=(), dtype=tf.int64),  # Label
        ),
    )

    # Batch and prefetch for efficiency
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def main():
    """
    Main pipeline to create and test CIFAR-10 dataset for VGG16 preprocessing.
    """
    try:
        # Process the training dataset
        print("Creating CIFAR-10 training dataset...")
        train_dataset = create_tf_dataset(batch_type="train")

        # Display a sample batch from the training dataset
        print("Iterating through a batch of preprocessed training data...")
        for images, labels in train_dataset.take(1):
            print(
                f"Training Batch - Image shape: {images.shape}, Labels shape: {labels.shape}"
            )

        # Process the test dataset
        print("Creating CIFAR-10 test dataset...")
        test_dataset = create_tf_dataset(batch_type="test")

        # Display a sample batch from the test dataset
        print("Iterating through a batch of preprocessed test data...")
        for images, labels in test_dataset.take(1):
            print(
                f"Test Batch - Image shape: {images.shape}, Labels shape: {labels.shape}"
            )

        print("Dataset creation and preprocessing completed successfully.")

    except Exception as e:
        # Handle errors gracefully and provide feedback
        print(f"An error occurred during preprocessing: {e}")


if __name__ == "__main__":
    main()
