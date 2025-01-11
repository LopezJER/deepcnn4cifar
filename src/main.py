import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

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
    batch_files = (
        [
            os.path.join(DATA_DIR, f"data_batch_{i}") for i in range(1, 6)
        ]  # Training batches
        if batch_type == "train"
        else [os.path.join(DATA_DIR, "test_batch")]  # Test batch
    )

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
        for images, labels in load_cifar10_data(batch_type):
            for img, label in zip(images, labels):
                preprocessed_img = preprocess_image_for_vgg16(img, TARGET_SIZE)
                yield preprocessed_img, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def visualize_sample_images(dataset, title):
    """
    Visualize a few images from the dataset.

    Args:
        dataset: TensorFlow dataset containing (images, labels).
        title: Title of the visualization plot.
    """
    plt.figure(figsize=(10, 5))
    for images, labels in dataset.take(1):  # Take one batch
        for i in range(5):  # Visualize 5 images
            plt.subplot(1, 5, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(f"Label: {labels[i].numpy()}")
            plt.axis("off")
        break
    plt.suptitle(title)
    plt.show()


def main():
    """
    Main pipeline to create and test CIFAR-10 dataset for VGG16 preprocessing.
    """
    try:
        print("Creating CIFAR-10 training dataset...")
        train_dataset = create_tf_dataset(batch_type="train")

        print("Iterating through a batch of preprocessed training data...")
        for images, labels in train_dataset.take(1):
            print(
                f"Training Batch - Image shape: {images.shape}, Labels shape: {labels.shape}"
            )
            assert images.shape[1:] == (224, 224, 3), "Image dimensions are incorrect."

        visualize_sample_images(train_dataset, title="Training Dataset")

        print("Creating CIFAR-10 test dataset...")
        test_dataset = create_tf_dataset(batch_type="test")

        print("Iterating through a batch of preprocessed test data...")
        for images, labels in test_dataset.take(1):
            print(
                f"Test Batch - Image shape: {images.shape}, Labels shape: {labels.shape}"
            )
            assert images.shape[1:] == (224, 224, 3), "Image dimensions are incorrect."

        visualize_sample_images(test_dataset, title="Test Dataset")

        print("Dataset creation and preprocessing completed successfully.")

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")


if __name__ == "__main__":
    main()
