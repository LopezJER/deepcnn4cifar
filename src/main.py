import sys
import os
import tensorflow as tf
from src.utils import unpickle
from src.preprocessing import preprocess_image_for_vgg16
from config import DATA_DIR, BATCH_SIZE, TARGET_SIZE

# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_cifar10_data(batch_type="train"):
    """
    Load CIFAR-10 data directly from disk in batches.

    Args:
        batch_type: Type of dataset to load ("train" or "test").

    Returns:
        Generator yielding (images, labels) tuples.
    """
    # Get the appropriate file paths
    batch_files = (
        [os.path.join(DATA_DIR, f"data_batch_{i}") for i in range(1, 6)]
        if batch_type == "train"
        else [os.path.join(DATA_DIR, "test_batch")]
    )

    # Load data from each file
    for file_path in batch_files:
        batch = unpickle(file_path)
        images = batch[b"data"]
        labels = batch[b"labels"]
        yield images, labels


def create_tf_dataset(batch_type="train"):
    """
    Create a TensorFlow dataset for CIFAR-10 with on-the-fly preprocessing.

    Args:
        batch_type: Type of dataset to load ("train" or "test").

    Returns:
        A TensorFlow dataset object.
    """

    def generator():
        for images, labels in load_cifar10_data(batch_type):
            for img, label in zip(images, labels):
                yield preprocess_image_for_vgg16(img, TARGET_SIZE), label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def main():
    """Main pipeline to create and test CIFAR-10 dataset for VGG16 preprocessing."""
    # Create training dataset
    train_dataset = create_tf_dataset(batch_type="train")

    # Test dataset creation
    print("Iterating through a batch of preprocessed data...")
    for images, labels in train_dataset.take(1):
        print(f"Batch image shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")


if __name__ == "__main__":
    main()
