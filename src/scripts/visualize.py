import os
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from torchvision import datasets, transforms
from src.utils.load_data import (
    calculate_dataset_statistics,
)

from src.core.gradcam import GradCAM
from src.core.config import paths
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def visualize_cifar10_with_labels(
    dataset,
    classes,
    num_examples=5,
    save_path="outputs/cifar10_visualization.png",
):
    """
    Visualizes the CIFAR-10 dataset with each row corresponding to a class
    and 5 examples per class. Only the original images (without any transformations) are shown.

    Parameters:
        dataset: The CIFAR-10 dataset (torchvision.datasets).
        classes: List of class names in CIFAR-10.
        num_examples: Number of examples per class (default: 5).
        save_path: Path to save the generated visualization.
    """
    try:
        num_classes = len(classes)
        fig, axes = plt.subplots(
            num_classes,
            num_examples + 1,
            figsize=((num_examples + 1) * 2, num_classes * 2),
        )
        fig.suptitle(
            "CIFAR-10 Dataset: 10 Classes with 5 Examples Each",
            fontsize=20,
            fontweight="bold",
        )

        # Dictionary to track examples per class
        examples_collected = {cls: 0 for cls in range(num_classes)}

        for img, label in dataset:
            if examples_collected[label] < num_examples:
                row = label
                col = (
                    examples_collected[label] + 1
                )  # Shift by 1 for the class name column
                ax = axes[row, col]

                # Convert the PIL image to numpy array for plotting
                img = np.array(img)  # PIL to numpy

                ax.imshow(img)
                ax.axis("off")
                examples_collected[label] += 1

                # Add class name to the first column of each row
                if col == 1:
                    class_ax = axes[row, 0]
                    class_ax.axis("off")
                    class_ax.text(
                        0.5,
                        0.5,
                        classes[row],
                        fontsize=14,
                        ha="center",
                        va="center",
                        rotation=0,
                        fontweight="bold",
                    )

            # Break when we have enough examples
            if all(v >= num_examples for v in examples_collected.values()):
                break

        # Turn off all the unused axes
        for row in range(num_classes):
            for col in range(num_examples + 1):
                if col > examples_collected[row]:
                    axes[row, col].axis("off")

        # Save and display the visualization
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Save the figure
        logger.info(f"Visualization saved to {save_path}")
        plt.tight_layout(
            rect=[0, 0, 1, 0.98]
        )  # Adjust layout to make space for the title
        plt.show()

    except Exception as e:
        logger.error(f"Error in visualize_cifar10_with_labels: {e}")


def plot_image_transformation(ax, image, title):
    """
    Plots a single image with a title.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        image (numpy array): The image to be displayed.
        title (str): Title for the image.
    """
    ax.imshow(image)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def visualize_image_transformations():
    """
    Visualizes CIFAR-10 images before and after transformations.
    Shows the original image, resized image, and normalized image.
    """
    try:
        # Load CIFAR-10 original dataset without transformations
        logger.info("Loading original CIFAR-10 dataset...")
        original_dataset = datasets.CIFAR10(root="./data", train=True, download=True)

        # Define transformations for resized and normalized images
        dataset_mean, dataset_std = calculate_dataset_statistics()

        resized_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        normalized_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=dataset_mean.tolist(), std=dataset_std.tolist()
                ),
            ]
        )

        # Select one image per class using the same indices for both datasets
        class_names = original_dataset.classes
        images_per_class = {}

        for idx in range(len(original_dataset)):
            image, label = original_dataset[idx]
            if label not in images_per_class:
                images_per_class[label] = idx
            if len(images_per_class) == len(class_names):
                break

        fig, axes = plt.subplots(
            len(class_names),
            3,
            figsize=(12, 4 * len(class_names)),
            constrained_layout=True,
        )

        for idx, (label, img_idx) in enumerate(images_per_class.items()):
            # Original image
            original_image = original_dataset[img_idx][0]

            # Resized image
            resized_image = resized_transform(original_image)

            # Normalized image
            normalized_image = normalized_transform(original_image)

            # Convert images to display format (H, W, C)
            resized_image_np = resized_image.permute(1, 2, 0).numpy()
            normalized_image_np = normalized_image.permute(1, 2, 0).numpy()

            plot_image_transformation(
                axes[idx, 0],
                np.array(original_image),
                f"Original: {class_names[label]}",
            )

            plot_image_transformation(
                axes[idx, 1], resized_image_np, f"Resized: {class_names[label]}"
            )
            plot_image_transformation(
                axes[idx, 2], normalized_image_np, f"Normalized: {class_names[label]}"
            )

        fig.suptitle(
            "CIFAR-10 Transformations: Original, Resized, and Normalized",
            fontsize=18,
            fontweight="bold",
        )

        # Save the visualization
        output_path = os.path.join(
            paths["outputs_dir"], "cifar10_image_transformations.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")  # Save the figure
        logger.info(f"Visualization saved to {output_path}")

        # Show the plot
        plt.show()

    except Exception as e:
        logger.error(f"Error in visualize_image_transformations: {e}")


# Function to visualize top mistakes
def visualize_top_mistakes(cm, class_names, top_n=5):
    """
    Visualizes the top model mistakes.
    """
    assert isinstance(cm, np.ndarray), "Confusion matrix must be a numpy array"
    assert isinstance(class_names, list), "Class names must be a list"
    assert isinstance(top_n, int) and top_n > 0, "top_n must be a positive integer"

    try:
        np.fill_diagonal(cm, 0)
        mistakes = [
            (class_names[i], class_names[j], cm[i, j])
            for i in range(len(class_names))
            for j in range(len(class_names))
            if cm[i, j] > 0
        ]
        mistakes = sorted(mistakes, key=lambda x: x[2], reverse=True)[:top_n]
        labels = [f"{mistake[0]} → {mistake[1]}" for mistake in mistakes]
        values = [mistake[2] for mistake in mistakes]
        plt.figure(figsize=(10, 6))
        plt.barh(labels, values, color="salmon")
        plt.xlabel("Number of Mistakes")
        plt.ylabel("Misclassification Pairs")
        plt.title(f"Top {top_n} Model Mistakes")
        plt.gca().invert_yaxis()
        plt.show()
        return mistakes
    except Exception as e:
        logger.error(f"Error in visualize_top_mistakes: {e}")
        return []


# Function to visualize grouped misclassified images
def visualize_mistakes_images_grouped_with_row_titles(
    true_labels,
    predicted_labels,
    test_dataset,
    top_mistakes,
    class_names,
    images_per_category=5,
):
    """
    Visualizes grouped misclassified images.
    """
    assert isinstance(true_labels, np.ndarray), "true_labels must be a numpy array"
    assert isinstance(
        predicted_labels, np.ndarray
    ), "predicted_labels must be a numpy array"
    assert isinstance(
        test_dataset, torch.utils.data.Dataset
    ), "test_dataset must be a valid dataset object"
    assert isinstance(top_mistakes, list), "top_mistakes must be a list"
    assert isinstance(class_names, list), "class_names must be a list"
    assert (
        isinstance(images_per_category, int) and images_per_category > 0
    ), "images_per_category must be a positive integer"

    try:
        rows = len(top_mistakes)
        cols = images_per_category + 1
        plt.figure(figsize=(15, 3 * rows))

        for row_idx, (true_class, predicted_class, _) in enumerate(top_mistakes):
            true_class_idx = class_names.index(true_class)
            predicted_class_idx = class_names.index(predicted_class)
            misclassified_images = []

            for i in range(len(true_labels)):
                if true_labels[i] == true_class_idx and predicted_labels[i] == predicted_class_idx:
                    misclassified_images.append(test_dataset[i][0])

                if len(misclassified_images) == images_per_category:
                    break

            # Plot the misclassified images
            for col_idx, img in enumerate(misclassified_images):
                ax = plt.subplot(rows, cols, row_idx * cols + col_idx + 2)
                ax.imshow(img)
                ax.axis("off")
                if col_idx == 0:
                    ax.set_title(
                        f"{true_class} → {predicted_class}", fontsize=12, fontweight="bold"
                    )
            if row_idx == 0:
                ax.set_title("Misclassified Images", fontsize=14)

        # Save and display the grouped misclassified images
        output_path = os.path.join(
            paths["outputs_dir"], "misclassified_images_grouped.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")  # Save the figure
        logger.info(f"Visualization saved to {output_path}")
        plt.show()

    except Exception as e:
        logger.error(f"Error in visualize_mistakes_images_grouped_with_row_titles: {e}")

# Function to visualize filters (cut off in the snippet, hence I will provide an outline of how it could be handled)
def visualize_filters(model, layer_name, output_path):
    """
    Visualizes filters of a particular convolutional layer.
    """
    try:
        layer = dict(model.named_modules())[layer_name]
        filters = layer.weight.data.cpu().numpy()

        # Plot the filters
        n_filters = filters.shape[0]
        n_cols = 8
        n_rows = (n_filters // n_cols) + 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
        for i in range(n_filters):
            ax = axes[i // n_cols, i % n_cols]
            ax.imshow(filters[i, 0, :, :], cmap="gray")
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Filters saved to {output_path}")
        plt.show()

    except Exception as e:
        logger.error(f"Error in visualize_filters: {e}")
