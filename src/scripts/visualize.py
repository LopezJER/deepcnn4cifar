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
    get_cifar_dataloaders,
    get_debug_dataloaders,
)
from src.utils.load_model import load_model
from src.core.gradcam import GradCAM
from src.core.config import paths, model_setup, debug, class_names
import logging
from torch.utils.data import DataLoader
from src.scripts.evaluate import evaluate_model
# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def visualize_cifar10_with_labels(classes, num_examples=5, save_path="outputs/cifar10_visualization.png"
):
    """
    Visualizes the CIFAR-10 dataset with each row corresponding to a class
    and 5 examples per class. Displays original CIFAR-10 images (without transformations).
    """
    try:
        # Load CIFAR-10 original dataset without transformations
        logger.info("Loading original CIFAR-10 dataset...")
        dataset = datasets.CIFAR10(root="./data", train=True, download=True)
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

        examples_collected = {cls: 0 for cls in range(num_classes)}

        for img, label in dataset:
            label = int(label)  # Ensure label is an integer

            if examples_collected[label] < num_examples:
                row = label
                col = examples_collected[label] + 1
                ax = axes[row, col]

                # Ensure img is in (H, W, C) format for imshow()
                if isinstance(img, torch.Tensor):
                    img = img.permute(
                        1, 2, 0
                    ).numpy()  # Convert tensor (C, H, W) to (H, W, C)

                ax.imshow(img)
                ax.axis("off")
                examples_collected[label] += 1

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
                        fontweight="bold",
                    )

            if all(v >= num_examples for v in examples_collected.values()):
                break

        for row in range(num_classes):
            for col in range(num_examples + 1):
                if col > examples_collected[row]:
                    axes[row, col].axis("off")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Visualization saved to {save_path}")
        plt.tight_layout(rect=[0, 0, 1, 0.98])
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
    # If image is in range [0, 255], convert to [0,1] for imshow()
    if image.dtype == np.uint8:
        ax.imshow(image)  # Display as is
    else:
        ax.imshow(np.clip(image, 0, 1))  # Only clip for normalized images

    ax.set_title(title, fontsize=12)
    ax.axis("off")


def visualize_image_transformations():
    """
    Visualizes CIFAR-10 images before and after transformations.
    Columns: Original, Resized, and Normalized.
    """
    try:
        logger.info("Loading original CIFAR-10 dataset...")
        original_dataset = datasets.CIFAR10(root="./data", train=True, download=True)

        dataset_mean, dataset_std = calculate_dataset_statistics()

        resized_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224))]
        )
        normalized_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=dataset_mean.tolist(), std=dataset_std.tolist()
                ),
            ]
        )

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
            original_image = original_dataset[img_idx][0]

            # Fix: Convert PIL to RGB before converting to numpy
            original_image_np = np.array(original_image.convert("RGB"))

            resized_image = resized_transform(original_image)
            normalized_image = normalized_transform(original_image)

            resized_image_np = resized_image.permute(1, 2, 0).numpy()
            normalized_image_np = normalized_image.permute(1, 2, 0).numpy()

            plot_image_transformation(
                axes[idx, 0], original_image_np, f"Original: {class_names[label]}"
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

        output_path = os.path.join(
            paths["outputs_dir"], "cifar10_image_transformations.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Visualization saved to {output_path}")

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
        plt.tight_layout()
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
    max_images_per_category=5,
):
    """
    Visualizes grouped misclassified images.
    - Figure-wide title at the top
    - Each row represents one misclassification type
    - Row titles are on the left side of the images
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
        isinstance(max_images_per_category, int) and max_images_per_category > 0
    ), "max_images_per_category must be a positive integer"

    try:
        num_rows = min(len(top_mistakes), 5)  # Always keep 5 rows max
        misclassified_data = []

        # Collect misclassified images
        for true_class, predicted_class, _ in top_mistakes[:num_rows]:
            true_class_idx = class_names.index(true_class)
            predicted_class_idx = class_names.index(predicted_class)
            images = []

            for i in range(len(true_labels)):
                if (
                    true_labels[i] == true_class_idx
                    and predicted_labels[i] == predicted_class_idx
                ):
                    images.append(test_dataset[i][0])  # Extract image
                if len(images) == max_images_per_category:
                    break

            misclassified_data.append((true_class, predicted_class, images))

        # Determine max columns dynamically
        max_cols = (
            max(len(images) for _, _, images in misclassified_data)
            if misclassified_data
            else 1
        )

        fig, axes = plt.subplots(num_rows, max_cols + 1, figsize=(12, 3 * num_rows))
        fig.suptitle("Misclassified Images by Category", fontsize=18, fontweight="bold")

        # Plot misclassified images
        for row_idx, (true_class, predicted_class, images) in enumerate(
            misclassified_data
        ):
            num_cols = len(
                images
            )  # Use actual number of misclassified images for this row

            # Add text label as the first column in the row
            axes[row_idx, 0].text(
                0.5,
                0.5,
                f"True: {true_class}\nPredicted: {predicted_class}",
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
            )
            axes[row_idx, 0].axis("off")  # Hide the subplot axis

            for col_idx in range(max_cols):
                ax = axes[row_idx, col_idx + 1]  # Offset by 1 since col 0 is the label

                if col_idx < num_cols:
                    img = (
                        images[col_idx].permute(1, 2, 0).cpu().numpy()
                    )  # Convert PyTorch tensor to NumPy
                    img = (img - img.min()) / (
                        img.max() - img.min()
                    )  # Normalize for display
                    ax.imshow(img)
                ax.axis("off")

        plt.tight_layout(
            rect=[0.1, 0, 1, 0.96]
        )  # Adjust layout for the figure-wide title
        output_path = os.path.join("outputs", "misclassified_images_grouped.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()

    except Exception as e:
        print(f"Error in visualize_mistakes_images_grouped_with_row_titles: {e}")


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


def run_visualization_pipeline():
    """
    Runs the visualization pipeline using the trained model and dataset.
    Uses debug mode if enabled, loading a reduced dataset.

    This function:
    - Loads the best trained model from Hugging Face.
    - Loads a smaller debug dataset (CIFAR-10) if debug mode is ON.
    - Otherwise, loads the full CIFAR-10 test dataset.
    - Runs visualization functions for dataset and transformations.
    """
    try:
        logger.info("Starting visualization pipeline...")

        # Load the best trained model from Hugging Face
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model = load_model()  # Load trained model from Hugging Face
        model.to(device)  # Move model to GPU if available

        # Check if Debug Mode is ON
        if debug["on"]:
            logger.info("Debug mode enabled: Using a smaller dataset for visualization.")
            dataloaders = get_cifar_dataloaders(include_test=True, test_only=True)

            # Ensure test_loader exists before applying debug mode
            if "test" not in dataloaders or dataloaders["test"] is None:
                logger.error("Error: test_loader not found in dataloaders!")
                raise ValueError("test_loader is None. Cannot proceed with visualization.")

            # Apply Debug Mode to reduce dataset size
            _, _, test_loader = get_debug_dataloaders(test_loader=dataloaders.get("test"))

            # Ensure test_loader is still valid after debug mode
            if test_loader is None:
                logger.error("Error: test_loader is None after applying get_debug_dataloaders()!")
                raise ValueError("test_loader is None. Check get_debug_dataloaders().")

        else:
            # Load the full dataset when not in debug mode
            logger.info("Debug mode OFF: Using full dataset.")
            dataloaders = get_cifar_dataloaders(include_test=True, test_only=True)

            if "test" not in dataloaders or dataloaders["test"] is None:
                logger.error("Error: test_loader not found in dataloaders!")
                raise ValueError("test_loader is None. Cannot proceed with visualization.")

            test_loader = dataloaders["test"]

        '''# Run visualization of CIFAR-10 images
        logger.info("Running visualize_cifar10_with_labels()...")
        visualize_cifar10_with_labels(class_names, num_examples=5, save_path="outputs/cifar10_visualization.png")
        
        # Run visualization of image transformations
        logger.info("Running visualize_image_transformations()...")
        visualize_image_transformations()

        logger.info("Visualization pipeline completed successfully.")'''

        # Run Model Evaluation to Get Predictions and Metrics
        logger.info("Running model evaluation...")
        accuracy, avg_loss, precision, recall, f1, cm, ne, all_probs = evaluate_model(
            model, test_loader, device, num_classes=len(class_names)
        )

        # Visualize Top Mistakes
        logger.info("Running visualize_top_mistakes()...")
        top_mistakes = visualize_top_mistakes(cm, class_names, top_n=5)

        # Visualize Misclassified Images
        logger.info("Running visualize_mistakes_images_grouped_with_row_titles()...")
        true_labels = np.array(ne.argmax(axis=1))  # Convert one-hot to label indices
        predicted_labels = np.array(all_probs.argmax(axis=1))

        visualize_mistakes_images_grouped_with_row_titles(
            true_labels,
            predicted_labels,
            test_loader.dataset,
            top_mistakes,
            class_names,
            max_images_per_category=5,
        )

    except Exception as e:
        logger.error(f"Error in visualization pipeline: {e}")
        raise


# Run when executing `visualize.py` directly
if __name__ == "__main__":
    run_visualization_pipeline()
