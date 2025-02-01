import os
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import argparse
from sklearn.metrics import confusion_matrix
from PIL import Image
from torchvision import datasets, transforms
from src.utils.load_data import (
    get_cifar_dataloaders,
    get_debug_dataloaders,
    calculate_dataset_statistics,
)
from src.utils.load_model import load_model, setup_device
from src.core.gradcam import GradCAM
from src.core.config import model_setup, debug, class_names, paths
from src.scripts.evaluate import get_predictions


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
        print(f"Visualization saved to {save_path}")
        plt.tight_layout(
            rect=[0, 0, 1, 0.98]
        )  # Adjust layout to make space for the title
        plt.show()

    except Exception as e:
        print(f"Error in visualize_cifar10_with_labels: {e}")


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
        print("Loading original CIFAR-10 dataset...")
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
        print(f"Visualization saved to {output_path}")

        # Show the plot
        plt.show()

    except Exception as e:
        print(f"Error in visualize_image_transformations: {e}")


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
        print(f"Error in visualize_top_mistakes: {e}")
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
            mistake_indices = np.where(
                (true_labels == true_class_idx)
                & (predicted_labels == predicted_class_idx)
            )[0]

            plt.subplot(rows, cols, row_idx * cols + 1)
            plt.text(
                0.5,
                0.5,
                f"True: {true_class}\nPredicted: {predicted_class}",
                fontsize=12,
                ha="center",
                va="center",
                fontweight="bold",
            )
            plt.axis("off")

            for col_idx in range(min(images_per_category, len(mistake_indices))):
                image, _ = test_dataset[mistake_indices[col_idx]]
                image = image.permute(1, 2, 0).numpy()
                mean = np.array([0.49139968, 0.48215841, 0.44653091])
                std = np.array([0.24703223, 0.24348513, 0.26158784])
                image = std * image + mean
                image = np.clip(image, 0, 1)

                plt.subplot(rows, cols, row_idx * cols + col_idx + 2)
                plt.imshow(image)
                plt.axis("off")

        plt.suptitle("Misclassified Images by Category", fontsize=16, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.show()
    except Exception as e:
        print(f"Error in visualize_mistakes_images_grouped_with_row_titles: {e}")


# Visualize Filters
def visualize_filters(model, layers, num_filters=8):
    """
    Visualizes filters from multiple layers.
    """
    assert isinstance(layers, list), "Layers should be a list of layer names"
    assert all(
        isinstance(layer, str) for layer in layers
    ), "All layer names should be strings"
    assert (
        isinstance(num_filters, int) and num_filters > 0
    ), "num_filters should be a positive integer"

    try:
        num_layers = len(layers)
        fig, axes = plt.subplots(num_layers, num_filters, figsize=(20, 3 * num_layers))

        for layer_idx, layer_name in enumerate(layers):
            if layer_name not in dict(model.named_modules()):
                raise ValueError(f"Layer '{layer_name}' not found in model")

            layer = dict(model.named_modules())[layer_name]
            if not hasattr(layer, "weight") or layer.weight is None:
                raise ValueError(
                    f"Layer '{layer_name}' does not have a valid weight attribute"
                )

            filters = layer.weight.data.clone().cpu()
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)

            for filter_idx in range(num_filters):
                ax = axes[layer_idx, filter_idx] if num_layers > 1 else axes[filter_idx]
                ax.imshow(filters[filter_idx, 0].numpy(), cmap="gray")
                ax.axis("off")
                if filter_idx == 0:
                    ax.set_ylabel(layer_name, fontsize=12)

        plt.suptitle("Filters from Different Layers", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in visualize_filters: {e}")


# Visualize Heatmap
def visualize_heatmap(img, cam, alpha=0.5):
    """
    Overlays a heatmap on an image.

    Parameters:
        img: Input image (HxWxC, numpy array).
        cam: Class Activation Map (CAM).
        alpha: Transparency of the overlay.
    """
    assert isinstance(img, np.ndarray), "Input image must be a numpy array"
    assert isinstance(cam, np.ndarray), "CAM must be a numpy array"
    assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"

    try:
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(np.uint8(255 * img), 1 - alpha, heatmap, alpha, 0)

        plt.imshow(overlay)
        plt.axis("off")
        plt.title("Grad-CAM Heatmap")
        plt.show()
    except Exception as e:
        print(f"Error in visualize_heatmap: {e}")


# Grad-CAM Visualization for a Single Image
def visualize_with_gradcam(model, img_path, target_class, target_layer, device):
    """
    Generate and visualize Grad-CAM heatmap for a specific image.
    """
    assert isinstance(img_path, str), "Image path must be a string"
    assert isinstance(target_class, int), "Target class must be an integer"
    assert isinstance(target_layer, str), "Target layer must be a string"

    try:
        grad_cam = GradCAM(model, target_layer)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(
                f"Error: Unable to read image at '{img_path}'. Please check the file path."
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        input_tensor = torch.tensor(input_img).permute(2, 0, 1).unsqueeze(0).to(device)
        cam = grad_cam.generate_cam(input_tensor, target_class)
        visualize_heatmap(img, cam, alpha=0.5)
    except Exception as e:
        print(f"Error in visualize_with_gradcam: {e}")


def visualize_dataset_with_gradcam(
    model, dataset, target_class, target_layer, device, num_samples=5
):
    """
    Visualize Grad-CAM for multiple images from the dataset.
    """
    assert isinstance(target_class, int), "Target class must be an integer"
    assert isinstance(target_layer, str), "Target layer must be a string"
    assert (
        isinstance(num_samples, int) and num_samples > 0
    ), "num_samples must be a positive integer"

    try:
        grad_cam = GradCAM(model, target_layer)
        fig, axes = plt.subplots(num_samples, 1, figsize=(6, num_samples * 3))

        for idx in range(num_samples):
            img, label = dataset[idx]  # Get an image from the dataset
            input_tensor = img.unsqueeze(0).to(device)  # Convert to batch format

            # Generate CAM
            cam = grad_cam.generate_cam(input_tensor, target_class)

            # Convert image to numpy for visualization
            img_np = img.permute(1, 2, 0).numpy()

            # Overlay heatmap and display
            cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(np.uint8(255 * img_np), 1 - 0.5, heatmap, 0.5, 0)

            axes[idx].imshow(overlay)
            axes[idx].axis("off")
            axes[idx].set_title(f"Sample {idx+1} - True Class: {label}")

        plt.show()
    except Exception as e:
        print(f"Error in visualize_dataset_with_gradcam: {e}")


def generate_vgg_architecture(architecture="vgg11"):
    """
    Compiles the LaTeX file for the given VGG architecture (VGG11 or VGG16) and generates a PDF.

    Parameters:
        architecture (str): "vgg16" or "vgg11".
    """
    assert architecture in [
        "vgg11",
        "vgg16",
    ], "Architecture must be either 'vgg11' or 'vgg16'"

    tex_file = f"./core/assets/custom_{architecture}.tex"
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    pdf_filename = os.path.join(output_dir, f"{architecture}_architecture.pdf")

    if not os.path.isfile(tex_file):
        raise FileNotFoundError(f"Error: {tex_file} not found!")

    try:
        subprocess.run(
            ["pdflatex", "-output-directory", output_dir, tex_file], check=True
        )
        print(f"PDF generated successfully: {pdf_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling LaTeX file {tex_file}: {e}")


"""
def main():
    try:
        # Argument Parser
        parser = argparse.ArgumentParser(
            description="Visualize Grad-CAM heatmaps for a single image or dataset."
        )
        parser.add_argument(
            "--image",
            type=str,
            help="Path to the input image (e.g., --image path/to/image.png)",
        )
        parser.add_argument(
            "--dataset",
            action="store_true",
            help="Run Grad-CAM on dataset images instead of a single image",
        )
        parser.add_argument(
            "--layer",
            type=str,
            default="features.10",
            help="Target layer for Grad-CAM (default: features.10)",
        )
        parser.add_argument(
            "--target_class",
            type=int,
            default=5,
            help="Target class index (default: 5)",
        )
        parser.add_argument(
            "--num_samples",
            type=int,
            default=5,
            help="Number of dataset samples to visualize (default: 5)",
        )

        args = parser.parse_args()

        # Load device
        device = setup_device()
        assert device is not None, "Device setup failed."

        # Load model
        model = load_model()
        assert model is not None, "Model loading failed."
        model.to(device)
        model.eval()

        # Run Grad-CAM for a Single Image (if specified)
        if args.image:
            print(f"Running Grad-CAM for image: {args.image}")
            try:
                visualize_with_gradcam(
                    model, args.image, args.target_class, args.layer, device
                )
            except Exception as e:
                print(f"Error in visualize_with_gradcam: {e}")
            return  # Exit after single image Grad-CAM to avoid unnecessary computations

        # Load dataset
        try:
            dataloaders = get_cifar_dataloaders(include_test=True)
            assert "test" in dataloaders, "Test dataloader is missing."

            # Apply debug mode if enabled
            if debug["on"]:
                print("Debug mode enabled: Using a smaller test dataset.")
                dataloaders["train"], dataloaders["val"], dataloaders["test"] = (
                    get_debug_dataloaders(
                        train_loader=dataloaders.get("train"),
                        val_loader=dataloaders.get("val"),
                        test_loader=dataloaders.get("test"),
                    )
                )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        # Run Visualization for Dataset Transformations
        try:
            visualize_image_transformations()
        except Exception as e:
            print(f"Error in visualize_image_transformations: {e}")

        test_loader = dataloaders["test"]
        test_dataset = test_loader.dataset

        # Generate predictions
        try:
            true_labels, predicted_labels = get_predictions(
                model, dataloaders["test"], device
            )
        except Exception as e:
            print(f"Error in get_predictions: {e}")
            return

        # Generate confusion matrix
        try:
            cm = confusion_matrix(true_labels, predicted_labels)
            class_names = test_dataset.classes
        except Exception as e:
            print(f"Error computing confusion matrix: {e}")
            return

        # Visualize top mistakes
        try:
            top_mistakes = visualize_top_mistakes(cm, class_names, top_n=5)
        except Exception as e:
            print(f"Error in visualize_top_mistakes: {e}")
            return

        # Visualize grouped misclassified images
        try:
            visualize_mistakes_images_grouped_with_row_titles(
                true_labels,
                predicted_labels,
                dataloaders["test"].dataset,
                top_mistakes,
                class_names,
            )
        except Exception as e:
            print(f"Error in visualize_mistakes_images_grouped_with_row_titles: {e}")

        # **NEW: Generate and Plot Heatmap**
        try:
            print("Generating Grad-CAM heatmap for a sample image...")
            sample_idx = 0
            sample_image, _ = test_dataset[sample_idx]  # Get first test image
            sample_image = sample_image.unsqueeze(0).to(device)  # Convert to batch

            grad_cam = GradCAM(model, args.layer)
            cam = grad_cam.generate_cam(sample_image, args.target_class)

            sample_image_np = sample_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            visualize_heatmap(sample_image_np, cam)
        except Exception as e:
            print(f"Error in visualize_heatmap: {e}")

        # **NEW: Visualize Filters**
        try:
            visualize_filters(
                model,
                layers=[
                    "features.0",
                    "features.5",
                    "features.10",
                ],  # Adjust layers based on model
                num_filters=8,
            )
        except Exception as e:
            print(f"Error in visualize_filters: {e}")

        # Generate Architecture Visualizations
        try:
            generate_vgg_architecture("vgg16")
            generate_vgg_architecture("vgg11")
        except Exception as e:
            print(f"Error in generate_vgg_architecture: {e}")

    except Exception as e:
        print(f"Unexpected error in main execution: {e}")
"""


def main():
    """
    Main function to call visualization functions.
    """
    # Load the original CIFAR-10 dataset (without transformations)
    print("Loading original CIFAR-10 dataset...")
    original_dataset = datasets.CIFAR10(root="./data", train=True, download=True)

    """
    # Call visualization function to visualize CIFAR-10 classes with 5 examples each (using original data)
    visualize_cifar10_with_labels(
        dataset=original_dataset,
        classes=original_dataset.classes,
        num_examples=5,
        save_path=os.path.join(
            paths["outputs_dir"], "cifar10_classes_visualization.png"
        ),
    ) 
    """
    # Call the visualization function for image transformations (original, resized, normalized)
    visualize_image_transformations()


if __name__ == "__main__":
    main()
