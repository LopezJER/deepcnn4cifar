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
from src.core.config import model_setup, debug, class_names,paths
from torch.utils.data import DataLoader
import sys


num_classes = model_setup["num_classes"]


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
        plt.savefig(paths['outputs_dir'], bbox_inches='tight', dpi=300)
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
    tex_file = f"assets/{architecture}_template.tex"
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

def get_debug_dataloader(test_loader):
    """Extracts a smaller subset of the dataset for debugging and returns new DataLoaders."""

    def extract_subset(loader, num_images):
        data, labels = [], []
        try: 
            for i, (images, lbls) in enumerate(loader):
                print(i)
                data.extend(images)
                labels.extend(lbls)
                if len(data) >= num_images:
                    return list(zip(data[:num_images], labels[:num_images]))
            return list(zip(data, labels))
        except Exception as e:
            print(e)

    print(f"Debug mode: Evaluating with {debug['test_size']} images")
    
    test_subset = extract_subset(test_loader, debug['test_size'])
    test_loader = DataLoader(test_subset, batch_size=debug['batch_size'], shuffle=False)
    
    return test_loader

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
            "--architecture",
            action="store_true",
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

        if len(sys.argv) == 1:  # Only the script name is present, no additional args
            print("No arguments provided. Please refer to README.md for instructions.")
            parser.print_help()  # Show help message with usage instructions
            sys.exit(1)

        args = parser.parse_args()
        device = setup_device()
        assert device is not None, "Device setup failed."

        print("Loading model...")
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

        if args.arch:
            generate_vgg_architecture("vgg16")
            generate_vgg_architecture("vgg11")

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
    print("Hi")
    main()

    