import os
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from sklearn.metrics import confusion_matrix
from src.utils.load_data import get_cifar_dataloaders
from src.utils.load_model import load_model, setup_device
from src.core.gradcam import GradCAM
from src.core.config import num_classes


def visualize_image_transformations(dataloaders):
    """
    Visualizes image transformations including original, resized, and normalized images.

    Parameters:
    - dataloaders (dict): Dictionary of DataLoaders for 'train', 'val', and optionally 'test'.

    Returns:
    - Displays transformed images.
    """
    assert isinstance(dataloaders, dict), "Expected dataloaders to be a dictionary"
    assert "train" in dataloaders, "Dataloader dictionary must contain 'train' key"

    try:
        train_loader = dataloaders["train"]
        train_data = train_loader.dataset.dataset
        train_mean = train_data.transform.transforms[-1].mean
        train_std = train_data.transform.transforms[-1].std
        images_per_class = {}
        classes = train_data.classes
        for image, label in train_loader.dataset:
            if label not in images_per_class:
                images_per_class[label] = (image, label)
            if len(images_per_class) == num_classes:
                break

        fig, axes = plt.subplots(
            num_classes, 3, figsize=(12, 4 * num_classes), constrained_layout=True
        )
        for idx, (label, (image, _)) in enumerate(images_per_class.items()):
            image_original = np.asarray(image)
            image_resized = image.permute(1, 2, 0).numpy()
            image_normalized_display = np.clip(
                image_resized * train_std + train_mean, 0, 1
            )
            axes[idx, 0].imshow(image_original)
            axes[idx, 0].set_title(f"Original: {classes[label]}", fontsize=12)
            axes[idx, 0].axis("off")
            axes[idx, 1].imshow(image_resized)
            axes[idx, 1].set_title("Resized", fontsize=12)
            axes[idx, 1].axis("off")
            axes[idx, 2].imshow(image_normalized_display)
            axes[idx, 2].set_title("Normalized", fontsize=12)
            axes[idx, 2].axis("off")
        fig.suptitle(
            "Visualization of Images: Original, Resized, and Normalized",
            fontsize=18,
            fontweight="bold",
        )
        plt.show()
    except Exception as e:
        print(f"Error in visualize_image_transformations: {e}")


# Function to generate predictions and true labels
def get_predictions(model, dataloader, device):
    """
    Generates predictions and true labels.
    """
    assert model is not None, "Model cannot be None"
    assert dataloader is not None, "Dataloader cannot be None"
    assert device is not None, "Device cannot be None"

    try:
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(preds.cpu().numpy())
        return np.array(true_labels), np.array(predicted_labels)
    except Exception as e:
        print(f"Error in get_predictions: {e}")
        return np.array([]), np.array([])


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
            default="conv2d_block3.2",
            help="Target layer for Grad-CAM (default: conv2d_block3.2)",
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
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        # Run Visualization for Dataset Transformations
        try:
            visualize_image_transformations(dataloaders)
        except Exception as e:
            print(f"Error in visualize_image_transformations: {e}")

        test_loader = dataloaders["test"]
        test_dataset = test_loader.dataset

        # Generate predictions
        try:
            true_labels, predicted_labels = get_predictions(model, test_loader, device)
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
                true_labels, predicted_labels, test_dataset, top_mistakes, class_names
            )
        except Exception as e:
            print(f"Error in visualize_mistakes_images_grouped_with_row_titles: {e}")

        # Run Grad-CAM for Dataset Images
        if args.dataset:
            print(f"Running Grad-CAM on {args.num_samples} dataset images...")
            try:
                visualize_dataset_with_gradcam(
                    model,
                    test_dataset,
                    args.target_class,
                    args.layer,
                    device,
                    args.num_samples,
                )
            except Exception as e:
                print(f"Error in visualize_dataset_with_gradcam: {e}")

        # Generate Architecture Visualizations
        try:
            generate_vgg_architecture("vgg16")
            generate_vgg_architecture("vgg11")
        except Exception as e:
            print(f"Error in generate_vgg_architecture: {e}")

    except Exception as e:
        print(f"Unexpected error in main execution: {e}")


if __name__ == "__main__":
    main()
