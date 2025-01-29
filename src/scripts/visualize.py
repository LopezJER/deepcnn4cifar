import os
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
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
    train_loader = dataloaders["train"]

    # Get dataset mean and std from train_loader
    train_data = (
        train_loader.dataset.dataset
    )  # Accessing the raw dataset inside TransformDataset
    train_mean = train_data.transform.transforms[-1].mean
    train_std = train_data.transform.transforms[-1].std

    # Sample images from different classes
    images_per_class = {}
    classes = train_data.classes  # Extract class names from dataset
    for image, label in train_loader.dataset:
        if label not in images_per_class:
            images_per_class[label] = (image, label)
        if len(images_per_class) == num_classes:
            break

    # Create the figure and subplots
    fig, axes = plt.subplots(
        num_classes, 3, figsize=(12, 4 * num_classes), constrained_layout=True
    )

    for idx, (label, (image, _)) in enumerate(images_per_class.items()):
        # Original image
        image_original = np.asarray(image)

        # Resized image
        image_resized = image.permute(1, 2, 0).numpy()

        # Normalized image
        image_normalized = image.permute(1, 2, 0).numpy()
        image_normalized_display = (
            image_normalized * train_std + train_mean
        )  # De-normalize for visualization
        image_normalized_display = np.clip(image_normalized_display, 0, 1)

        # Plot original
        axes[idx, 0].imshow(image_original)
        axes[idx, 0].set_title(f"Original: {classes[label]}", fontsize=12)
        axes[idx, 0].axis("off")

        # Plot resized
        axes[idx, 1].imshow(image_resized)
        axes[idx, 1].set_title("Resized", fontsize=12)
        axes[idx, 1].axis("off")

        # Plot normalized
        axes[idx, 2].imshow(image_normalized_display)
        axes[idx, 2].set_title("Normalized", fontsize=12)
        axes[idx, 2].axis("off")

    # Add an overall title to the figure
    fig.suptitle(
        "Visualization of Images: Original, Resized, and Normalized",
        fontsize=18,
        fontweight="bold",
    )
    plt.show()


# Function to generate predictions and true labels
def get_predictions(model, dataloader, device):
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


# Function to visualize top mistakes
def visualize_top_mistakes(cm, class_names, top_n=5):
    np.fill_diagonal(cm, 0)  # Ignore correct predictions

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


# Function to visualize grouped misclassified images
def visualize_mistakes_images_grouped_with_row_titles(
    true_labels,
    predicted_labels,
    test_dataset,
    top_mistakes,
    class_names,
    images_per_category=5,
):
    rows = len(top_mistakes)
    cols = images_per_category + 1

    plt.figure(figsize=(15, 3 * rows))

    for row_idx, (true_class, predicted_class, _) in enumerate(top_mistakes):
        true_class_idx = class_names.index(true_class)
        predicted_class_idx = class_names.index(predicted_class)
        mistake_indices = np.where(
            (true_labels == true_class_idx) & (predicted_labels == predicted_class_idx)
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


# Visualize Filters
def visualize_filters(model, layers, num_filters=8):
    """
    Visualizes filters from multiple layers.

    Parameters:
        model: Trained model instance.
        layers: List of layer names to visualize filters from.
        num_filters: Number of filters to visualize from each layer.
    """
    num_layers = len(layers)
    fig, axes = plt.subplots(num_layers, num_filters, figsize=(20, 3 * num_layers))

    for layer_idx, layer_name in enumerate(layers):
        layer = dict(model.named_modules())[layer_name]
        filters = layer.weight.data.clone().cpu()

        # Normalize filters to [0, 1]
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


# Visualize Heatmap
def visualize_heatmap(img, cam, alpha=0.5):
    """
    Overlays a heatmap on an image.

    Parameters:
        img: Input image (HxWxC, numpy array).
        cam: Class Activation Map (CAM).
        alpha: Transparency of the overlay.
    """
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.uint8(255 * img), 1 - alpha, heatmap, alpha, 0)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Heatmap Visualization")
    plt.show()


def visualize_with_gradcam(model, test_dataset, target_class, target_layer, device):
    grad_cam = GradCAM(model, target_layer)

    # Select an image from the dataset
    img, _ = test_dataset[0]
    input_image = img.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Generate CAM
    cam = grad_cam.generate_cam(input_image, target_class)

    # Visualize heatmap
    visualize_heatmap(img.permute(1, 2, 0).numpy(), cam, alpha=0.5)


def generate_vgg_architecture(architecture="vgg11"):
    """
    Compiles the LaTeX file for the given VGG architecture (VGG11 or VGG16) and generates a PDF.

    Parameters:
        architecture (str): "vgg16" or "vgg11".
    """
    # Define paths for LaTeX source and output directory
    tex_file = (
        f"./core/assets/custom_{architecture}.tex"  # Access LaTeX from assets folder
    )
    output_dir = "./outputs"  # Where the PDFs will be saved
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    pdf_filename = os.path.join(output_dir, f"{architecture}_architecture.pdf")

    # Check if LaTeX file exists
    if not os.path.isfile(tex_file):
        print(f"Error: {tex_file} not found!")
        return

    # Compile the LaTeX file into a PDF using pdflatex
    try:
        subprocess.run(
            ["pdflatex", "-output-directory", output_dir, tex_file], check=True
        )
        print(f"PDF generated successfully: {pdf_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling LaTeX file {tex_file}: {e}")


# Example usage
if __name__ == "__main__":
    # Load device
    device = setup_device()

    # Load datasets
    dataloaders = get_cifar_dataloaders(include_test=True)

    # Run Visualization
    visualize_image_transformations(dataloaders)

    test_loader = dataloaders["test"]
    test_dataset = test_loader.dataset

    # Load model
    model = load_model()
    model.to(device)
    model.eval()

    # Generate predictions
    true_labels, predicted_labels = get_predictions(model, test_loader, device)

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    class_names = test_dataset.classes

    # Visualize top mistakes
    top_mistakes = visualize_top_mistakes(cm, class_names, top_n=5)

    # Visualize grouped misclassified images
    visualize_mistakes_images_grouped_with_row_titles(
        true_labels, predicted_labels, test_dataset, top_mistakes, class_names
    )

    # Visualize GradCAM
    target_layer = "conv2d_block3.2"  # Example target layer
    target_class = 5  # Example target class
    visualize_with_gradcam(model, test_dataset, target_class, target_layer, device)

    generate_vgg_architecture("vgg16")
    generate_vgg_architecture("vgg11")
