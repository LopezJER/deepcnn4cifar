import os
import subprocess
import shutil
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import datasets, transforms
from src.utils.load_data import (
    calculate_dataset_statistics,
    get_cifar_dataloaders,
    get_debug_dataloaders,
)
from src.utils.load_model import load_model
from src.core.gradcam import GradCAM
from src.core.config import paths, model_setup, debug, class_names, latex_path
import logging
from src.scripts.evaluate import evaluate_model

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def visualize_cifar10_with_labels(
    classes, num_examples=5, save_path="outputs/cifar10_visualization.png"
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


def plot_image_transformation(
    ax, image, title, is_normalized=False, dataset_mean=None, dataset_std=None
):
    """
    Plots a single image with a title.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        image (numpy array or torch.Tensor): The image to be displayed.
        title (str): Title for the image.
        is_normalized (bool): Whether the image was normalized (if True, it is denormalized for display).
        dataset_mean (list): Mean values for normalization.
        dataset_std (list): Std values for normalization.
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)

    if is_normalized and dataset_mean is not None and dataset_std is not None:
        # Denormalize image for correct visualization
        mean = np.array(dataset_mean)
        std = np.array(dataset_std)
        image = (image * std) + mean
        image = np.clip(image, 0, 1)  # Ensure values remain valid for display

    ax.imshow(image)
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

        # Dynamically compute dataset statistics
        dataset_mean, dataset_std = calculate_dataset_statistics()

        resized_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224))]
        )
        normalized_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=dataset_mean, std=dataset_std),
            ]
        )

        class_names = original_dataset.classes
        images_per_class = {}

        # Select one image per class
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

            # Convert PIL to NumPy (ensures RGB format)
            original_image_np = np.array(original_image.convert("RGB"))

            # Apply transformations
            resized_image = resized_transform(original_image)
            normalized_image = normalized_transform(original_image)

            # Convert tensors to NumPy arrays for visualization
            resized_image_np = resized_image.permute(1, 2, 0).numpy()
            normalized_image_np = normalized_image.permute(1, 2, 0).numpy()

            # Plot original, resized, and normalized (with denormalization)
            plot_image_transformation(
                axes[idx, 0], original_image_np, f"Original: {class_names[label]}"
            )
            plot_image_transformation(
                axes[idx, 1], resized_image_np, f"Resized: {class_names[label]}"
            )
            plot_image_transformation(
                axes[idx, 2],
                normalized_image_np,
                f"Normalized: {class_names[label]}",
                is_normalized=True,
                dataset_mean=dataset_mean,
                dataset_std=dataset_std,
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


def plot_data_split(dataset_path="data", output_path="outputs/data_split.png"):
    """
    Automatically determines the dataset split (Train, Validation, Test) and plots it beautifully.

    Args:
        dataset_path (str): Path where CIFAR-10 is stored/downloaded.
        output_path (str): Path to save the output image.
    """
    # Load CIFAR-10 dataset (Only to determine sizes)
    transform = transforms.ToTensor()
    full_dataset = datasets.CIFAR10(
        root=dataset_path, train=True, download=True, transform=transform
    )

    # Train-Validation Split Ratio (from config)
    train_size = int((1 - model_setup["val_split"]) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Load CIFAR-10 test set
    test_dataset = datasets.CIFAR10(
        root=dataset_path, train=False, download=True, transform=transform
    )
    test_size = len(test_dataset)

    # Data labels and sizes
    labels = ["Train", "Validation", "Test"]
    sizes = [train_size, val_size, test_size]
    colors = [
        "firebrick",
        "teal",
        "m",
    ]  # Modern color palette (Blue, Orange, Green)

    # Plot pie chart with improved aesthetics
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 14, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2, "antialiased": True},
        pctdistance=0.85,  # Moves percentage inside slices
    )

    # Style improvements
    plt.setp(
        autotexts, size=13, weight="bold", color="white"
    )  # Style percentage labels
    plt.setp(texts, size=14)  # Style labels

    # Add circle to make it a donut chart for better aesthetics
    center_circle = plt.Circle((0, 0), 0.70, fc="white")
    fig.gca().add_artist(center_circle)

    # Title settings
    plt.title("Dataset Split", fontsize=18, fontweight="bold", pad=20)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    print(f"Figure saved as {output_path}")
    plt.show()


def generate_vgg_visualization(model_type, output_dir="outputs"):
    """
    Generates the VGG network architecture visualization using LaTeX and moves the output to the specified folder.

    Args:
        model_type (str): The model type, either "vgg11" or "vgg16".
        output_dir (str): Directory to save the generated PDF.
    """

    # Validate model type
    if model_type not in latex_path:
        print(f"Error: Model type '{model_type}' not found in config!")
        return

    # Get the LaTeX file path
    latex_file = os.path.join("assets", latex_path[model_type])

    # Ensure the LaTeX file exists
    if not os.path.exists(latex_file):
        print(f"Error: LaTeX file '{latex_file}' not found!")
        return

    # Ensure pdflatex is installed
    try:
        subprocess.run(
            ["pdflatex", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("Error: 'pdflatex' is not installed. Please install TeX Live or MiKTeX.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract filename without extension
    latex_filename = os.path.basename(latex_file)  # e.g., 'vgg11_template.tex'
    pdf_filename = (
        os.path.splitext(latex_filename)[0] + ".pdf"
    )  # e.g., 'vgg11_template.pdf'

    # Move into assets directory to run LaTeX with correct paths
    assets_dir = os.path.dirname(latex_file)
    os.chdir(assets_dir)  # Change working directory to 'assets/'

    try:
        # Run LaTeX compilation twice for proper rendering
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", latex_filename], check=True
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", latex_filename], check=True
        )
    except subprocess.CalledProcessError:
        print(
            f"Error during LaTeX compilation for '{latex_filename}'. Check your LaTeX code."
        )
        os.chdir("..")  # Return to original directory
        return

    # Move back to the original directory
    os.chdir("..")

    # Move the generated PDF to the output directory
    pdf_source_path = os.path.join(assets_dir, pdf_filename)
    pdf_output_path = os.path.join(output_dir, pdf_filename)

    if os.path.exists(pdf_source_path):
        shutil.move(pdf_source_path, pdf_output_path)
        print(f"PDF successfully generated and moved to: {pdf_output_path}")
    else:
        print(f"Error: '{pdf_filename}' was not generated. Check LaTeX errors.")


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


# Function to denormalize CIFAR-10 images
def denormalize(img, mean, std):
    """
    Converts a normalized image tensor back to its original scale.

    Args:
        img (torch.Tensor): Normalized image tensor (C, H, W).
        mean (tuple): Mean used for normalization (per channel).
        std (tuple): Standard deviation used for normalization (per channel).

    Returns:
        numpy.ndarray: Denormalized image in the range [0,1].
    """
    img = img.clone().cpu().numpy().transpose(1, 2, 0)  # Convert to (H, W, C)
    img = img * std + mean  # Reverse normalization
    img = np.clip(img, 0, 1)  # Ensure values are within valid range
    return img


def visualize_cam_on_image(img, cam, alpha=0.5):
    """
    Overlays the Grad-CAM heatmap on an input image.

    Args:
        img (numpy.ndarray): Original image (H, W, C) in range [0,1].
        cam (numpy.ndarray): Grad-CAM heatmap (H, W).
        alpha (float): Transparency factor.

    Returns:
        numpy.ndarray: Overlayed image.
    """
    # Resize CAM to match the original image size
    cam_resized = cv2.resize(
        cam, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
    )

    # Normalize CAM for visualization
    cam_resized = (cam_resized - cam_resized.min()) / (
        cam_resized.max() - cam_resized.min() + 1e-6
    )

    # Convert CAM to a heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Overlay heatmap onto the original image
    overlay = cv2.addWeighted(np.uint8(img * 255), 1 - alpha, heatmap, alpha, 0)

    return overlay


def get_random_unique_category_images(dataset, class_names, num_images=5):
    """
    Randomly selects `num_images` unique category images from the dataset.

    Args:
        dataset (torchvision Dataset): CIFAR-10 dataset.
        class_names (list): List of class names.
        num_images (int): Number of unique category images to retrieve.

    Returns:
        list: Selected images (numpy format).
        list: Image tensors (preprocessed for model input).
        list: Corresponding class labels.
    """
    images, tensors, labels = [], [], []

    mean, std = calculate_dataset_statistics()  # CIFAR-10 mean, std

    # Collect all indices per class
    class_indices = {i: [] for i in range(len(class_names))}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[int(label)].append(idx)  # Ensure label is int

    # Shuffle indices within each class
    for label in class_indices:
        random.shuffle(class_indices[label])

    # Randomly select one image per class, ensuring unique categories
    selected_classes = random.sample(list(class_indices.keys()), num_images)
    for label in selected_classes:
        idx = class_indices[label].pop()  # Pick a random index from that class
        img, _ = dataset[idx]
        img_np = denormalize(img, mean, std)  # Convert tensor to numpy format
        tensors.append(img.unsqueeze(0))  # Add batch dimension
        images.append(img_np)
        labels.append(label)

    return images, tensors, labels


def visualize_gradcam_multiple_layers(
    model,
    dataset,
    class_names,
    layer_names,
    output_path="outputs/gradcam_results.png",
    num_images=5,
):
    """
    Generates Grad-CAM visualizations for multiple layers on multiple images.
    Each row represents a different input image, and the columns represent Grad-CAM visualizations for different layers.

    Args:
        model (torch.nn.Module): Trained model.
        dataset (torchvision Dataset): CIFAR-10 dataset.
        class_names (list): Class names of CIFAR-10.
        layer_names (list): List of target layers for Grad-CAM.
        output_path (str): Path to save the visualization.
        num_images (int): Number of images to visualize Grad-CAM on.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Select `num_images` unique images from different classes
    images, tensors, labels = get_random_unique_category_images(
        dataset, class_names, num_images=num_images
    )

    num_layers = len(layer_names)

    fig, axes = plt.subplots(
        num_images,  # Remove extra row for layer titles
        num_layers + 1,  # +1 for the original image column
        figsize=(3 * (num_layers + 1), 3 * num_images),
        gridspec_kw={
            "hspace": 0.6,
            "wspace": 0.5,
        },  # Increase space between rows and columns
    )

    fig.suptitle(
        "Grad-CAM Visualizations for Multiple Images and Layers",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    # Add layer titles on top row
    for col_idx, layer_name in enumerate(layer_names):
        axes[0, col_idx + 1].set_title(f"Layer {layer_name}", fontsize=14)

    for row_idx in range(num_images):
        sample_img_np = images[row_idx]  # Original image (numpy format)
        sample_tensor = tensors[row_idx].to(device)  # Model input
        sample_label = labels[row_idx]  # True class label

        # **First Column: Input Image**
        axes[row_idx, 0].imshow(sample_img_np)
        axes[row_idx, 0].set_title(f"{class_names[sample_label]}", fontsize=14)
        axes[row_idx, 0].axis("off")

        # **Grad-CAM for each layer**
        for col_idx, layer_name in enumerate(layer_names):
            grad_cam = GradCAM(model, target_layer=layer_name)  # Initialize Grad-CAM
            cam = grad_cam.generate_cam(
                sample_tensor, target_class=sample_label
            )  # Get CAM

            # Apply Grad-CAM overlay with viridis colormap
            cam_resized = cv2.resize(
                cam, (sample_img_np.shape[1], sample_img_np.shape[0])
            )
            cam_resized = (cam_resized - cam_resized.min()) / (
                cam_resized.max() - cam_resized.min() + 1e-6
            )
            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam_resized), cv2.COLORMAP_VIRIDIS
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(
                np.uint8(sample_img_np * 255), 0.5, heatmap, 0.5, 0
            )

            # Show Grad-CAM visualization
            axes[row_idx, col_idx + 1].imshow(overlay)
            axes[row_idx, col_idx + 1].axis("off")

    # Add a heatmap color bar on the right
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])  # Adjust position of colorbar
    heatmap = np.linspace(0, 1, 256).reshape(256, 1)
    cbar_img = plt.imshow(heatmap, cmap="viridis", aspect="auto")
    plt.colorbar(cbar_img, cax=cbar_ax)
    cbar_ax.set_title("Activation", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved as {output_path}")
    plt.show()


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
            logger.info(
                "Debug mode enabled: Using a smaller dataset for visualization."
            )
            dataloaders = get_cifar_dataloaders(include_test=True, test_only=True)

            # Ensure test_loader exists before applying debug mode
            if "test" not in dataloaders or dataloaders["test"] is None:
                logger.error("Error: test_loader not found in dataloaders!")
                raise ValueError(
                    "test_loader is None. Cannot proceed with visualization."
                )

            # Apply Debug Mode to reduce dataset size
            _, _, test_loader = get_debug_dataloaders(
                test_loader=dataloaders.get("test")
            )

            # Ensure test_loader is still valid after debug mode
            if test_loader is None:
                logger.error(
                    "Error: test_loader is None after applying get_debug_dataloaders()!"
                )
                raise ValueError("test_loader is None. Check get_debug_dataloaders().")

        else:
            # Load the full dataset when not in debug mode
            logger.info("Debug mode OFF: Using full dataset.")
            dataloaders = get_cifar_dataloaders(include_test=True, test_only=True)

            if "test" not in dataloaders or dataloaders["test"] is None:
                logger.error("Error: test_loader not found in dataloaders!")
                raise ValueError(
                    "test_loader is None. Cannot proceed with visualization."
                )

            test_loader = dataloaders["test"]

        '''# Run visualization of CIFAR-10 images
        logger.info("Running visualize_cifar10_with_labels()...")
        visualize_cifar10_with_labels(
            class_names, num_examples=5, save_path="outputs/cifar10_visualization.png"
        )

        # Run visualization of image transformations
        logger.info("Running visualize_image_transformations()...")
        visualize_image_transformations()

        logger.info("Visualization pipeline completed successfully.")

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

        '''# Run activation visualization on a sample image
        # Select a single image from the test dataset
        image_index = 10  # Choose an index from the test set
        image, label = test_loader.dataset[image_index]  # Extract image

        # Run visualization
        logger.info("Running visualize_activations...")

        layer_names = [
            "conv2d_block1.0",  # First convolution layer in block 1
            "conv2d_block2.0",  # First convolution layer in block 2
            "conv2d_block3.0",  # First convolution layer in block 3
            "conv2d_block3.4",  # Third convolution layer in block 3 (if applicable)
            "conv2d_block4.4",  # Third convolution layer in block 4 (if applicable)
            "conv2d_block5.4",  # Third convolution layer in block 5 (if applicable)
        ]

        # Run Grad-CAM visualization on 5 random images
        visualize_gradcam_multiple_layers(
            model,
            test_loader.dataset,
            class_names,
            layer_names,
            output_path="outputs/gradcam_results.png",
        )
        logger.info("Visualization pipeline completed successfully.")

        '''plot_data_split()

        generate_vgg_visualization("vgg11")  # Generates VGG-11 PDF
        generate_vgg_visualization("vgg16")  # Generates VGG-16 PDF'''

    except Exception as e:
        logger.error(f"Error in visualization pipeline: {e}")
        raise


# Run when executing `visualize.py` directly
if __name__ == "__main__":
    run_visualization_pipeline()
