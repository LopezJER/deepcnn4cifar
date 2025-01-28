import os
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, auc, confusion_matrix
from core.model import VGG_Network
from utils.load_model import load_model
from torch.utils.data import Dataset, DataLoader, random_split


# Function to generate predictions and true labels
def get_predictions(model, dataloader, device):
    """
    Get predictions and true labels for a dataset.

    Parameters:
        model: Trained model.
        dataloader: DataLoader for the dataset.
        device: Device to run the predictions on.

    Returns:
        true_labels: Numpy array of true labels.
        predicted_labels: Numpy array of predicted labels.
    """
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
    """
    Visualize the top N mistakes from the confusion matrix.

    Parameters:
        cm: Confusion matrix (numpy array).
        class_names: List of class names.
        top_n: Number of top mistakes to visualize.
    """
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
    """
    Visualize multiple misclassified images for each misclassification category with row-wise titles.

    Parameters:
        true_labels: Numpy array of true labels.
        predicted_labels: Numpy array of predicted labels.
        test_dataset: Dataset object to retrieve images.
        top_mistakes: List of top misclassification pairs.
        class_names: List of class names.
        images_per_category: Number of images to display per misclassification pair.
    """
    rows = len(top_mistakes)
    cols = images_per_category + 1  # Add one column for the row titles

    plt.figure(figsize=(15, 3 * rows))

    for row_idx, (true_class, predicted_class, _) in enumerate(top_mistakes):
        true_class_idx = class_names.index(true_class)
        predicted_class_idx = class_names.index(predicted_class)
        mistake_indices = np.where(
            (true_labels == true_class_idx) & (predicted_labels == predicted_class_idx)
        )[0]

        # Add the row title
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


if __name__ == "__main__":
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataset
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.49139968, 0.48215841, 0.44653091],
                std=[0.24703223, 0.24348513, 0.26158784],
            ),
        ]
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, transform=transform, download=True
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load model
    model = VGG_Network(input_size=(3, 224, 224), num_classes=10, config="vgg16").to(
        device
    )
    model = load_model(
        model,
        "/kaggle/input/session-2-vgg16-cifar/session2_vgg16_best_model.pth",
        device,
    )

    # Get predictions and true labels
    true_labels, predicted_labels = get_predictions(model, test_loader, device)

    # Load evaluation data
    evaluation_data = torch.load(
        "/kaggle/input/vgg16-evaluation-data/evaluation_data.pth"
    )
    cm = evaluation_data["confusion_matrix"]
    class_names = evaluation_data["class_names"]

    # Visualize top mistakes
    top_mistakes = visualize_top_mistakes(cm, class_names, top_n=5)

    # Visualize grouped misclassified images
    visualize_mistakes_images_grouped_with_row_titles(
        true_labels, predicted_labels, test_dataset, top_mistakes, class_names
    )


def generate_vgg_architecture(architecture="vgg16"):
    """
    Generates a LaTeX-based visualization for VGG architectures (VGG16 or VGG11).

    Parameters:
        architecture (str): "vgg16" or "vgg11".
    """
    # Define LaTeX filename
    tex_filename = f"custom_{architecture}.tex"
    pdf_filename = f"custom_{architecture}.pdf"
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    # LaTeX source code
    tex_code = (
        r"""
    \documentclass[border=15pt, multi, tikz]{standalone}
    \usepackage{import}
    \subimport{PlotNeuralNet/layers/}{init}
    \usetikzlibrary{positioning}

    % Define Colors
    \def\ConvColor{rgb:yellow,5;red,2.5;white,5}
    \def\ConvReluColor{rgb:yellow,5;red,5;white,5}
    \def\PoolColor{rgb:red,1;black,0.3}
    \def\FcColor{rgb:blue,5;red,2.5;white,5}
    \def\SoftmaxColor{rgb:magenta,5;black,7}

    \begin{document}
    \begin{tikzpicture}
    """
        + (
            """
    % VGG16 Architecture
    \pic[shift={(0,0,0)}] at (0,0,0) {Box={name=input,caption=Input,xlabel={{"3",""}},ylabel=224,zlabel=224,fill=\ConvColor,height=40,width=1,depth=40}};
    \pic[shift={(1.5,0,0)}] at (input-east) {RightBandedBox={name=conv1,caption=Conv1,xlabel={{"64","64"}},ylabel=224,zlabel=224,fill=\ConvColor,bandfill=\ConvReluColor,height=40,width={2,2},depth=40}};
    \pic[shift={(4,0,0)}] at (conv1-east) {Box={name=fc,caption=FC,xlabel={{"4096",""}},fill=\FcColor,height=3,width=3,depth=80}};
    """
            if architecture == "vgg16"
            else """
    % VGG11 Architecture
    \pic[shift={(0,0,0)}] at (0,0,0) {Box={name=input,caption=Input,xlabel={{"3",""}},ylabel=224,zlabel=224,fill=\ConvColor,height=40,width=1,depth=40}};
    \pic[shift={(1.5,0,0)}] at (input-east) {Box={name=conv1,caption=Conv1,xlabel={{"64",""}},ylabel=224,zlabel=224,fill=\ConvColor,height=40,width=2,depth=40}};
    \pic[shift={(4,0,0)}] at (conv1-east) {Box={name=fc,caption=FC,xlabel={{"4096",""}},fill=\FcColor,height=3,width=3,depth=80}};
    """
        )
        + """
    \end{tikzpicture}
    \end{document}
    """
    )

    # Write the .tex file
    with open(tex_filename, "w") as f:
        f.write(tex_code)

    # Compile the .tex file to .pdf
    try:
        subprocess.run(["pdflatex", tex_filename], check=True)
        os.rename(pdf_filename, os.path.join(output_dir, pdf_filename))
        print(f"{architecture.upper()} architecture PDF saved to {output_dir}")
    except subprocess.CalledProcessError:
        print(f"Error compiling {tex_filename}")


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


# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(self._save_activations)
                module.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_class):
        self.model.zero_grad()
        output = self.model(input_image)
        output[:, target_class].backward()

        weights = torch.mean(self.gradients, dim=(2, 3))
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i, :, :]
        cam = torch.clamp(cam, min=0).cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min())


if __name__ == "__main__":
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataset
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.49139968, 0.48215841, 0.44653091],
                std=[0.24703223, 0.24348513, 0.26158784],
            ),
        ]
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, transform=transform, download=True
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load model
    model = VGG_Network(input_size=(3, 224, 224), num_classes=10, config="vgg16").to(
        device
    )
    model = load_model(
        model,
        "/kaggle/input/session-2-vgg16-cifar/session2_vgg16_best_model.pth",
        device,
    )

    # Get predictions and true labels
    true_labels, predicted_labels = get_predictions(model, test_loader, device)

    # Load evaluation data
    evaluation_data = torch.load(
        "/kaggle/input/vgg16-evaluation-data/evaluation_data.pth"
    )
    cm = evaluation_data["confusion_matrix"]
    class_names = evaluation_data["class_names"]

    # Visualize top mistakes
    top_mistakes = visualize_top_mistakes(cm, class_names, top_n=5)

    # Visualize grouped misclassified images
    visualize_mistakes_images_grouped_with_row_titles(
        true_labels, predicted_labels, test_dataset, top_mistakes, class_names
    )
# Example usage
generate_vgg_architecture("vgg16")
generate_vgg_architecture("vgg11")
