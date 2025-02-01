import torch
import os
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.core.config import model_setup, paths, debug
from src.utils.load_model import load_model
from src.utils.load_data import get_cifar_dataloaders
from torch.utils.data import DataLoader
from collections import defaultdict
from src.scripts.visualize import visualize_mistakes_images_grouped_with_row_titles


class DebugDataset(torch.utils.data.Dataset):
    """Custom dataset to retain transforms and attributes when using a subset."""

    def __init__(self, original_dataset, subset_data):
        self.original_dataset = original_dataset  # Keep reference to original dataset
        self.data = subset_data  # Store the reduced subset
        self.transform = getattr(original_dataset, "transform", None)  # Keep transform
        self.classes = getattr(original_dataset, "classes", None)  # Retain class names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        # Convert Tensor to PIL Image if needed
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)  # Apply original dataset's transform

        return image, label


def get_debug_dataloader(test_loader):
    """Extracts a smaller subset of the dataset for debugging and returns new DataLoaders."""
    def extract_subset(loader, num_images):
        data, labels = [], []
        for images, lbls in loader:
            data.extend(images)
            labels.extend(lbls)
            if len(data) >= num_images:
                return list(zip(data[:num_images], labels[:num_images]))
        return list(zip(data, labels))

    print(f"Debug mode: Evaluating with {debug['test_size']} images")
    
    test_subset = extract_subset(test_loader, debug['test_size'])
    test_loader = DataLoader(test_subset, batch_size=debug['batch_size'], shuffle=False)
    
    return test_loader




# Evaluate the model
def evaluate_model(model, dataloader, device, num_classes):
    try:
        assert model is not None, "Model is None."
        assert dataloader is not None, "Dataloader is None."
        assert num_classes > 1, "Number of classes must be greater than 1."

        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.exp(outputs)  # Convert to probabilities
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        assert len(all_labels) > 0, "No labels collected from the dataset."

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted"
        )
        cm = confusion_matrix(all_labels, all_preds)
        accuracy = 100 * (all_labels == all_preds).sum() / len(all_labels)
        avg_loss = total_loss / len(dataloader)
        ne = label_binarize(all_labels, classes=np.arange(num_classes))

        return (
            accuracy,
            avg_loss,
            precision,
            recall,
            f1,
            cm,
            ne,
            np.array(all_probs),
        )
    except AssertionError as e:
        print(f"Assertion error during evaluation: {e}")
    except Exception as e:
        print(f"Unexpected error during model evaluation: {e}")
        raise


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


def calculate_confusion_matrix(true_labels, predicted_labels):
    """
    Computes the confusion matrix.

    Args:
        true_labels (np.ndarray): True labels.
        predicted_labels (np.ndarray): Predicted labels.

    Returns:
        np.ndarray: Confusion matrix.
    """
    try:
        cm = confusion_matrix(true_labels, predicted_labels)
        return cm
    except Exception as e:
        print(f"Error calculating confusion matrix: {e}")
        return None


def calculate_roc_curve(all_labels_oh, all_probs, class_names):
    """
    Computes ROC curve data for each class.

    Args:
        all_labels_oh (np.ndarray): One-hot encoded true labels.
        all_probs (np.ndarray): Model predicted probabilities.
        class_names (list): List of class names.

    Returns:
        dict: False positive rate (fpr), true positive rate (tpr), and AUC values.
    """
    try:
        assert (
            all_labels_oh.shape == all_probs.shape
        ), "Mismatch between labels and probabilities."

        fpr, tpr, roc_auc = {}, {}, {}
        for i, class_name in enumerate(class_names):
            fpr[i], tpr[i], _ = roc_curve(all_labels_oh[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        return {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}
    except Exception as e:
        print(f"Error calculating ROC curve: {e}")
        return None


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plots and saves the confusion matrix as a heatmap with proper positioning.

    Args:
        cm (np.ndarray): Confusion matrix.
        class_names (list): Class names.
        save_path (str): Path to save the plot.
    """
    try:
        os.makedirs(
            os.path.dirname(save_path), exist_ok=True
        )  # Ensure output directory exists

        plt.figure(figsize=(12, 10))  # Increase figure size for better positioning
        heatmap_obj = sns.heatmap(  # Avoid using 'heatmap' to prevent name conflicts
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted", fontweight="bold", fontsize=14)
        plt.ylabel("True", fontweight="bold", fontsize=14)
        plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)

        # Add "Sample Amount" text beside the color bar
        colorbar = heatmap_obj.collections[0].colorbar
        colorbar.ax.set_ylabel("Sample Amount", fontsize=12, fontweight="bold")

        # Adjust spacing to center the plot
        plt.subplots_adjust(left=0.05, right=1, top=0.95, bottom=0.05)
        plt.tight_layout
        plt.savefig(save_path)  # Save the plot as PNG
        print(f"Confusion matrix saved to {save_path}")
        plt.close()  # Close plot to free memory
    except Exception as e:
        print(f"Error in plotting confusion matrix: {e}")


def plot_roc_curves(roc_data, class_names, save_path):
    """
    Plots and saves ROC curves for multi-class classification.

    Args:
        roc_data (dict): Dictionary containing fpr, tpr, and roc_auc for each class.
        class_names (list): List of class names.
        save_path (str): Path to save the plot.
    """
    try:
        os.makedirs(
            os.path.dirname(save_path), exist_ok=True
        )  # Ensure output directory exists

        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(class_names):
            plt.plot(
                roc_data["fpr"][i],
                roc_data["tpr"][i],
                label=f"{class_name} (AUC = {roc_data['roc_auc'][i]:.2f})",
            )
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess (AUC = 0.50)")
        plt.xlabel("False Positive Rate", fontweight="bold")
        plt.ylabel("True Positive Rate", fontweight="bold")
        plt.title("ROC Curves", fontsize=16, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)  # Save the plot as PNG
        print(f"ROC curves saved to {save_path}")
        plt.close()  # Close plot to free memory
    except Exception as e:
        print(f"Error in plotting ROC curves: {e}")


if __name__ == "__main__":
    try:
        # Load configuration and model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = load_model()
        assert model is not None, "Failed to load model."

        # **Apply Debug Mode if Enabled**
        if debug["on"]:
            print("Debug mode enabled: Using a smaller test dataset.")
            test_loader = get_debug_dataloader(
                test_loader=test_loader["test"]
            )

        # Evaluate model
        evaluation_results = evaluate_model(
            model, test_loader, device, num_classes=model_setup["num_classes"]
        )
        if evaluation_results:
            accuracy, loss, precision, recall, f1, cm, labels_oh, probs = (
                evaluation_results
            )

            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Loss: {loss:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            plot_confusion_matrix(cm, class_names)
            plot_roc_curves(labels_oh, probs, class_names)

        # First, we need to get true and predicted labels from model predictions
        true_labels = np.argmax(labels_oh, axis=1)  # Convert one-hot back to class indices
        predicted_labels = np.argmax(probs, axis=1)  # Get predicted classes from probabilities

        # Find the top mistake categories
        mistake_counts = defaultdict(int)
        for true_idx, pred_idx in zip(true_labels, predicted_labels):
            if true_idx != pred_idx:
                mistake_pair = (class_names[true_idx], class_names[pred_idx])
                mistake_counts[mistake_pair] += 1

        # Sort mistakes by frequency
        top_mistakes = [(true, pred, count) 
                        for (true, pred), count in sorted(mistake_counts.items(), 
                                                        key=lambda x: x[1], 
                                                        reverse=True)][:5]  # Show top 5 mistakes

        # Call the visualization function
        visualize_mistakes_images_grouped_with_row_titles(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
            test_dataset=test_loader.dataset,  # Get the dataset from the DataLoader
            top_mistakes=top_mistakes,
            class_names=class_names,
            images_per_category=5  # Show 5 examples per mistake category
        )

    except AssertionError as e:
        print(f"Assertion error in main execution: {e}")
    except Exception as e:
        print(f"Unexpected error in main execution: {e}")
        raise
