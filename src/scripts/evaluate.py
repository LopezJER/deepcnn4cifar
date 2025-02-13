import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import logging
from src.core.model import *
from torchvision import transforms
from src.core.config import model_setup, debug
from src.utils.load_model import load_model
from src.utils.load_data import get_cifar_dataloaders
from torch.utils.data import DataLoader
from collections import defaultdict


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DebugDataset(torch.utils.data.Dataset):
    """Custom dataset to retain transforms and attributes when using a subset."""

    def __init__(self, original_dataset, subset_data):
        """
        Initializes the DebugDataset.

        Args:
            original_dataset (torch.utils.data.Dataset): The original dataset to extract the transform and classes from.
            subset_data (list): The subset of data (images, labels) to use for debugging.
        """
        self.original_dataset = original_dataset  # Keep reference to original dataset
        self.data = subset_data  # Store the reduced subset
        self.transform = getattr(original_dataset, "transform", None)  # Keep transform
        self.classes = getattr(original_dataset, "classes", None)  # Retain class names

    def __len__(self):
        """Returns the size of the subset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the image and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is the transformed image and label is the ground truth label.
        """
        image, label = self.data[idx]

        # Convert Tensor to PIL Image if needed
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)  # Apply original dataset's transform

        return image, label


def get_debug_dataloader(test_loader):
    """
    Extracts a smaller subset of the dataset for debugging and returns new DataLoaders.

    Args:
        test_loader (DataLoader): The original DataLoader containing the test data.

    Returns:
        DataLoader: A new DataLoader for the subset of test data.
    """

    def extract_subset(loader, num_images):
        """
        Extracts a subset of images and labels from the loader.

        Args:
            loader (DataLoader): The DataLoader to extract from.
            num_images (int): The number of images to extract.

        Returns:
            list: A list of (image, label) pairs.
        """
        data, labels = [], []
        for images, lbls in loader:
            data.extend(images)
            labels.extend(lbls)
            if len(data) >= num_images:
                return list(zip(data[:num_images], labels[:num_images]))
        return list(zip(data, labels))

    logger.info(f"Debug mode: Evaluating with {debug['test_size']} images")

    test_subset = extract_subset(test_loader, debug["test_size"])
    test_loader = DataLoader(test_subset, batch_size=debug["batch_size"], shuffle=False)

    return test_loader


def evaluate_model(model, dataloader, device, num_classes):
    """
    Evaluates the model on the given dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): The DataLoader containing the test data.
        device (torch.device): The device to run the model on.
        num_classes (int): The number of classes in the dataset.

    Returns:
        tuple: Accuracy, average loss, precision, recall, F1 score, confusion matrix, labels one-hot, and probabilities.
    """
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
        logger.error(f"Assertion error during evaluation: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during model evaluation: {e}")
        raise


def plot_confusion_matrix(cm, class_names):
    """
    Plots the confusion matrix.

    Args:
        cm (ndarray): The confusion matrix.
        class_names (list): List of class names.

    Raises:
        AssertionError: If the confusion matrix is not square or class names don't match.
    """
    try:
        assert cm.shape[0] == cm.shape[1], "Confusion matrix must be square."
        assert (
            len(class_names) == cm.shape[0]
        ), "Class names must match confusion matrix dimensions."

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()
    except AssertionError as e:
        logger.error(f"Assertion error in confusion matrix plot: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in confusion matrix plot: {e}")
        raise


def plot_roc_curves(all_labels_oh, all_probs, class_names):
    """
    Plots the ROC curves for each class.

    Args:
        all_labels_oh (ndarray): One-hot encoded true labels.
        all_probs (ndarray): Predicted probabilities.
        class_names (list): List of class names.

    Raises:
        AssertionError: If the shape of labels and probabilities do not match.
    """
    try:
        assert (
            all_labels_oh.shape == all_probs.shape
        ), "Shape mismatch between labels and probabilities."

        fpr, tpr, roc_auc = {}, {}, {}
        for i, class_name in enumerate(class_names):
            fpr[i], tpr[i], _ = roc_curve(all_labels_oh[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(class_names):
            plt.plot(fpr[i], tpr[i], label=f"{class_name} (AUC = {roc_auc[i]:.2f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random Guess (AUC = 0.50)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.show()
    except AssertionError as e:
        logger.error(f"Assertion error in ROC curve plot: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in ROC curve plot: {e}")
        raise


def run_evaluate_pipeline(model=None):
    try:
        # Load configuration and model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        if model is None:
            model = load_model()

        # Load test data
        test_loader = get_cifar_dataloaders(include_test=True, test_only=True)
        assert test_loader is not None, "Test DataLoader is None."

        # **Apply Debug Mode if Enabled**
        if debug["on"]:
            logger.info("Debug mode enabled: Using a smaller test dataset.")
            test_loader = get_debug_dataloader(test_loader=test_loader["test"])

        # Evaluate model
        evaluation_results = evaluate_model(
            model, test_loader, device, num_classes=model_setup["num_classes"]
        )
        return evaluation_results
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}")
        raise


if __name__ == "__main__":
    run_evaluate_pipeline()
