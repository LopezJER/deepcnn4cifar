import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from src.core.model import VGG_Network
from torchvision import datasets, transforms
from src.core.config import model_setup, hyperparams, paths, debug, class_names
from src.utils.load_model import load_model
from src.utils.load_data import get_cifar_dataloaders
from torch.utils.data import DataLoader


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


def get_debug_dataloaders(train_loader=None, val_loader=None, test_loader=None):
    """Extracts a smaller subset of the dataset for debugging and retains transformations."""

    def extract_subset(loader, num_images):
        dataset = loader.dataset
        subset_data = []
        for images, labels in loader:
            subset_data.extend(zip(images, labels))
            if len(subset_data) >= num_images:
                break
        return subset_data[:num_images], dataset  # Return dataset for transform

    print(
        f"Debug mode: Using {debug['train_size'] + debug['val_size'] + debug['test_size']} images and {debug['num_epochs']} epochs"
    )

    if train_loader is not None:
        train_subset, train_dataset = extract_subset(train_loader, debug["train_size"])
        train_loader = DataLoader(
            DebugDataset(train_dataset, train_subset),
            batch_size=debug["batch_size"],
            shuffle=True,
        )

    if val_loader is not None:
        val_subset, val_dataset = extract_subset(val_loader, debug["val_size"])
        val_loader = DataLoader(
            DebugDataset(val_dataset, val_subset),
            batch_size=debug["batch_size"],
            shuffle=False,
        )

    if test_loader is not None:
        test_subset, test_dataset = extract_subset(test_loader, debug["test_size"])
        test_loader = DataLoader(
            DebugDataset(test_dataset, test_subset),
            batch_size=debug["batch_size"],
            shuffle=False,
        )
        return (test_loader,)  # Ensure single value return

    return None


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


# Plot confusion matrix
def plot_confusion_matrix(cm, class_names):
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
        print(f"Assertion error in confusion matrix plot: {e}")
    except Exception as e:
        print(f"Unexpected error in confusion matrix plot: {e}")
        raise


# Plot ROC curves
def plot_roc_curves(all_labels_oh, all_probs, class_names):
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
        print(f"Assertion error in ROC curve plot: {e}")
    except Exception as e:
        print(f"Unexpected error in ROC curve plot: {e}")
        raise


if __name__ == "__main__":
    try:
        # Load configuration and model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model_path = os.path.join(
            paths["outputs_dir"], f"{model_setup['arch']}_checkpoint.pth"
        )
        assert os.path.exists(model_path), f"Model file not found: {model_path}"

        model = load_model()

        # Load test data
        test_loader = get_cifar_dataloaders(include_test=True, test_only=True)
        assert test_loader is not None, "Test DataLoader is None."

        # **Apply Debug Mode if Enabled**
        if debug["on"]:
            print("Debug mode enabled: Using a smaller test dataset.")
            (test_loader["test"],) = get_debug_dataloaders(
                test_loader=test_loader["test"]
            )

        # Evaluate model
        evaluation_results = evaluate_model(
            model, test_loader["test"], device, num_classes=model_setup["num_classes"]
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

    except AssertionError as e:
        print(f"Assertion error in main execution: {e}")
    except Exception as e:
        print(f"Unexpected error in main execution: {e}")
        raise
