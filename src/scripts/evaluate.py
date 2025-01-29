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
from src.core.config import class_names
from src.utils.load_model import load_model
from src.utils.load_data import get_dataloader


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

        model = VGG_Network(
            input_size=(3, 224, 224), num_classes=10, config="vgg16"
        ).to(device)

        model_path = "outputs/vgg16.pth"
        assert os.path.exists(model_path), f"Model file not found: {model_path}"

        model = load_model(model, model_path)

        # Load test data
        test_loader = get_dataloader("data/test", batch_size=64, train=False)
        assert test_loader is not None, "Test DataLoader is None."

        # Evaluate model
        evaluation_results = evaluate_model(model, test_loader, device, num_classes=10)
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
