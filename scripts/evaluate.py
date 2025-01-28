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
from core.model import VGG_Network
from utils.load_model import load_model
from utils.load_data import get_dataloader


# Evaluate the model
def evaluate_model(model, dataloader, device, num_classes):
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

            probs = torch.exp(outputs)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = 100 * (all_labels == all_preds).sum() / len(all_labels)
    avg_loss = total_loss / len(dataloader)
    all_labels_oh = label_binarize(all_labels, classes=np.arange(num_classes))

    return (
        accuracy,
        avg_loss,
        precision,
        recall,
        f1,
        cm,
        all_labels_oh,
        np.array(all_probs),
    )


# Plot confusion matrix
def plot_confusion_matrix(cm, class_names):
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


# Plot ROC curves
def plot_roc_curves(all_labels_oh, all_probs, class_names):
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


if __name__ == "__main__":
    # Load configuration and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG_Network(input_size=(3, 224, 224), num_classes=10, config="vgg16").to(
        device
    )
    model = load_model(model, "outputs/vgg16.pth")

    # Load test data
    test_loader = get_dataloader("data/test", batch_size=64, train=False)

    # Evaluate
    accuracy, loss, precision, recall, f1, cm, labels_oh, probs = evaluate_model(
        model, test_loader, device, num_classes=10
    )
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Loss: {loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Display results
    class_names = [
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ]
    plot_confusion_matrix(cm, class_names)
    plot_roc_curves(labels_oh, probs, class_names)
