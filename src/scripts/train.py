import torch
from torch import nn
import time
import os
import matplotlib.pyplot as plt
import logging
from src.core.config import model_setup, paths, debug
from src.utils.load_model import get_hyperparams  # Import the function
from src.utils.load_model import load_model, setup_device
from src.utils.load_data import (
    get_cifar_dataloaders,
    get_debug_dataloaders,
)

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Function to train the model
def train_model(model, train_loader, val_loader, hyps, device):
    """
    Train the model with validation.

    Args:
        model (torch.nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        hyps (dict): Hyperparameters including optimizer, scheduler, etc.
        device (torch.device): Device to use for training (CPU or GPU).

    Returns:
        tuple: Train losses, validation losses, and epoch times.
    """
    try:
        num_epochs = debug["num_epochs"] if debug["on"] else hyps["num_epochs"]
        optimizer = hyps["optimizer"]
        scheduler = hyps["scheduler"]
        patience = hyps["early_stopping_patience"]
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        train_losses = []
        val_losses = []
        epoch_times = []

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
            model.train()

            train_running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Debugging NaN Loss
                assert not torch.isnan(loss), "Loss is NaN. Check your data and model."

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )  # Clip Gradients
                optimizer.step()

                train_running_loss += loss.item()
                logger.info(f"Batch [{i+1}/{len(train_loader)}] - Loss: {loss.item():.4f}")

            # Validation Phase
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()

            # Epoch Metrics
            average_train_loss = train_running_loss / len(train_loader)
            average_val_loss = val_running_loss / len(val_loader)
            scheduler.step(average_val_loss)

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_time)

            train_losses.append(average_train_loss)
            val_losses.append(average_val_loss)

            logger.info(
                f"Train Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}, Time: {epoch_time:.2f}s"
            )

            # Save Best Model
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                epochs_without_improvement = 0
                model_save_path = os.path.join(
                    paths["outputs_dir"], f"{model_setup['arch']}_checkpoint.pth"
                )
                os.makedirs(paths["outputs_dir"], exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                    },
                    model_save_path,
                )
                logger.info(f"Saved best model to '{model_save_path}'")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        return train_losses, val_losses, epoch_times

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise


def visualize_losses(train_losses, val_losses):
    """
    Plot and save training and validation loss.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
    """
    try:
        output_dir = paths["outputs_dir"]
        os.makedirs(output_dir, exist_ok=True)
        loss_plot_path = os.path.join(output_dir, "train_val_loss.png")

        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Train and Validation Loss", fontweight="bold")
        plt.grid()
        plt.tight_layout()
        plt.savefig(loss_plot_path)

        logger.info(f"Loss plot saved to: {loss_plot_path}")
    except Exception as e:
        logger.error(f"An error occurred while visualizing losses: {e}")
        raise


def run_train_pipeline():
    """
    Main function to set up and train the model.
    """
    try:
        logger.info("Starting training of VGG16...")
        device = setup_device()
        vgg_model = load_model()
        hyps = get_hyperparams(vgg_model)

        # Handle Debug Mode
        if debug["on"]:
            logger.info("Debug mode enabled: Using a smaller dataset for training.")
            dataloaders = get_cifar_dataloaders()
            train_loader, val_loader, _ = get_debug_dataloaders(
                dataloaders["train"], dataloaders["val"], None
            )
        else:
            dataloaders = get_cifar_dataloaders()
            train_loader, val_loader = dataloaders["train"], dataloaders["val"]

        # Ensure DataLoaders Are Not Empty
        assert train_loader, "Training DataLoader is missing."
        assert val_loader, "Validation DataLoader is missing."
        assert len(train_loader) > 0, "Training DataLoader is empty."
        assert len(val_loader) > 0, "Validation DataLoader is empty."

        # Train the Model
        train_losses, val_losses, _ = train_model(
            vgg_model, train_loader, val_loader, hyps, device
        )

    except AssertionError as e:
        logger.error(f"Assertion error: {e}")
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        raise

    return vgg_model, train_losses, val_losses  # we need this variable if calling from main


if __name__ == "__main__":
    run_train_pipeline()
