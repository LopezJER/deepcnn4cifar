import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from src.core.model import VGG_Network
from src.core.config import model_setup, hyperparams, paths, debug
from src.utils.load_model import load_model, setup_device
from src.utils.load_data import get_cifar_dataloaders

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

        if debug['on']:
            num_epochs = debug['num_epochs']
            train_loader, val_loader = get_debug_dataloaders(train_loader, val_loader)
                    
        else:
            num_epochs = hyps['num_epochs']
            
        optimizer = hyps['optimizer']
        scheduler = hyps['scheduler']
        patience = hyps['early_stopping_patience']
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        train_losses = []
        val_losses = []
        epoch_times = []

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print(f"Epoch [{epoch+1}/{num_epochs}]")
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip Gradients
                optimizer.step()

                train_running_loss += loss.item()
                print(f"Batch [{i+1}/{len(train_loader)}] - Loss: {loss.item():.4f}")

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

            print(f"Train Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}, Time: {epoch_time:.2f}s")

            # Save Best Model
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                epochs_without_improvement = 0
                model_save_path = os.path.join(paths['outputs_dir'], f"{model_setup['arch']}_checkpoint.pth")
                if not os.path.isdir(paths['outputs_dir']):
                    os.mkdir(paths['outputs_dir'])
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss
                }, model_save_path)
                print(f"Saved best model to '{model_save_path}'")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return train_losses, val_losses, epoch_times

    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise

def get_debug_dataloaders(train_loader, val_loader):
    """Extracts a smaller subset of the dataset for debugging and returns new DataLoaders."""
    def extract_subset(loader, num_images):
        data, labels = [], []
        for images, lbls in loader:
            data.extend(images)
            labels.extend(lbls)
            if len(data) >= num_images:
                return list(zip(data[:num_images], labels[:num_images]))
        return list(zip(data, labels))

    print(f"Debug mode: Using {debug['train_size'] + debug['val_size']} images and {debug['num_epochs']} epochs")
    
    train_subset = extract_subset(train_loader, debug['train_size'])
    val_subset = extract_subset(val_loader, debug['val_size'])

    train_loader = DataLoader(train_subset, batch_size=debug['batch_size'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=debug['batch_size'], shuffle=False)
    
    return train_loader, val_loader


def get_hyperparams(model):
    """
    Configure hyperparameters including optimizer and scheduler.

    Args:
        model (torch.nn.Module): The neural network model.

    Returns:
        dict: Dictionary containing hyperparameters.
    """
    try:
        optimizer = optim.SGD(model.parameters(), momentum=hyperparams['momentum'], lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)

        hyps = {key: value for key, value in hyperparams.items()}
        hyps['optimizer'], hyps['scheduler'] = optimizer, scheduler
        return hyps
    except KeyError as e:
        print(f"Missing hyperparameter: {e}")
        raise

def visualize_losses(train_losses, val_losses):
    """
    Plot and save training and validation loss.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
    """
    try:
        output_dir = paths['outputs_dir']
        os.makedirs(output_dir, exist_ok=True)
        loss_plot_path = os.path.join(output_dir, "train_val_loss.png")

        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Train and Validation Loss')
        plt.grid()
        plt.savefig(loss_plot_path)

        print(f"Loss plot saved to: {loss_plot_path}")
    except Exception as e:
        print(f"An error occurred while visualizing losses: {e}")
        raise

def main():
    """
    Main function to set up and train the model.
    """
    try:
        print("Starting training of VGG16...")
        vgg_model = load_model()
        dataloaders = get_cifar_dataloaders()
        hyps = get_hyperparams(vgg_model)
        device = setup_device()
        train_loader, val_loader = dataloaders['train'], dataloaders['val']

        # Ensure DataLoaders are not empty
        assert len(train_loader) > 0, "Training DataLoader is empty."
        assert len(val_loader) > 0, "Validation DataLoader is empty."

        train_losses, val_losses, _ = train_model(vgg_model, train_loader, val_loader, hyps, device)
        visualize_losses(train_losses, val_losses)
    except AssertionError as e:
        print(f"Assertion error: {e}")
    except Exception as e:
        print(f"An error occurred in the main function: {e}")
        raise

if __name__ == '__main__':
    main()
