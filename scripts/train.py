import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from core.model import VGG_Network
from core.config import model_setup, hyperparams, paths
from utils.load_model import load_model, setup_device
from utils.load_data import get_cifar_dataloaders

# Function to train the model
def train_model(model, train_loader, val_loader, hyps, device):

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
            if torch.isnan(loss):
                print(f"NaN detected! Inputs: {inputs}, Outputs: {outputs}")
                return train_losses, val_losses, epoch_times

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

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            epochs_without_improvement = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, f'{paths['outputs_dir']}/{model_setup['arch']}_checkpoint.pth')
            print(f"Saved best model to '{paths['outputs_dir']}/{model_setup['arch']}_checkpoint.pth'")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return train_losses, val_losses, epoch_times

def main():
    # Train the model and capture losses
    print("Starting training of VGG16...")



def get_hyperparams(model):
    ### As per Simonyan and Zisserman's (2014)  paper:
    optimizer = optim.SGD(model.parameters(), momentum=hyperparams['momentum'], lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)
    
    hyps = {key: value for key, value in hyperparams.items()}
    hyps['optimizer'], hyps['scheduler'] = optimizer, scheduler
    return hyps

def visualize_losses(train_losses, val_losses):
    # Visualize train and validation loss
    output_dir = paths['outputs']
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
    plt.show()

    print(f"Loss plot saved to: {loss_plot_path}")


def main():
    vgg_model = load_model()
    dataloaders = get_cifar_dataloaders()
    hyps = get_hyperparams(vgg_model)
    device = setup_device()
    train_loader, val_loader = dataloaders['train'], dataloaders['val']
    train_losses, val_losses, _ = train_model(vgg_model, train_loader, val_loader, hyps, device)
    visualize_losses(train_losses, val_losses)

if __name__ == '__main__':
    main()

