from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
from src.core.config import model_setup, hyperparams

class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def calculate_dataset_statistics():
    train_data = datasets.CIFAR10(root='./data', train=True, download=True)

    # Calculate mean and std
    # as per Github discussions (paulkorir, 2018): https://github.com/facebookarchive/fb.resnet.torch/issues/180#issuecomment-433419706
    x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
    train_mean = np.mean(x, axis=(0, 1)) / 255.0
    train_std = np.std(x, axis=(0, 1)) / 255.0

    return train_mean, train_std
def get_cifar_dataloaders(include_test = False, test_only = False):
    print("Loading data...")
    cifar_dataloader = {}

    batch_size = hyperparams['batch_size']

    dataset_mean, dataset_std = calculate_dataset_statistics()

    print("Processing and augmenting data...")
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, hue=0.1),
        transforms.Normalize(mean=dataset_mean.tolist(), std=dataset_std.tolist())
    ])

    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=dataset_mean.tolist(), std=dataset_std.tolist())
    ])


    if not test_only:

        # Load raw dataset without transforms first
        train_dataset_raw = datasets.CIFAR10(root='./data', train=True, transform=None, download=True)

        # Split raw dataset
        val_split = model_setup['val_split']
        train_size = int((1 - val_split) * len(train_dataset_raw))
        val_size = len(train_dataset_raw) - train_size
        train_subset_raw, val_subset_raw = random_split(train_dataset_raw, [train_size, val_size])

        # Create separate datasets with appropriate transforms
        train_data = TransformDataset(train_subset_raw, train_transform)
        val_data = TransformDataset(val_subset_raw, val_test_transform)

        # Create DataLoaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        cifar_dataloader['train'] = train_loader
        cifar_dataloader['val'] = val_loader

        if include_test:
            test_dataset = datasets.CIFAR10(root='./data', train=False, transform=val_test_transform, download=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            cifar_dataloader['test']  = test_loader

        
    else:
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=val_test_transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        cifar_dataloader['test'] = test_loader
    print("Successfully preprocessed and split data.")
    return cifar_dataloader






