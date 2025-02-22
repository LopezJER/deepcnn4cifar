import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
from src.core.config import model_setup, hyperparams, debug

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformDataset(Dataset):
    """
    Custom dataset wrapper that applies transformations to an existing dataset.
    """
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.classes = getattr(dataset, "classes", None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def calculate_dataset_statistics():
    """
    Calculates mean and standard deviation of the CIFAR-10 training dataset.
    
    Returns:
        tuple: (train_mean, train_std) for normalization.
    """
    logger.info("Calculating dataset statistics...")
    train_data = datasets.CIFAR10(root="./data", train=True, download=True)
    
    x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
    train_mean = np.mean(x, axis=(0, 1)) / 255.0
    train_std = np.std(x, axis=(0, 1)) / 255.0
    
    logger.info(f"Dataset mean: {train_mean}, Dataset std: {train_std}")
    return train_mean, train_std


def get_cifar_dataloaders(include_test=False, test_only=False):
    """
    Loads CIFAR-10 dataset and applies necessary transformations.
    
    Args:
        include_test (bool): Whether to include test dataset.
        test_only (bool): Whether to load only the test dataset.
    
    Returns:
        dict: Dictionary containing DataLoaders for train, val, and test (if included).
    """
    logger.info("Loading CIFAR-10 data...")
    cifar_dataloader = {}
    batch_size = hyperparams["batch_size"]
    dataset_mean, dataset_std = calculate_dataset_statistics()

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, hue=0.1),
        transforms.Normalize(mean=dataset_mean.tolist(), std=dataset_std.tolist()),
    ])

    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=dataset_mean.tolist(), std=dataset_std.tolist()),
    ])

    if not test_only:
        train_dataset_raw = datasets.CIFAR10(root="./data", train=True, transform=None, download=True)
        val_split = model_setup["val_split"]
        train_size = int((1 - val_split) * len(train_dataset_raw))
        val_size = len(train_dataset_raw) - train_size
        train_subset_raw, val_subset_raw = random_split(train_dataset_raw, [train_size, val_size])

        train_data = TransformDataset(train_subset_raw, train_transform)
        val_data = TransformDataset(val_subset_raw, val_test_transform)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        cifar_dataloader["train"] = train_loader
        cifar_dataloader["val"] = val_loader

        if include_test:
            test_dataset = datasets.CIFAR10(root="./data", train=False, transform=val_test_transform, download=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            cifar_dataloader["test"] = test_loader
    else:
        test_dataset = datasets.CIFAR10(root="./data", train=False, transform=val_test_transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        cifar_dataloader["test"] = test_loader

    logger.info("Data loading and preprocessing complete.")
    return cifar_dataloader


class DebugDataset(Dataset):
    """
    Dataset wrapper for debugging purposes, allowing extraction of smaller subsets.
    """
    
    def __init__(self, dataset, subset):
        self.dataset = dataset
        self.subset = subset
        self.classes = dataset.classes

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]


def get_debug_dataloaders(train_loader=None, val_loader=None, test_loader=None):
    """
    Extracts a smaller subset of the dataset for debugging while maintaining transformations.
    
    Args:
        train_loader (DataLoader, optional): DataLoader for training data.
        val_loader (DataLoader, optional): DataLoader for validation data.
        test_loader (DataLoader, optional): DataLoader for test data.
    
    Returns:
        tuple: Debugging DataLoaders for train, val, and test.
    """
    def extract_subset(loader, num_images):
        dataset = loader.dataset
        subset_data = []
        for images, labels in loader:
            subset_data.extend(zip(images, labels))
            if len(subset_data) >= num_images:
                break
        return subset_data[:num_images], dataset

    logger.info(
        f"Debug mode: Using {debug['train_size'] + debug['val_size'] + debug['test_size']} images "
        f"and {debug['num_epochs']} epochs"
    )

    if train_loader is not None:
        train_subset, train_dataset = extract_subset(train_loader, debug["train_size"])
        train_loader = DataLoader(DebugDataset(train_dataset, train_subset), batch_size=debug["batch_size"], shuffle=True)

    if val_loader is not None:
        val_subset, val_dataset = extract_subset(val_loader, debug["val_size"])
        val_loader = DataLoader(DebugDataset(val_dataset, val_subset), batch_size=debug["batch_size"], shuffle=False)

    if test_loader is not None:
        test_subset, test_dataset = extract_subset(test_loader, debug["test_size"])
        test_loader = DataLoader(DebugDataset(test_dataset, test_subset), batch_size=debug["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader
