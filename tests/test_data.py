import pytest
from src.utils.load_data import get_cifar_dataloaders
from src.core.config import model_setup, hyperparams
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

class_labels = np.array(range(0, model_setup['num_classes']))
samp_data = get_cifar_dataloaders(test_only=True)
samp_data = samp_data['test'].dataset

def has_data(data):
    return len(data) > 0

def has_valid_labels(labels):
    correct_type = labels.dtype == torch.long
    correct_values = np.all(np.isin(labels, class_labels))
    return correct_type and correct_values

def has_correct_shape(images):
    return tuple(images.shape[1:]) == (3, 224, 224)

def has_correct_batch_size(images, labels):
    return labels.shape[0] == images.shape[0] == hyperparams['batch_size']

def has_correct_preprocessing(images):
    correct_type = images.dtype == torch.float32
    no_nan_values = not torch.isnan(images).any()
    no_inf_values = not torch.isinf(images).any()

    return correct_type and no_nan_values and no_inf_values


# Test to ensure the dataset has data
def test_has_data():
    assert has_data(samp_data)


# Test to ensure labels are valid (correct type and values)
def test_has_valid_labels():
    random_sampler = RandomSampler(samp_data)
    random_loader = DataLoader(samp_data, sampler=random_sampler, batch_size=hyperparams['batch_size'])

    for images, labels in random_loader:
        assert has_valid_labels(labels)
        break  # Only check on one random batch


# Test to ensure images have the correct shape
def test_has_correct_shape():
    random_sampler = RandomSampler(samp_data)
    random_loader = DataLoader(samp_data, sampler=random_sampler, batch_size=hyperparams['batch_size'])

    for images, labels in random_loader:
        assert has_correct_shape(images)
        break  # Only check on one random batch


# Test to ensure the batch size is correct
def test_has_correct_batch_size():
    random_sampler = RandomSampler(samp_data)
    random_loader = DataLoader(samp_data, sampler=random_sampler, batch_size=hyperparams['batch_size'])

    for images, labels in random_loader:
        assert has_correct_batch_size(images, labels)
        break  # Only check on one random batch


# Test to ensure images have the correct preprocessing (no NaN, no Inf, and correct type)
def test_has_correct_preprocessing():
    random_sampler = RandomSampler(samp_data)
    random_loader = DataLoader(samp_data, sampler=random_sampler, batch_size=hyperparams['batch_size'])

    for images, labels in random_loader:
        assert has_correct_preprocessing(images)
        break  # Only check on one random batch
