import pytest
from utils.load_data import get_cifar_dataloaders, TransformDataset
from core.config import model_setup
import numpy as np

def test_get_cifar_dataloaders():
    samp_data = get_cifar_dataloaders(test_only = True)
    samp_data = samp_data['test']
    assert (len(samp_data.dataset) > 0)
    class_labels = np.array(range(0, model_setup['num_classes']))

    # Iterate through the DataLoader to collect all data
    for i, (images, labels) in enumerate(samp_data):
        print(images.shape, labels.shape)
        assert np.all(np.isin(labels,  class_labels))
        print("Passed 1")
        assert tuple(images.shape[1:]) == (3, 224, 224)
        print("Passed 2")
