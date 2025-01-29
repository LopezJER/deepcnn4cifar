import pytest
from src.utils.load_model import load_model, setup_device, get_pretrained_weights, verify_or_download_weights, transfer_weights_from_state_dict
from unittest import mock
from src.core.model import VGG_Network
from src.core.config import model_setup


def test_setup_device():
    device = setup_device()
    assert device in ("cuda", "cpu") 

def test_transfer_weights_from_state_dict():
    vgg_model = VGG_Network(model_setup['input_size'], model_setup['num_classes'], config='vgg16').to('cpu')
    with pytest.raises(KeyError):
        transfer_weights_from_state_dict({}, vgg_model)

def correct_input_size(model):
    return model.input_size == model_setup['input_size']

def correct_num_classes(model):
    return model.num_classes == 10

def correct_arch(model):
    return model.arch in ['vgg16', 'vgg11']

def test_load_model():
    fake_model_setup = {
        'input_size': model_setup['input_size'],
        'num_classes': model_setup['num_classes'],
        'arch': model_setup['arch'],
        'use_pretrained_weights': False,
        'pretrained_weights_arch': model_setup['pretrained_weights_arch'],
    }

    with mock.patch('src.utils.load_model.model_setup', fake_model_setup):
        vgg_model = load_model()
        assert correct_input_size(vgg_model) and correct_num_classes(vgg_model) and correct_arch(vgg_model)
