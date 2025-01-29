from src.core.model import VGG_Network
from src.core.config import model_setup, weights, paths
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import os


def setup_device():
    device = (
        "cuda"
        if torch.cuda.is_available() and model_setup['use_cuda']
        else "cpu"
    )
    return device

def transfer_weights_from_state_dict(state_dict, model):
    """
    Transfer the first Conv layer of each block from VGG11 to the corresponding block in VGG16.
    Randomly initialize additional layers.
    """
    with torch.no_grad():
        # Block 1
        model.conv2d_block1[0].weight.copy_(state_dict['conv2d_block1.0.weight'])
        model.conv2d_block1[0].bias.copy_(state_dict['conv2d_block1.0.bias'])

        # Block 2
        model.conv2d_block2[0].weight.copy_(state_dict['conv2d_block2.0.weight'])
        model.conv2d_block2[0].bias.copy_(state_dict['conv2d_block2.0.bias'])

        # Block 3
        model.conv2d_block3[0].weight.copy_(state_dict['conv2d_block3.0.weight'])
        model.conv2d_block3[0].bias.copy_(state_dict['conv2d_block3.0.bias'])

        model.conv2d_block3[2].weight.copy_(state_dict['conv2d_block3.2.weight'])
        model.conv2d_block3[2].bias.copy_(state_dict['conv2d_block3.2.bias'])


        # Fully connected layers
        model.linear1.weight.copy_(state_dict['linear1.weight'])
        model.linear1.bias.copy_(state_dict['linear1.bias'])
        model.linear2.weight.copy_(state_dict['linear2.weight'])
        model.linear2.bias.copy_(state_dict['linear2.bias'])
        model.linear3.weight.copy_(state_dict['linear3.weight'])
        model.linear3.bias.copy_(state_dict['linear3.bias'])
        
        # Randomly initialize other Conv layers
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if not hasattr(layer.weight, "_is_transferred"):  # Avoid re-initializing transferred layers
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    print("Successfully transferred compatible weights.")

def verify_or_download_weights(pretrained_weights_arch):
    model = pretrained_weights_arch
    repo_id = weights['repo_id']
    models_dir = paths['local_models_dir']
    if not os.path.isdir(models_dir):
        print("Creating models directory")
        os.mkdir(f'{models_dir}')
    if not os.path.isfile(f'{models_dir}/{model}.pth'):

        try:
            model_path = hf_hub_download(
                repo_id=repo_id, 
                filename=f"{model}.pth", 
                local_dir="models"
            )

            print(f"Downloaded to {model_path}")
        except Exception as e:
            raise HFModelNotFoundException("HuggingFace model not found!")
            
    else:
        print(f"{model}.pth found in {models_dir}!")

def get_pretrained_weights(models_dir, pretrained_weights_arch, device):
    try:
        weights_path = os.path.join(models_dir, f"{pretrained_weights_arch}.pth")
        
        # Check if the file exists before loading
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")

        weights_object = torch.load(weights_path, map_location=torch.device(device))
        
        if 'model_state_dict' in weights_object:
            return weights_object['model_state_dict']
        return weights_object

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Failed to load pretrained weights: {e}")

    return None  # Return None if loading fails

    
class WeightTransferException(Exception):
    pass

class HFModelNotFoundException(Exception):
    pass

def load_model():
    input_size = model_setup['input_size']
    num_classes = model_setup['num_classes']
    arch = model_setup['arch'] 
    use_pretrained_weights = model_setup['use_pretrained_weights']
    pretrained_weights_arch = model_setup['pretrained_weights_arch']
    models_dir = paths['local_models_dir']
    device = setup_device()

    vgg_model = VGG_Network(input_size, num_classes, config=arch).to(device)
    if use_pretrained_weights:
        verify_or_download_weights(pretrained_weights_arch)
        pretrained_weights = get_pretrained_weights(models_dir, pretrained_weights_arch, device)
        if pretrained_weights_arch == arch: 
            vgg_model.load_state_dict(pretrained_weights)
            print("Successfully loaded pre-trained weights.")
        elif pretrained_weights_arch == 'vgg11' and arch == 'vgg16': 
            transfer_weights_from_state_dict(pretrained_weights, vgg_model)
        else:
            raise WeightTransferException(f"Weights transfer from {pretrained_weights_arch} to {arch} not supported!")
    return vgg_model


