import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from src.core.model import VGG_Network
from src.core.config import model_setup, weights, paths
import logging
import torch.optim as optim
from src.core.config import hyperparams


# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def setup_device():
    """
    Set up the device for model training/inference based on availability of CUDA.

    Returns:
        str: Device type ('cuda' or 'cpu')
    """
    device = (
        "cuda"
        if torch.cuda.is_available() and model_setup['use_cuda']
        else "cpu"
    )
    return device

def transfer_weights_from_state_dict(state_dict, model):
    """
    Transfer weights from a pre-trained model (e.g., VGG11) to a VGG16 model.
    Transfers the first Conv layer of each block and initializes additional layers randomly.

    Args:
        state_dict (dict): Pre-trained model's state_dict with weights.
        model (nn.Module): The model to which weights are transferred.
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

    logger.info("Successfully transferred compatible weights.")

def verify_or_download_weights(pretrained_weights_arch):
    """
    Verifies if the pre-trained weights are present locally. If not, downloads them from HuggingFace.

    Args:
        pretrained_weights_arch (str): Architecture of the pre-trained weights (e.g., 'vgg11', 'vgg16').

    Raises:
        HFModelNotFoundException: If the model is not found on HuggingFace.
    """
    model = pretrained_weights_arch
    repo_id = weights['repo_id']
    models_dir = paths['local_models_dir']
    
    if not os.path.isdir(models_dir):
        logger.info("Creating models directory")
        os.mkdir(f'{models_dir}')
    
    if not os.path.isfile(f'{models_dir}/{model}.pth'):
        try:
            model_path = hf_hub_download(
                repo_id=repo_id, 
                filename=f"{model}.pth", 
                local_dir="models"
            )
            logger.info(f"Downloaded to {model_path}")
        except Exception as e:
            raise HFModelNotFoundException("HuggingFace model not found!")
    else:
        logger.info(f"{model}.pth found in {models_dir}!")

def get_pretrained_weights(models_dir, pretrained_weights_arch, device):
    """
    Loads pre-trained weights from the specified local directory.

    Args:
        models_dir (str): Directory containing the pre-trained models.
        pretrained_weights_arch (str): Name of the pre-trained model architecture.
        device (str): Device to load the model weights onto ('cpu' or 'cuda').

    Returns:
        dict or None: The loaded weights dictionary or None if an error occurs.
    """
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
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"Failed to load pretrained weights: {e}")

    return None  # Return None if loading fails

class WeightTransferException(Exception):
    """Custom exception for weight transfer errors."""
    pass

class HFModelNotFoundException(Exception):
    """Custom exception when the HuggingFace model is not found."""
    pass

def load_model():
    """
    Loads the VGG model with optional pre-trained weights. Transfers weights from VGG11 to VGG16 if needed.

    Returns:
        VGG_Network: The instantiated VGG model with transferred or loaded weights.
    """
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
            logger.info("Successfully loaded pre-trained weights.")
        elif pretrained_weights_arch == 'vgg11' and arch == 'vgg16': 
            transfer_weights_from_state_dict(pretrained_weights, vgg_model)
        else:
            raise WeightTransferException(f"Weights transfer from {pretrained_weights_arch} to {arch} not supported!")
    
    return vgg_model

def get_hyperparams(model):
    """
    Configure hyperparameters including optimizer and scheduler.

    Args:
        model (torch.nn.Module): The neural network model.

    Returns:
        dict: Dictionary containing hyperparameters.
    """
    try:
        optimizer = optim.SGD(
            model.parameters(),
            momentum=hyperparams["momentum"],
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.1
        )

        hyps = {key: value for key, value in hyperparams.items()}
        hyps["optimizer"], hyps["scheduler"] = optimizer, scheduler
        return hyps
    except KeyError as e:
        print(f"Missing hyperparameter: {e}")
        raise

