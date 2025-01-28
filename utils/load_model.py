from core.model import VGG_Network
from core.config import model_setup
import torch
from huggingface_hub import hf_hub_download

def setup_device():
    device = (
        "cuda"
        if torch.cuda.is_available() and model_setup['use_cuda']
        else "cpu"
    )
    return device

def transfer_weights_from_state_dict(state_dict, vgg16_model):
    """
    Transfer the first Conv layer of each block from VGG11 to the corresponding block in VGG16.
    Randomly initialize additional layers.
    """
    with torch.no_grad():
        # Block 1
        vgg16_model.conv2d_block1[0].weight.copy_(state_dict['conv2d_block1.0.weight'])
        vgg16_model.conv2d_block1[0].bias.copy_(state_dict['conv2d_block1.0.bias'])

        # Block 2
        vgg16_model.conv2d_block2[0].weight.copy_(state_dict['conv2d_block2.0.weight'])
        vgg16_model.conv2d_block2[0].bias.copy_(state_dict['conv2d_block2.0.bias'])

        # Block 3
        vgg16_model.conv2d_block3[0].weight.copy_(state_dict['conv2d_block3.0.weight'])
        vgg16_model.conv2d_block3[0].bias.copy_(state_dict['conv2d_block3.0.bias'])

        vgg16_model.conv2d_block3[2].weight.copy_(state_dict['conv2d_block3.2.weight'])
        vgg16_model.conv2d_block3[2].bias.copy_(state_dict['conv2d_block3.2.bias'])


        # Fully connected layers
        vgg16_model.linear1.weight.copy_(state_dict['linear1.weight'])
        vgg16_model.linear1.bias.copy_(state_dict['linear1.bias'])
        vgg16_model.linear2.weight.copy_(state_dict['linear2.weight'])
        vgg16_model.linear2.bias.copy_(state_dict['linear2.bias'])
        vgg16_model.linear3.weight.copy_(state_dict['linear3.weight'])
        vgg16_model.linear3.bias.copy_(state_dict['linear3.bias'])
        
        # Randomly initialize other Conv layers
        for layer in vgg16_model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if not hasattr(layer.weight, "_is_transferred"):  # Avoid re-initializing transferred layers
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    print("Successfully transferred compatible weights from VGG11 to VGG16.")

def download_model():
    repo_id = "<username>/<repository_name>"  # e.g., "yourname/my-model"
    filename = "model.pth"  # The file you uploaded to the repo

    # Download the .pth file from Hugging Face Hub
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # Load the model weights
    model = YourModelClass()  # Replace with your model architecture
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

def load_model():
    input_size = model_setup['arch']
    num_classes = model_setup['num_classes']
    arch = model_setup['arch']                         ]
    device = setup_device()
    vgg16_weights = download_model())
    
    # Initialize the VGG16 model
    vgg16_model = VGG_Network(model_setup['input_size'], model_setup['num_classes'], config=model_setup['arch']).to(device)

    # Transfer weights from VGG11 to VGG16
    print("Transferring weights from VGG11 to VGG16...")
    transfer_weights_from_state_dict(state_dict, vgg16_model)


    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg16_model.parameters(), momentum=momentum, lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)
