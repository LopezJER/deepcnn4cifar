import logging
from torch import nn

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class VGG_Network(nn.Module):
    """
    VGG Network model class based on VGG11 or VGG16 architectures for image classification.
    
    This class constructs the VGG model with configurable input size, number of output classes,
    and network architecture (VGG11 or VGG16).

    Attributes:
        input_size (tuple): The input size of the image (channels, height, width).
        num_classes (int): The number of output classes for classification.
        arch (str): The architecture of the VGG network (either 'vgg11' or 'vgg16').
        conv2d_block1 (nn.Sequential): The first convolutional block.
        conv2d_block2 (nn.Sequential): The second convolutional block.
        conv2d_block3 (nn.Sequential): The third convolutional block.
        conv2d_block4 (nn.Sequential): The fourth convolutional block.
        conv2d_block5 (nn.Sequential): The fifth convolutional block.
        linear1 (nn.Linear): The first fully connected layer.
        relu1 (nn.ReLU): ReLU activation for the first fully connected layer.
        dropout1 (nn.Dropout): Dropout for the first fully connected layer.
        linear2 (nn.Linear): The second fully connected layer.
        relu2 (nn.ReLU): ReLU activation for the second fully connected layer.
        dropout2 (nn.Dropout): Dropout for the second fully connected layer.
        linear3 (nn.Linear): The final output layer.
    """

    def __init__(self, input_size, num_classes, config='vgg16'):
        """
        Initializes the VGG Network object based on the specified configuration.
        
        Args:
            input_size (tuple): The input size of the image (channels, height, width).
            num_classes (int): The number of output classes for classification.
            config (str): The configuration for the VGG network architecture ('vgg11' or 'vgg16'). Default is 'vgg16'.
        """
        super(VGG_Network, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.arch = config

        if self.arch == 'vgg11':
            logger.info("Constructing VGG11")
            self.conv2d_block1 = self.conv2d_block(self.input_size[0], 64, 1)
            self.conv2d_block2 = self.conv2d_block(64, 128, 1)
            self.conv2d_block3 = self.conv2d_block(128, 256, 2)
            self.conv2d_block4 = self.conv2d_block(256, 512, 2)
            self.conv2d_block5 = self.conv2d_block(512, 512, 2)

        elif self.arch == 'vgg16':
            logger.info("Constructing VGG16")
            self.conv2d_block1 = self.conv2d_block(self.input_size[0], 64, 2)
            self.conv2d_block2 = self.conv2d_block(64, 128, 2)
            self.conv2d_block3 = self.conv2d_block(128, 256, 3)
            self.conv2d_block4 = self.conv2d_block(256, 512, 3)
            self.conv2d_block5 = self.conv2d_block(512, 512, 3)

        # Fully connected layers
        self.linear1 = nn.Linear(512 * 7 * 7, 4096)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(4096, self.num_classes)

    def conv2d_block(self, in_channels, out_channels, num_layers):
        """
        Creates a convolutional block consisting of multiple convolutional layers 
        followed by ReLU activations and max pooling.
        
        Args:
            in_channels (int): The number of input channels to the first convolutional layer.
            out_channels (int): The number of output channels for each convolutional layer.
            num_layers (int): The number of convolutional layers in the block.
        
        Returns:
            nn.Sequential: A sequential container of convolutional layers, activations, and pooling layers.
        """
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers-1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        logger.debug(f"Created a conv2d block with {num_layers} layers.")

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass through the VGG network.
        
        Args:
            x (Tensor): The input tensor representing the image batch.
        
        Returns:
            Tensor: The log-softmax output tensor of the model for classification.
        """
        x = self.conv2d_block1(x)
        x = self.conv2d_block2(x)
        x = self.conv2d_block3(x)
        x = self.conv2d_block4(x)
        x = self.conv2d_block5(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        logger.info("Forward pass complete. Output generated.")
        return nn.functional.log_softmax(x, dim=1)  # Output of the model
