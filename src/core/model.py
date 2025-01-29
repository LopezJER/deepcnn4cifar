from torch import nn

class VGG_Network(nn.Module):
    def __init__(self, input_size, num_classes, config='vgg16'):
        super(VGG_Network, self).__init__()
        conv_blocks = []

        if config == 'vgg11':
            print("Constructing VGG11")
            self.conv2d_block1 = self.conv2d_block(input_size[0], 64, 1)
            self.conv2d_block2 = self.conv2d_block(64, 128, 1)
            self.conv2d_block3 = self.conv2d_block(128, 256, 2)
            self.conv2d_block4 = self.conv2d_block(256, 512, 2)
            self.conv2d_block5 = self.conv2d_block(512, 512, 2)

        elif config == 'vgg16':
            print("Constructing VGG16")
            self.conv2d_block1 = self.conv2d_block(input_size[0], 64, 2)
            self.conv2d_block2 = self.conv2d_block(64, 128, 2)
            self.conv2d_block3 = self.conv2d_block(128, 256, 3)
            self.conv2d_block4 = self.conv2d_block(256, 512, 3)
            self.conv2d_block5 = self.conv2d_block(512, 512, 3)

        self.linear1 = nn.Linear(512 * 7 * 7, 4096)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(4096, num_classes)

    def conv2d_block(self, in_channels, out_channels, num_layers):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers-1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv2d_block1(x)
        x = self.conv2d_block2(x)
        x = self.conv2d_block3(x)
        x = self.conv2d_block4(x)
        x = self.conv2d_block5(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        return nn.functional.log_softmax(x, dim=1) #output

