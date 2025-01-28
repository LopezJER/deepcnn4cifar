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
