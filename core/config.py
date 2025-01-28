model_setup = {
    "arch" : "VGG16",  # or "VGG11"
    "input_size" : (3, 224, 224),
    "num_classes" : 10,
    "use_cuda": True
}
hyperparams = {
    "learning_rate" : 1e-2,
    "num_epochs" : 100,  # Early stopping will end training before 1000 epochs if validation loss plateaus
    "weight_decay" : 5 * 1e-4,
    "momentum" : 0.9
}

weights = {
    repo_id = "ikinglopez1/vgg4cifar"  # e.g., "yourname/my-model"
    filename = "model.pth"  # The file you uploaded to the repo
}