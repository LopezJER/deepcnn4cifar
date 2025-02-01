model_setup = {
    "arch": "vgg16",
    "use_pretrained_weights": True,  ### OUR pretrained weights
    "pretrained_weights_arch": "vgg16",  # for example, if current architecture is VGG16,
    # but we want to transfer some pretrained VGG11 layer weights
    # to current VGG16 architectue
    "input_size": (3, 224, 224),
    "val_split": 0.2,
    "num_classes": 10,
    "use_cuda": True,
}

debug = {
    "on": True,
    "train_size": 64,
    "val_size": 8,
    "test_size": 64,
    "batch_size": 8,
    "num_epochs": 2,
}

hyperparams = {
    "learning_rate": 1e-2,
    "batch_size": 64,
    "num_epochs": 3,  # Early stopping will end training before 100 epochs if validation loss plateaus
    "weight_decay": 5 * 1e-4,
    "momentum": 0.9,
    "early_stopping_patience": 5,
}

weights = {
    "repo_id": "ikinglopez1/vgg4cifar",
}

paths = {"local_models_dir": "models", "outputs_dir": "outputs"}

class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]
