model_setup = {
    "arch" : "vgg16", 
    "use_pretrained_weights": True,
    "pretrained_weights_arch": "vgg11", #for example, if current architecture is VGG16,
                                        #but we want to transfer some pretrained VGG11 layer weights
                                        #to current VGG16 architectue
    "input_size" : (3, 224, 224),
    "num_classes" : 10,
    "use_cuda": True,

}
hyperparams = {
    "learning_rate" : 1e-2,
    "num_epochs" : 3,  # Early stopping will end training before 100 epochs if validation loss plateaus
    "weight_decay" : 5 * 1e-4,
    "momentum" : 0.9,
    "early_stopping_patience" : 5
}

weights = {
    "repo_id" : "ikinglopez1/vgg4cifar",
}

paths = {
    "local_models_dir" : "models",
    "outputs_dir" : "outputs"
}