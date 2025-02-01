import torch.optim as optim
from src.core.config import hyperparams


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
