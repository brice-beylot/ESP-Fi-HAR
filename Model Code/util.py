# Utility functions for ESP-Fi HAR
# Centralized "Factory" for initializing models, datasets, and hyperparameters

from dataset import ESP_Fi_HAR_Dataset
from ESP_Fi_model import *
import torch


def load_data_n_model(dataset_name, model_name, root):
    """
    Load ESP-Fi HAR dataset and corresponding model.
    This function acts as a centralized configuration hub, ensuring the correct
    hyperparameters (epochs, batch size) are automatically paired with the chosen model.

    Args:
        dataset_name (str): 'ESP_Fi_HAR' (Used as a safety check)
        model_name (str): The architecture to build (e.g., 'ResNet18', 'Transformer')
        root (str): Dataset root directory path

    Returns:
        data_loader (list): A list of 4 DataLoaders (one for each environment)
        model (nn.Module): The instantiated PyTorch neural network
        train_epoch (int): The target number of training epochs for this specific model
    """

    # 1. Safety Check: Ensure the user didn't misspell the dataset name
    if dataset_name != 'ESP-Fi_HAR':
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Only 'ESP-Fi_HAR' is supported."
        )

    print('Using dataset: ESP-Fi HAR')

    # Run, Walk, Jump, Squat, Arm Wave, Turn, Fall
    num_classes = 7

    # =====================
    # Model Selection & Hyperparameter Tuning
    # =====================
    # Different architectures require different training strategies. 
    # For example, recurrent models (LSTM/GRU) often need more epochs to converge,
    # while heavy attention models (Transformer) require tiny batch sizes to prevent GPU memory crashes.

    if model_name == 'CNN':
        model = CNN(num_classes)
        train_epoch = 50
        batch_size = 32

    elif model_name == 'ResNet18':
        model = ESP_Fi_ResNet18(num_classes)
        train_epoch = 50
        batch_size = 32

    elif model_name == 'Transformer':
        model = ESP_Fi_Transformer(num_classes)
        train_epoch = 100
        # Notice the tiny batch size (4)! Transformers compute "attention" between 
        # every single time-step (950x950 matrix), which consumes massive VRAM.
        batch_size = 4

    elif model_name == 'GRU':
        model = ESP_Fi_GRU(num_classes)
        train_epoch = 100
        batch_size = 64

    elif model_name == 'LSTM':
        model = ESP_Fi_LSTM(num_classes)
        train_epoch = 100
        batch_size = 32

    elif model_name == 'MobileNetV3':
        model = MobileNetV3(num_classes)
        train_epoch = 50
        batch_size = 32

    elif model_name == 'EfficientNetLite':
        model = EfficientNetLite(num_classes)
        train_epoch = 50
        batch_size = 32

    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # =====================
    # Dataset & DataLoader Assembly
    # =====================
    # We create 4 distinct DataLoaders, one for each physical room/environment.
    # CRITICAL: shuffle=False is used for ALL of them. 
    # Because we are using Leave-One-Environment-Out Cross-Validation, these individual 
    # loaders act strictly as the *Test Sets*. The actual shuffling of the training data
    # happens later in run.py when we combine 3 of them using ConcatDataset.

    env1_loader = torch.utils.data.DataLoader(
        dataset=ESP_Fi_HAR_Dataset(
            root_dir=root,
            split='Env.1(corridor)'
        ),
        batch_size=batch_size, 
        shuffle=False
    )

    env2_loader = torch.utils.data.DataLoader(
        dataset=ESP_Fi_HAR_Dataset(
            root_dir=root,
            split='Env.2(office)'
        ),
        batch_size=batch_size, 
        shuffle=False
    )

    env3_loader = torch.utils.data.DataLoader(
        dataset=ESP_Fi_HAR_Dataset(
            root_dir=root,
            split='Env.3(boardrooms)'
        ),
        batch_size=batch_size, 
        shuffle=False
    )

    env4_loader = torch.utils.data.DataLoader(
        dataset=ESP_Fi_HAR_Dataset(
            root_dir=root,
            split='Env.4(laboratory)'
        ),
        batch_size=batch_size,
        shuffle=False
    )

    # Package them neatly into a list so run.py can easily loop through them for Cross-Validation
    data_loader = [env1_loader, env2_loader, env3_loader, env4_loader]

    return data_loader, model, train_epoch