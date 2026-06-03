# ESP-Fi HAR Dataset Loader
# Input size: 1 × 950 × 52 (Channels x Time x Subcarriers)
# Modality: CSI Amplitude (CSIamp)

import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import uniform_filter1d

class ESP_Fi_HAR_Dataset(Dataset):
    """
    ESP-Fi HAR Dataset

    Directory structure:
    root_dir/
        Env.1(corridor)/
            arm_wave/
            fall/
            ...
    
    Each .mat file should contain:
        CSIamp: ndarray with shape (950, 52) or (52, 950)

    Output:
        x: Tensor of shape (1, 950, 52)
        y: LongTensor label
    """

    def __init__(self,
                 root_dir: str,
                 split: str = "Env.1(corridor)",
                 modal: str = "CSIamp",
                 transform=None):
        """
        Args:
            root_dir (str): Root directory of ESP-Fi HAR dataset
            split (str): The specific environment folder to load (e.g., 'Env.1(corridor)')
            modal (str): The dictionary key to look for inside the .mat file (default: CSIamp)
            transform (callable, optional): Optional data augmentations/transforms
        """
        # Save parameters to the class instance
        self.root_dir = root_dir
        self.split = split
        self.modal = modal
        self.transform = transform
        
        # Define the exact order of activities. 
        # The index in this list automatically becomes the integer label (e.g., 'run' = 0, 'fall' = 6)
        self.activities = ['run', 'walk', 'jump',
                           'squat', 'arm_wave', 'turn', 'fall']

        # Initialize empty lists to hold our data and labels in RAM
        self.data = []
        self.labels = []
        self.file_paths = []

        # Immediately validate the folder paths and load the data upon creation
        self._check_dataset_structure()
        self._load_dataset()

    def _check_dataset_structure(self):
        """Check whether dataset directory structure is valid before loading to avoid hidden crashes."""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(
                f"Root directory not found: {self.root_dir}")

        split_dir = os.path.join(self.root_dir, self.split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Ensure every single activity folder exists inside the environment
        for act in self.activities:
            act_path = os.path.join(split_dir, act)
            if not os.path.exists(act_path):
                raise FileNotFoundError(f"Activity folder missing: {act_path}")

    def _load_dataset(self):
        """Load all CSI samples from the hard drive into RAM for fast training"""
        print(f"[ESP-Fi HAR] Loading {self.split} set...")
        print("[ESP-Fi HAR] Activity order:", self.activities)

        split_dir = os.path.join(self.root_dir, self.split)

        # Loop through each activity and its corresponding integer label
        for label, act in enumerate(self.activities):
            act_dir = os.path.join(split_dir, act)
            
            # Find every single MATLAB (.mat) file in this specific activity folder
            mat_files = glob.glob(os.path.join(act_dir, "*.mat"))

            for mat_path in mat_files:
                try:
                    # Read the MATLAB file into a Python dictionary
                    mat = sio.loadmat(mat_path)

                    # Ensure the data we need (CSIamp) actually exists in the file
                    if self.modal not in mat:
                        print(
                            f"Warning: '{self.modal}' not found in {mat_path}")
                        continue

                    # Extract the raw NumPy array
                    x = mat[self.modal]

                    # Standardize the shape to (Time: 950, Subcarriers: 52)
                    if x.shape == (950, 52):
                        pass
                    elif x.shape == (52, 950):
                        x = x.T  # Transpose the matrix if it was saved backwards
                    else:
                        raise ValueError(
                            f"Unexpected shape {x.shape} in {mat_path}"
                        )

                    # Z-score normalization: (Value - Mean) / Standard Deviation
                    # This forces the data to have a mean of 0 and a standard deviation of 1.
                    # It prevents large signal spikes from destabilizing the neural network gradients.
                    # 1e-8 is added to prevent a mathematically illegal "divide by zero" error.
                    x = (x - np.mean(x)) / (np.std(x) + 1e-8)

                    # Reshape to (1, 950, 52) to mimic a 1-channel grayscale image.
                    # PyTorch Conv2D layers strictly require a "Channel" dimension at the front.
                    x = x.reshape(1, 950, 52).astype(np.float32)

                    # Store the processed sample and its label in RAM
                    self.data.append(x)
                    self.labels.append(label)
                    self.file_paths.append(mat_path)

                except Exception as e:
                    print(f"Error loading {mat_path}: {e}")

        # Convert the massive Python lists into highly optimized NumPy arrays
        self.data = np.asarray(self.data, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64)

        # Fail-safe if the folders were empty
        if len(self.data) == 0:
            raise RuntimeError("No valid data samples loaded.")

        print(f"[ESP-Fi HAR] Loaded {len(self.data)} samples.")

    def __len__(self):
        """Mandatory PyTorch function: Tells the DataLoader how many total samples exist."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Mandatory PyTorch function: Fetches a single sample when the DataLoader asks for it.
        This is where we transition from NumPy arrays to PyTorch Tensors.
        """
        # 1. Fetch the raw NumPy array for this specific sample
        raw_x = self.data[idx]

        # 2. Apply the Moving Average Denoising
        # size=5 means it averages 5 time-steps at a time. axis=-1 ensures it only smooths the time dimension.
        smoothed_x = uniform_filter1d(raw_x, size=5, axis=-1)

        # 3. Convert the smoothed NumPy array into a PyTorch Tensor
        x = torch.from_numpy(smoothed_x.copy())

        # 4. Fetch the label
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        # 5. Apply any optional data augmentations (like random noise or cropping) if provided
        if self.transform:
            x = self.transform(x)

        # 6. Capture the path (usefull for displaying data)
        path = self.file_paths[idx]

        return x, y, path


def get_dataloader(root_dir,
                   split="Env.1(corridor)",
                   batch_size=64,
                   shuffle=False,
                   num_workers=0):
    """
    Build DataLoader for ESP-Fi HAR.
    A DataLoader acts as the 'engine' that pumps batches of data from the Dataset into the GPU.

    Args:
        root_dir (str): Dataset root directory
        split (str): Environment folder to load
        batch_size (int): How many samples to feed the model at once
        shuffle (bool): Whether to randomize the order of the data
        num_workers (int): How many CPU cores to use for background data loading

    Returns:
        torch.utils.data.DataLoader
    """
    # 1. Instantiate the Dataset (loads all data into RAM)
    dataset = ESP_Fi_HAR_Dataset(
        root_dir=root_dir,
        split=split,
        modal="CSIamp"
    )

    # 2. Wrap it in a DataLoader engine
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True # Speeds up the transfer of data from CPU RAM to GPU VRAM
    )