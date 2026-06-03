# ESP-Fi HAR Dataset Loader
# Input size: 1 × 950 × 52 (Channels x Time x Subcarriers)
# Modality: CSI Amplitude (CSIamp)

import glob
import numpy as np
import os
import random
import scipy.io as sio
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
        path : string containing the file path
    """

    def __init__(self,
                 pca_mode: str,
                 filter_name: str,
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
        self.pca_mode = pca_mode
        self.filter_name = filter_name
        self.data_augmentation = False
        
        # Define the exact order of activities. 
        # The index in this list automatically becomes the integer label (e.g., 'run' = 0, 'fall' = 6)
        self.activities = ['run', 'walk', 'jump',
                           'squat', 'arm_wave', 'turn', 'fall']

        # Initialize empty lists to hold our data, labels and file paths in RAM
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
                    
                    # ==================================================
                    # 1. OPTIONAL FILTERING (Operates on Time Axis 0)
                    # ==================================================
                    if self.filter_name == "Butterworth":
                        fs = 100
                        nyquist = fs / 2
                        b, a = butter(4, [0.5 / nyquist, 10.0 / nyquist], btype='band')
                        x = filtfilt(b, a, x, axis=0)
                        
                    elif self.filter_name == "SG":
                        x = savgol_filter(x, window_length=15, polyorder=3, axis=0)

                    # ==================================================
                    # 2. OPTIONAL PCA & RECONSTRUCTION
                    # ==================================================
                    if self.pca_mode != 'None':
                        pca = PCA(n_components=3)
                        scores = pca.fit_transform(x)
                        
                        if self.pca_mode == 'PCA_Reconstructed':
                            x = pca.inverse_transform(scores)
                        elif self.pca_mode == 'PCA_Only':
                            x = scores

                    # ==================================================
                    # 3. NORMALIZATION & SHAPING
                    # ==================================================
                    x = (x - np.mean(x)) / (np.std(x) + 1e-8)

                    # Determine the correct feature dimension
                    feature_dim = 3 if self.pca_mode == 'PCA_Only' else 52
                    
                    # Reshape to (1, 950, features) 
                    x = x.reshape(1, 950, feature_dim).astype(np.float32)
                    
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
        x = torch.from_numpy(raw_x.copy())

        # 2. Fetch the label
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        # ==========================================
        # 3. STOCHASTIC DATA AUGMENTATION (TRAINING ONLY)
        # ==========================================
        if self.data_augmentation:
            
            # a. Amplitude Scaling (30% chance)
            if random.random() < 0.3:
                scale = random.uniform(0.8, 1.2)
                x = x * scale
                
            # b. Gaussian Noise (30% chance)
            if random.random() < 0.3:
                std = random.uniform(0.01, 0.05)
                noise = torch.randn_like(x) * std
                x = x + noise
                
            # c. Time Warping (30% chance)
            if random.random() < 0.3:
                speed = random.uniform(0.8, 1.2)
                
                # Reshape for PyTorch Interpolation: (1, 950, 52) -> (1, 52, 950)
                x = x.permute(0, 2, 1) 
                
                orig_len = x.shape[2]
                warp_len = int(orig_len / speed)
                
                # Stretch/Compress the wave
                x = F.interpolate(x, size=warp_len, mode='linear', align_corners=False)
                # Resample back to exactly 950
                x = F.interpolate(x, size=orig_len, mode='linear', align_corners=False)
                
                # Return to normal shape: (1, 52, 950) -> (1, 950, 52)
                x = x.permute(0, 2, 1)

        # 5. Capture the path (usefull for displaying data)
        path = self.file_paths[idx]
        
        return x, y, path


def get_dataloader(pca_mode,
                   filter_name,
                   root_dir,
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
        pca_mode=pca_mode,
        filter_name=filter_name,
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