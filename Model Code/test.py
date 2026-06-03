import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.decomposition import PCA

# ==========================================
# DATA AUGMENTATION FUNCTIONS
# ==========================================


def apply_gaussian_noise(x, std_dev=0.2):
    """Injects random static to prevent the model from memorizing tiny spikes."""
    noise = np.random.normal(loc=0.0, scale=std_dev, size=x.shape)
    return x + noise


def apply_amplitude_scaling(x, scale_factor=1.5):
    """Simulates a fall happening closer to or further from the router."""
    return x * scale_factor


def apply_time_warping(x, speed_factor=1.3):
    """
    Simulates a faster (>1.0) or slower (<1.0) fall.
    Uses PyTorch interpolation to safely maintain the exact 950 timeline size.
    """
    # 1. Convert to PyTorch tensor and copy to avoid memory mapping errors
    x_tensor = torch.tensor(x.copy())

    # 2. Swap axes: PyTorch interpolate expects (Batch, Channels, Time) -> (1, 52, 950)
    x_tensor = x_tensor.permute(0, 2, 1)

    # 3. Calculate the new warped timeline length
    original_length = x_tensor.shape[2]
    warped_length = int(original_length / speed_factor)

    # 4. Stretch or compress the wave
    warped_tensor = F.interpolate(
        x_tensor, size=warped_length, mode='linear', align_corners=False)

    # 5. Resample exactly 950 points back out of the warped curve!
    restored_tensor = F.interpolate(
        warped_tensor, size=original_length, mode='linear', align_corners=False)

    # 6. Swap axes back to (Batch, Time, Subcarriers) -> (1, 950, 52)
    return restored_tensor.permute(0, 2, 1).numpy()

# ==========================================
# MAIN SCRIPT
# ==========================================


# Find every single MATLAB (.mat) file in this specific activity folder
mat_files = glob.glob(
    "C:/Users/brice/Desktop/Brice/Centrale Supelec/1ère année/Stage Zoe Care/ESP-Fi-HAR-1.0/Model Code/Data/Env.1(corridor)/fall/1-1-7-1.mat")
print(f"Found {len(mat_files)} .mat files to process.")
for mat_path in mat_files:
    try:
        # Read the MATLAB file into a Python dictionary
        mat = sio.loadmat(mat_path)

        # Ensure the data we need (CSIamp) actually exists in the file
        if "CSIamp" not in mat:
            print(f"Warning: 'CSIamp' not found in {mat_path}")
            continue

        # Extract the raw NumPy array
        x_raw = mat["CSIamp"]

        # Standardize the shape to (Time: 950, Subcarriers: 52)
        if x_raw.shape == (950, 52):
            x_raw = x_raw
        elif x_raw.shape == (52, 950):
            x_raw = x_raw.T
        else:
            raise ValueError(f"Unexpected shape {x_raw.shape} in {mat_path}")

        # 1. Butterworth Band-Pass Filter (0.5 Hz - 10.0 Hz)
        fs = 100
        nyquist = fs / 2
        b, a = butter(4, [0.5 / nyquist, 10.0 / nyquist], btype='band')

        # Apply zero-phase filter (axis=0 ensures it filters down the time steps!)
        x_filtered = filtfilt(b, a, x_raw, axis=0)

        # 2. PCA (Spatial compression of the filtered data)
        pca = PCA(n_components=3)
        scores = pca.fit_transform(x_filtered)

        # 3. Reconstruct back to 52 subcarriers for the CNN
        x = pca.inverse_transform(scores)

        # Z-score normalization
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        # Reshape to (1, 950, 52)
        x = x.reshape(1, 950, 52).astype(np.float32)

        # Apply the augmentations independently to see their effects
        x_noisy = apply_gaussian_noise(x, std_dev=0.2)
        x_scaled = apply_amplitude_scaling(x, scale_factor=0.8)
        x_warped = apply_time_warping(x, speed_factor=2.5)

        # ==========================================
        # VISUALIZATION (4 Subplots)
        # ==========================================
        fig = plt.figure(figsize=(12, 6))
        filename = os.path.basename(mat_path)

        # 1. Original Data
        x_raw = x_raw.reshape(1, 950, 52).astype(np.float32)
        plt.plot(x_raw[0, :, :], alpha=0.6)
        plt.title(f"1. ORIGINAL DATA: {filename}", fontweight='bold')
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
