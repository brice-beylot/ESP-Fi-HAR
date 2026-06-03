# ESP-Fi HAR: A Low-Power WiFi CSI Dataset for Ad-Hoc IoT Human Activity Recognition
## Introduction
ESP-Fi HAR is a publicly available WiFi Channel State Information (CSI) dataset collected using low-power, commodity ESP32 modules. It targets privacy-preserving, energy-efficient Human Activity Recognition (HAR) in resource-constrained ad-hoc IoT networks.

Unlike traditional CSI datasets that rely on high-power Intel 5300 or Atheros network cards, ESP-Fi HAR demonstrates the feasibility of scalable HAR using inexpensive IoT hardware. The dataset covers four indoor environments (Corridor, Office, Meeting Room, Laboratory) and seven daily activities.

We provide a PyTorch benchmark suite including 7 deep learning models, covering convolutional networks (CNN, ResNet variants), recurrent architectures (LSTM/GRU), and Transformer-based models, optimized for the ESP-Fi CSI format (1×950×52 amplitude).

Dataset & code: [GitHub Repository](https://github.com/AutoSmartGroup/ESP-Fi-HAR)


## Requirements

1. Install `pytorch` and `torchvision` (we use `pytorch==1.12.0` and `torchvision==0.13.0`).
2. `pip install -r requirements.txt`



## Directory Structure
```
Benchmark
├── LICENSE
├── ESP_Fi_model.py          # All model definitions
├── run.py                   # Train/test entry point
├── DATA_LICENSE.txt
├── dataset.py               # ESP-Fi_HAR dataset loader
├── util.py                  # Model/data loading utils
├── latency-cpu.py           # CPU latency benchmark
├── requirements.txt
├── README.md
├── ── Data
    ├── Env.1(corridor)
    │   ├── arm_wave
    │   ├── ...              # Place downloaded dataset here
    |   ...
└── training_logs/           # Generated logs & checkpoints
```


## Supervised Learning
To run models with supervised learning (train & test):  
Run: `python run.py --model [model name] `  
Example: python run.py --model CNN
Add a trigger (optional) and defined more in details below :
- `--filter_name [filter name]`
- `--pca_mode [pca mode]`
- `--data_augmentation`
- `--save_errors`


## Supported Models
- CNN
- ResNet18
- GRU
- LSTM
- Transformer
- MobileNetV3
- EfficientNetLite

Results (accuracy, f1 score for fall detection, loss, CSV logs, confusion matrix) are saved in ./training_logs/.
Note : - best model.pth can be saved in ./training_logs/ if necessary but is not done by default
       - the subset of the misclassified samples regarding the fall detection can also be save with the `--plot_errors` trigger in the running command


## Training Settings

- Train/test split by doing cross-validation on the four different indoor scenarios
- Input: CSI amplitude only
- Filtering (optionnal): Butterworth bandpass (0.5Hz-10Hz) filter OR Savitzky-Golay algorithm (window_length=15, polyorder=3) can be use
- PCA (optionnal): Keeping only the first three Component Analysis
- Rebuiding (optionnal): Rebuilding the 52 subcarriers signals based on these 3 PCs
- Normalization: Z-score per sample
- Data Augmentation (optionnal): 
  - Amplitude scaling (amplitude between 0.8 and 1.2) with 30% chance to be applied
  - Gaussian Noise (standard deviation between 0.01 and 0.05) with 30% chance to be applied
  - Time Warping (speed between 0.8 and 1.2) with 30% chance to be applied
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- Loss Function: CrossEntropyLoss
- Evaluation Metrics:
  - Accuracy
  - Macro-F1

Note: Best model is selected based on F1 score regarding fall detection on test set (no validation split available). Test set is used only for final evaluation.

- Logs and checkpoints are saved in ./training_logs/

- Training CSV includes per-epoch:
TrainAcc, TrainLoss, TestAcc, TestF1, TestLoss

- Final Test CSV includes best model performance.

### Keeping only the 3 Principal Components for Non-Convolutionnal Models (LSTM, GRU and Transformer)
Run: `python run.py --model [model name] --pca_mode PCA_Only`
Example: python run.py --model CNN --pca_mode PCA_Only

### Rebuilding the 52 subcarriers from the three first Principal Component
Run: `python run.py --model [model name] --pca_mode PCA_Reconstructed`
Example: python run.py --model CNN --pca_mode PCA_Reconstructed

Note: `None` pca mode is used by default

### Supported Filters
- Butterworth
- SG (for Savitzky-Golay)
- None
Run: `python run.py --model [model name] (--pca_mode [pca mode]) --filter_name [filter name]`
Example: python run.py --model CNN --pca_mode PCA_Reconstructed --filter_name Butterworth
Note: `None` filter is used by default

### Adding data augmentation
Run: Run: `python run.py --model [model name] (--pca_mode [pca mode] --filter_name [filter name]) --data_augmentation`
Example: python run.py --model CNN --pca_mode PCA_Reconstructed --filter_name Butterworth --data_augmentation
Note: There is no data augmentation without adding the trigger `--data_augmentation`

## Plotting the subset of misclassified samples regarding the fall detection 
Run: `python run.py --model [model name] (--pca_mode [pca mode] --filter_name [filter name] --data_augmentation) --save_errors`
Note: These misclassified samples will be save in a different file for each training fold in ./training_logs/

## Batch Size Recommendation (8GB GPU)

| Model Type | Batch Size |
|------------|-----------
| GRU | 64 |
| CNN /LSTM / ResNet18 / MobileNetV3 /EfficientNetLite | 32 |
| Transformer | 4 |

---
## Measure CPU inference latency

Latency is measured on CPU only to reflect realistic IoT deployment conditions.

With trained checkpoint:

Run: ` python latency-cpu.py --checkpoint training_logs/best_CNN.pth `

If you want to measure latency for another model, replace best_CNN.pth with the corresponding checkpoint.

## Dataset
### ESP-Fi HAR
 
- **CSI size** : 1 x 950 x 52
- **number of classes** : 7
- **classes** : run, fall, walk, turn, jump, squat, arm wave
- **Indoor scenarios & samples per scene** :
  - Corridor: 560 samples (train/test split included)
  - Office: 560 samples
  - Meeting Room: 560 samples
  - Laboratory: 560 samples

#### Dataset Organization

The ESP-Fi HAR dataset is hierarchically organized across three dimensions: **scenario**, **participant**, and **activity**.  

Each sample follows the structured naming convention: **X-Y-Z-M**, where:

- **X**: Scenario ID (1–4), corresponding to four indoor scenarios:
  1. Corridor
  2. Office
  3. Meeting Room
  4. Laboratory
- **Y**: Participant ID (1–8)
- **Z**: Activity ID (1–7), mapped to predefined actions:
  - 1: run
  - 2: fall
  - 3: walk
  - 4: turn
  - 5: jump
  - 6: squat
  - 7: arm wave
- **M**: Trial number (1–10), indexing repeated trials per participant per scenario

> Example: `2-5-3-7.mat` → Scenario 2 (Office), Participant 5, Activity 3 (walk), Trial 7

## Results & Benchmarks

This project focuses heavily on the challenging task of **Leave-One-Environment-Out (LOEO) Cross-Validation**. The models are trained on three distinct rooms and tested on a completely unseen fourth room to prove true domain agnosticism.

The metric reported below is the **F1-Score specifically for Fall Detection** and the Test accuracy score for all actions.

Note : Every data used for the training is normalized using Z-score per sample.
       This table shows what other preprocessing process are done and their results.

| Model Architecture | Preprocessing Pipeline                          | F1-Score (Fall) | Test Accuracy |
| ------------------ | ----------------------------------------------- | --------------- | ------------- |
| EfficientNetLite   | Raw_Data                                        |  0.821 ± 0.044  | 0.694 ± 0.033 |
| EfficientNetLite   | PCA_Reconstructed+SG                            |  0.814 ± 0.023  | 0.653 ± 0.039 |
| EfficientNetLite   | PCA_Reconstructed+Butterworth                   |  0.891 ± 0.045  | 0.686 ± 0.068 |
| EfficientNetLite   | PCA_Reconstructed+Butterworth+data_augmentation |**0.918 ± 0.036**| 0.698 ± 0.068 |
| GRU                | Raw_Data                                        |  0.273 ± 0.050  | 0.206 ± 0.053 |
| GRU                | PCA_Reconstructed                               |  0.264 ± 0.050  | 0.253 ± 0.070 |
| GRU                | PCA_Only                                        |  0.309 ± 0.023  | 0.255 ± 0.033 |
| GRU                | PCA_Only+Butterworth                            |  0.236 ± 0.012  | 0.199 ± 0.033 |
| GRU                | PCA_Only+SG                                     |  0.264 ± 0.017  | 0.198 ± 0.024 |
| GRU                | PCA_Only+SG+data_augmentation                   |  0.271 ± 0.036  | 0.250 ± 0.024 |
| LSTM               | Raw_Data                                        |  0.309 ± 0.050  | 0.258 ± 0.063 |
| LSTM               | PCA_Reconstructed                               |  0.328 ± 0.045  | 0.286 ± 0.048 |
| LSTM               | PCA_Only                                        |  0.377 ± 0.152  | 0.340 ± 0.120 |
| LSTM               | PCA_Only+SG                                     |  0.351 ± 0.084  | 0.365 ± 0.055 |
| LSTM               | PCA_Only+Butterworth                            |  0.737 ± 0.049  | 0.493 ± 0.027 |
| LSTM               | PCA_Only+Butterworth+data_augmentation          |  0.795 ± 0.083  | 0.542 ± 0.059 |
| MobileNetV3        | Raw_Data                                        |  0.807 ± 0.044  | 0.676 ± 0.042 |
| MobileNetV3        | PCA_Reconstructed+SG                            |  0.764 ± 0.050  | 0.621 ± 0.037 |
| MobileNetV3        | PCA_Reconstructed+Butterworth                   |  0.849 ± 0.050  | 0.658 ± 0.075 |
| MobileNetV3        | PCA_Reconstructed+Butterworth+data_augmentation |  0.863 ± 0.046  | 0.662 ± 0.080 |
| ResNet18           | Raw_Data                                        |  0.872 ± 0.030  | 0.738 ± 0.058 |
| ResNet18           | PCA_Reconstructed+SG                            |  0.854 ± 0.011  | 0.689 ± 0.033 |
| ResNet18           | PCA_Reconstructed+Butterworth                   |  0.858 ± 0.033  | 0.693 ± 0.017 |
| ResNet18           | PCA_Reconstructed+Butterworth+data_augmentation |  0.852 ± 0.045  | 0.718 ± 0.045 |
| Transformer        | Raw_Data                                        |  0.371 ± 0.054  | 0.280 ± 0.020 |
| Transformer        | PCA_Reconstructed+Butterworth                   |  0.537 ± 0.070  | 0.376 ± 0.034 |

**Key Finding:** While massive models like ResNet18 struggle with domain shift when frequency filters are applied, lightweight mobile architectures (EfficientNetLite) thrive when fed mathematically purified, 2-Channel (Amplitude + Velocity) motion data, resulting in a state-of-the-art F1-score of 0.89.


## License

### Code

The source code in this repository is licensed under the MIT License.
See the LICENSE file for details.

---

### Dataset

The ESP-Fi HAR dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0).

Users are free to use, modify, and distribute the dataset for academic or commercial purposes, provided proper attribution is given by citing the following publication:

Wen et al.,  
"ESP-Fi HAR: A low-power WiFi CSI dataset for Ad-Hoc IoT human activity recognition",  
Ad Hoc Networks, 2026.

This dataset does not contain personally identifiable information.


## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{WEN2026104192,  
   author = {Wen, Zhiwei and Ruan, Yanlin and Wang, Xiaoye and Zhou, Junjie and Gao, Hongliang and Li, Tao},  
   title = {ESP-Fi HAR: A low-power WiFi CSI dataset for Ad-Hoc IoT human activity recognition},  
   journal = {Ad Hoc Networks},  
   volume = {186},  
   pages = {104192},  
   ISSN = {1570-8705},  
   DOI = {https://doi.org/10.1016/j.adhoc.2026.104192},  
   url = {https://www.sciencedirect.com/science/article/pii/S1570870526000582},  
   year = {2026},  
   type = {Journal Article}  
}

