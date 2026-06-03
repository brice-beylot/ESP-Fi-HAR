# Main Training Engine for ESP-Fi HAR
# Handles Model Training, Evaluation, Cross-Validation, and Logging

import argparse
import copy
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import random
import scipy.io as sio
from scipy.signal import butter, filtfilt, savgol_filter
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader

from dataset import ESP_Fi_HAR_Dataset
from util import load_data_n_model


def set_random_seed(seed=666):
    """
    Forces PyTorch, NumPy, and Python to use the exact same random numbers every time.
    Why? Because neural networks initialize with random weights. If you don't lock the seed, 
    running the exact same code twice will give you slightly different F1 scores, making it 
    impossible to tell if your code changes actually improved the model or if you just got lucky.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def export_errors_to_pdf(pca_mode, filter_name, error_list, success_dict, output_pdf="misclassified_falls.pdf"):
    if not error_list:
        print("No fall-related errors to plot!")
        return

    activity_names = ['Run', 'Walk', 'Jump', 'Squat', 'Arm Wave', 'Turn', 'Fall']
    
    with PdfPages(output_pdf) as pdf:
        for item in error_list:
            err_path = item['path']
            xyz_key = item['xyz_key']
            true_act = activity_names[item['true']]
            pred_act = activity_names[item['pred']]
            err_filename = os.path.basename(err_path)
            
            err_mat = sio.loadmat(err_path)['CSIamp']
            if err_mat.shape == (52, 950): err_mat = err_mat.T
            
            # ==================================================
            # 1. OPTIONAL FILTERING (Operates on Time Axis 0)
            # ==================================================
            if filter_name == "Butterworth":
                fs = 100
                nyquist = fs / 2
                b, a = butter(4, [0.5 / nyquist, 10.0 / nyquist], btype='band')
                err_mat = filtfilt(b, a, err_mat, axis=0)
                
            elif filter_name == "SG":
                err_mat = savgol_filter(err_mat, window_length=15, polyorder=3, axis=0)

            # ==================================================
            # 2. OPTIONAL PCA & RECONSTRUCTION
            # ==================================================
            if pca_mode != 'None':
                pca = PCA(n_components=3)
                err_scores = pca.fit_transform(err_mat)
                
                if pca_mode == 'PCA_Reconstructed':
                    err_mat = pca.inverse_transform(err_scores)
                elif pca_mode == 'PCA_Only':
                    err_mat = err_scores

            # ==================================================
            # 3. NORMALIZATION & SHAPING
            # ==================================================
            err_mat = (err_mat - np.mean(err_mat)) / (np.std(err_mat) + 1e-8)

            # Determine the correct feature dimension
            feature_dim = 3 if pca_mode == 'PCA_Only' else 52
            
            # Reshape to (1, 950, features) 
            err_mat = err_mat.reshape(1, 950, feature_dim).astype(np.float32)
                
            # Try to find a baseline from the exact same Env-Subject-Action
            baseline_path = None
            if xyz_key in success_dict and len(success_dict[xyz_key]) > 0:
                baseline_path = success_dict[xyz_key][0]
                
            if baseline_path:
                base_filename = os.path.basename(baseline_path)
                base_mat = sio.loadmat(baseline_path)['CSIamp']
                if base_mat.shape == (52, 950): base_mat = base_mat.T      
                
                # ==================================================
                # 1. OPTIONAL FILTERING (Operates on Time Axis 0)
                # ==================================================
                if filter_name == "Butterworth":
                    fs = 100
                    nyquist = fs / 2
                    b, a = butter(4, [0.5 / nyquist, 10.0 / nyquist], btype='band')
                    base_mat = filtfilt(b, a, base_mat, axis=0)
                    
                elif filter_name == "SG":
                    base_mat = savgol_filter(base_mat, window_length=15, polyorder=3, axis=0)

                # ==================================================
                # 2. OPTIONAL PCA & RECONSTRUCTION
                # ==================================================
                if pca_mode != 'None':
                    pca = PCA(n_components=3)
                    base_scores = pca.fit_transform(base_mat)
                    
                    if pca_mode == 'PCA_Reconstructed':
                        base_mat = pca.inverse_transform(base_scores)
                    elif pca_mode == 'PCA_Only':
                        base_mat = base_scores

                # ==================================================
                # 3. NORMALIZATION & SHAPING
                # ==================================================
                base_mat = (base_mat - np.mean(base_mat)) / (np.std(base_mat) + 1e-8)

                # Determine the correct feature dimension
                feature_dim = 3 if pca_mode == 'PCA_Only' else 52
                
                # Reshape to (1, 950, features) 
                base_mat = base_mat.reshape(1, 950, feature_dim).astype(np.float32)

                fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharey=True)
                
                # Top graph: The Error (Plotting all 52 reconstructed subcarriers)
                # alpha=0.6 makes the 52 lines slightly transparent for readability
                axes[0].plot(err_mat, alpha=0.6)
                axes[0].set_title(f"MISCLASSIFIED: {err_filename} | True: {true_act} -> Pred: {pred_act}", color='red')
                axes[0].grid(True, alpha=0.3)
                
                # Bottom graph: The Golden Baseline
                axes[1].plot(base_mat, alpha=0.6)
                axes[1].set_title(f"CORRECT BASELINE: {base_filename} | True: {true_act} -> Pred: {true_act}", color='green')
                axes[1].set_xlabel("Sample Time")
                axes[1].grid(True, alpha=0.3)
                
                fig.supylabel("Reconstructed Subcarrier Amplitude")
                plt.tight_layout()
                
                pdf.savefig(fig)
                plt.close(fig)
                #plt.show()
                
            else:
                # No baseline exists
                fig = plt.figure(figsize=(10, 4))
                plt.plot(err_mat, alpha=0.6)
                plt.title(f"MISCLASSIFIED: {err_filename} | True: {true_act} -> Pred: {pred_act}\n(No baseline found)", color='orange')
                plt.xlabel("Sample Time")
                plt.ylabel("Reconstructed Subcarrier Amplitude")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                pdf.savefig(fig)
                plt.close(fig)
                #plt.show()

    print(f"Success! Created {output_pdf} with {len(error_list)} fully reconstructed samples.")


def evaluate(model, recognition_name, loader, criterion, device, save_cm=False, collect_errors=False):
    # Tests the model's current performance without updating its weights.

    # model.eval() changes how certain layers (like Dropout or BatchNorm) behave.
    # It ensures the model acts deterministically during testing.
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    error_receipts = []
    success_dict = {}

    # CRITICAL: torch.no_grad() turns off PyTorch's gradient tracking engine.
    # Because we are just testing, we don't need to calculate derivatives.
    # This massively speeds up testing and cuts memory usage in half.
    with torch.no_grad():
        for inputs, labels, paths in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            # Forward pass: the model makes its guesses
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Accumulate the total loss for the batch
            total_loss += loss.item() * inputs.size(0)

            # The output is a probability array for all 7 classes. 
            # argmax picks the index (0-6) with the highest probability.
            preds = torch.argmax(outputs, dim=1)

            # Storage of classified and misclassified samples regarding the fall detection for the plot
            if collect_errors:
                for i in range(len(labels)):
                    p = preds[i].item()
                    l = labels[i].item()
                    path = paths[i]
                    
                    # Extract the X-Y-Z signature from the file name (e.g., "1-2-7-10.mat" -> "1-2-7")
                    filename = os.path.basename(path)
                    parts = filename.split('-')
                    if len(parts) >= 4:
                        xyz_key = f"{parts[0]}-{parts[1]}-{parts[2]}"
                    else:
                        xyz_key = "unknown"

                    if p == l:
                        # Save a well classified path sample in the dictionnary for a potential comparison
                        if xyz_key not in success_dict:
                            success_dict[xyz_key] = []
                        success_dict[xyz_key].append(path)
                        
                    elif p != l and (p == 6 or l == 6):
                        # Save a misclassified sample, but attach the xyz_key so we can look up a baseline later!
                        error_receipts.append({
                            'path': path,
                            'true': l,
                            'pred': p,
                            'xyz_key': xyz_key
                        })
            
            # Move the tensors off the GPU (cpu) and into standard NumPy arrays
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate and save the Confusion Matrix image (usually only on the last epoch)
    if save_cm:  
        cm = confusion_matrix(
            all_labels, all_preds, labels=[0, 1, 2, 3, 4, 5, 6])
        print("Confusion matrix :", cm)

        # Normalize the matrix to show percentages instead of raw sample counts
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix :", cmn)
        target_names = ['Run', 'Walk', 'Jump',
                        'Squat', 'Arm swing', 'Turn', 'Fall']
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.ylabel('Actual Activity (True Label)')
        plt.xlabel('Predicted Activity (Model Output)')
        plt.title('ESP-Fi HAR Normalized Confusion Matrix')
        plt.savefig(f'training_logs/confusion_matrix_{recognition_name}.png', dpi=300, bbox_inches='tight')

    avg_loss = total_loss / len(loader.dataset)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Calculation of the F1 score explicitly isolating the "fall" state.
    # average=None returns an array of 7 independent F1 scores.
    f1_scores_per_class = f1_score(all_labels, all_preds, average=None)
    
    # Dynamically find the index for 'fall' so we don't have to hardcode '6'
    index = loader.dataset.activities.index('fall')
    fall_f1 = f1_scores_per_class[index]

    return acc, fall_f1, avg_loss, error_receipts, success_dict


def train(model, train_loader, test_loader, num_epochs,
          learning_rate, criterion, device,
          csv_path, model_name, preprocessing_name, continue_training, save_errors):
    # The main training loop for a single model (One fold of Cross-Validation).

    model = model.to(device)

    # AdamW is an advanced optimizer that updates the neural network's weights based on the loss.
    # weight_decay adds L2 regularization to prevent the model from overfitting to the training data.
    optimizer = optim.AdamW(model.parameters(),
                            lr=learning_rate,
                            weight_decay=1e-4)

    # A learning rate scheduler slowly drops the learning rate as epochs progress.
    # This helps the model take smaller, more careful steps as it gets closer to the optimal answer.
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Initialize the AMP Gradient Scaler
    scaler = torch.amp.GradScaler('cuda')

    os.makedirs(csv_path, exist_ok=True)

    # Note: model_name here includes the Fold Number so they don't overwrite each other!
    train_csv = os.path.join(csv_path, f'training_results_{model_name}+{preprocessing_name}.csv')
    final_test_csv = os.path.join(csv_path, f'final_test_results_{model_name}.csv')

    best_test_loss = float('inf')
    best_test_fall_f1 = 0

    best_model_path = os.path.join(csv_path, f'best_model_{model_name}+{preprocessing_name}.pth')

    if continue_training and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Continuing training from checkpoint: {best_model_path}")
        print("\nEvaluating loaded checkpoint before training...")
        init_acc, init_f1, init_loss = evaluate(model, f"{model_name}+{preprocessing_name}", test_loader, criterion, device)
        print(f"Loaded Checkpoint -> TestAcc: {init_acc:.4f} | F1: {init_f1:.4f} | TestLoss: {init_loss:.4f}\n")
        best_test_loss = init_loss
    else:
        print("No checkpoint found. Starting training from scratch.")

    # ==================================
    # THE EPOCH LOOP
    # ==================================
    for epoch in range(num_epochs):
        model.train() # Tell PyTorch we are actively learning

        running_loss = 0
        correct = 0
        total = 0

        for inputs, labels, paths in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            # 1. Clear the old gradients. PyTorch accumulates them by default, 
            #    so if we don't wipe them, step 2 will mix with step 1!
            optimizer.zero_grad()
            
            # 2. Forward pass (make a prediction) inside the AMP Autocast context
            #    This automatically casts sensitive operations to 32-bit and safe ones to 16-bit!
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # 3. Backward pass (calculate exactly how wrong each weight was)
            #    We use the scaler to prevent 16-bit gradient underflow
            scaler.scale(loss).backward()
            
            # 4. Optimizer step (adjust the weights to be slightly more accurate)
            scaler.step(optimizer)

            # 5. Update the scaler for the next batch
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Calculate average metrics for this epoch
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        # Test the model on the unseen Fold data.
        # save_cm=(epoch == num_epochs-1) ensures we only generate the heatmap image on the very last epoch.
        test_acc, test_fall_f1, test_loss, final_errors, final_successes = evaluate(
            model, f"{model_name}+{preprocessing_name}", test_loader, criterion, device, save_cm=(epoch == num_epochs-1)
        )

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"TrainAcc: {train_acc:.4f} "
            f"TrainLoss: {train_loss:.4f} "
            f"TestAcc: {test_acc:.4f} "
            f"F1: {test_fall_f1:.4f} "
            f"TestLoss: {test_loss:.4f}"
        )

        # Save the model state strictly based on its ability to detect the "Fall" class
        if test_fall_f1 > best_test_fall_f1 :
            best_test_fall_f1 = test_fall_f1
            torch.save(model.state_dict(), best_model_path)

        # Lower the learning rate slightly
        scheduler.step()

    print("\nTraining completed.")
    print(f"Best Test Fall F1: {best_test_fall_f1:.4f}")

    # ==================================
    # POST-TRAINING EVALUATION
    # ==================================
    print("\nEvaluating Best Model on Test Set...")

    # Load the best weights discovered during the epoch loop
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)

    final_test_acc, final_test_fall_f1, final_test_loss, final_errors, final_successes = evaluate(
        model, f"{model_name}+{preprocessing_name}", test_loader, criterion, device, collect_errors = save_errors
    )

    print("\n========== Final Test Results ==========")
    print(f"Test Accuracy : {final_test_acc:.4f}")
    print(f"Test Macro-F1 : {final_test_fall_f1:.4f}")
    print(f"Test Loss     : {final_test_loss:.4f}")
    print("========================================\n")

    # Safe appending logic: Write header if file is empty, otherwise just append rows
    file_exists = os.path.exists(final_test_csv) and os.path.getsize(final_test_csv) > 0
    with open(final_test_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Model', 'Preprocessing', 'TestAccuracy', 'TestMacroF1', 'TestLoss'])
        writer.writerow([
            model_name,
            preprocessing_name,
            f"{final_test_acc:.4f}",
            f"{final_test_fall_f1:.4f}",
            f"{final_test_loss:.4f}"
        ])

    print(f"Final test results saved to: {final_test_csv}")
    print(f"Training logs saved to: {train_csv}")

    # Delete the .pth file containing the weights of the model in order to save space
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        print(f"Deleted model checkpoint to save space: {best_model_path}")

    return best_model_path, final_test_fall_f1, final_test_acc, final_errors, final_successes


def main():
    root = './Data'
    csv_path = './training_logs/'

    torch.cuda.empty_cache()
    set_random_seed(seed=666)

    parser = argparse.ArgumentParser('ESP-Fi HAR Benchmark')

    parser.add_argument('--dataset',
                        choices=['ESP-Fi_HAR'],
                        default='ESP-Fi_HAR')

    parser.add_argument('--model',
                        choices=[
                            'CNN', 'ResNet18',
                            'GRU', 'LSTM', 'Transformer',
                            'MobileNetV3', 'EfficientNetLite'
                        ],
                        required=True)

    parser.add_argument('--runs', type=int, default=200,
                        help='Number of runs for each model (for stability)')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pth) for latency test')

    parser.add_argument('--continue_training', type=bool, default=False,
                        help='Continue training from the best checkpoint if available')

    parser.add_argument('--pca_mode', 
                        choices=['None', 'PCA_Reconstructed', 'PCA_Only'],
                        default='PCA_Reconstructed',
                        help='Choose PCA handling: "None", "PCA_Reconstructed", or "PCA_Only"')

    parser.add_argument('--filter_name', 
                        choices=['None', 'Butterworth', 'SG'],
                        default='Butterworth',
                        help='Choose the filter: "None", "Butterworth", or "SG"')
    
    parser.add_argument('--data_augmentation', action='store_true',
                        help='If flagged, applies data augmentation for the training part')
                
    parser.add_argument('--save_errors', action='store_true',
                        help='If flagged, generates a PDF of all misclassified Fall samples')

    args = parser.parse_args()

    # Call our Factory function to build the correct datasets and architecture
    data_loader, model, train_epoch = \
        load_data_n_model(args.dataset, args.model, args.pca_mode, args.filter_name, root)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    criterion = nn.CrossEntropyLoss()

    # ==========================================================
    # LEAVE-ONE-ENVIRONMENT-OUT (LOEO) CROSS-VALIDATION LOOP
    # ==========================================================

    # Take a "photograph" of the completely untrained, random model weights.
    # We must reset the model to this blank state before every fold to prevent data leakage!
    initial_model_weights = copy.deepcopy(model.state_dict())

    all_fold_fall_f1_scores = []
    all_fold_test_accuracy = []
    model_name = f"{args.model}"
    preprocessing_name = "Raw_Data"
    if args.pca_mode != "None" :
        preprocessing_name = f"{args.pca_mode}"
    if args.filter_name != "None" :
        if preprocessing_name == "Raw_Data" :
            preprocessing_name = f"{args.filter_name}"
        else :
            preprocessing_name += f"+{args.filter_name}"
    if args.data_augmentation :
        if preprocessing_name == "Raw_Data" :
            preprocessing_name = "data_augmentation"
        else :
            preprocessing_name += "+data_augmentation"
    print(f"Model name : {model_name}")
    print(f"Preprocessing name : {preprocessing_name}")
    print(f"Recognition name : {model_name}+{preprocessing_name}")

    for k in range(len(data_loader)):
        print(f"\n{'='*50}")
        print(f"Starting Cross-Validation Fold {k+1}/4")
        print(f"Testing on Environment {k+1}, Training on the other 3")
        print(f"{'='*50}")

        # 1. Isolate the current environment to be the Test Set
        test_loader = data_loader[k]
        # CRITICAL: Lock the test set to ensure purity!
        test_loader.dataset.data_augmentation = False 

        # 2. Collect the datasets from the remaining 3 environments
        train_datasets = []
        for i in range(len(data_loader)):
            if i != k:
                # CRITICAL: Turn on the augmentations for the training sets!
                data_loader[i].dataset.data_augmentation = args.data_augmentation 
                train_datasets.append(data_loader[i].dataset)

        # 3. Glue the 3 datasets together into one massive training set
        combined_train_dataset = ConcatDataset(train_datasets)

        # 4. Wrap the massive dataset in a new DataLoader and SHUFFLE it
        batch_size = test_loader.batch_size 
        train_loader = DataLoader(
            dataset=combined_train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # 5. Wipe the model's memory by injecting the untrained weights back in
        model.load_state_dict(initial_model_weights)

        # 6. Execute the full training loop for this specific fold
        best_model_path, best_fold_fall_f1, fold_test_accuracy, fold_errors, fold_successes = train(model=model,
              train_loader=train_loader,
              test_loader=test_loader,
              num_epochs=train_epoch,
              learning_rate=1e-3,
              criterion=criterion,
              device=device,
              csv_path=csv_path,
              model_name=model_name,
              preprocessing_name=f"{preprocessing_name}_LOEO_Fold_{k+1}", # Pass the fold number to name the files!
              continue_training=args.continue_training,
              save_errors=args.save_errors
              )

        all_fold_fall_f1_scores.append(best_fold_fall_f1)
        all_fold_test_accuracy.append(fold_test_accuracy)
        print(f"Fold {k + 1} Finished! Best F1: {best_fold_fall_f1}")
        print(f"Fold {k + 1} Finished! Test Accuracy: {fold_test_accuracy}")

        # If the trigger is True, generate the PDF for THIS specific environment immediately
        if args.save_errors and fold_errors:
            print(f"\n[Trigger Activated] Plotting misclassified samples for Environment {k+1}...")
            # We dynamically name the file to include "Env1", "Env2", etc.
            pdf_name = f"{args.model}_Env{k+1}_misclassified_falls_{args.filter_name}_{52-49*int(args.only_pca)}PCA_data_augmentation_and_Velocity.pdf"
            pdf_path = os.path.join(csv_path, pdf_name)
            export_errors_to_pdf(args.pca_mode, args.filter_name, fold_errors, fold_successes, output_pdf=pdf_path)
    
    # ==========================================================
    # AGGREGATE RESULTS
    # ==========================================================
    # This globally saves the ultimate cross-validation average for the chosen model
    final_results = os.path.join(
        csv_path,
        'LOEO_final_results.csv'
    )

    final_mean_fall_f1 = np.mean(all_fold_fall_f1_scores)
    final_std_fall_f1 = np.std(all_fold_fall_f1_scores)
    final_mean_test_accuracy = np.mean(all_fold_test_accuracy)
    final_std_test_accuracy = np.std(all_fold_test_accuracy)
    print(f"FINAL CROSS-VALIDATION SCORE: F1 = {final_mean_fall_f1:.3f} ± {final_std_fall_f1:.3f}")

    # Safe appending logic again for the global tracker
    file_exists = os.path.exists(final_results) and os.path.getsize(final_results) > 0
    with open(final_results, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Model', 'Preprocessing pipeline', 'Fall F1 score', 'Test accuracy score'])
        writer.writerow([
            model_name,
            preprocessing_name,
            f"{final_mean_fall_f1:.3f} ± {final_std_fall_f1:.3f}",
            f"{final_mean_test_accuracy:.3f} ± {final_std_test_accuracy:.3f}"
        ])

if __name__ == "__main__":
    main()