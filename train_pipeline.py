import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lightgbm import LGBMClassifier

from src.features import load_and_normalize_audio, extract_tabular_features, extract_mel_spectrogram
from src.models import AudioCNN, InMemoryDataset, compute_eer

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def process_file(file_path, label):
    """
    Worker function to process a single audio file and extract both tabular and Mel features.
    """
    try:
        y, sr = load_and_normalize_audio(file_path)
        tab_feat = extract_tabular_features(y, sr)
        mel_spec = extract_mel_spectrogram(y, sr)
        return tab_feat, mel_spec, label, file_path, True
    except Exception as e:
        return None, None, label, file_path, False

def load_dataset_parallel(dataset_dir):
    """
    Scans real and fake directories, extracts features in parallel, and loads into memory.
    """
    cache_file = os.path.join('models', 'feature_cache.pkl')
    if os.path.exists(cache_file):
        print("Loading pre-extracted features from cache...")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Cache load failed ({e}), re-extracting...")

    real_dir = os.path.join(dataset_dir, 'real')
    fake_dir = os.path.join(dataset_dir, 'fake')
    
    tasks = []
    
    print("Scanning dataset directories...")
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        raise ValueError(f"Dataset subdirectories 'real' and/or 'fake' not found in {dataset_dir}")
        
    for f in os.listdir(real_dir):
        if f.endswith('.wav'):
            tasks.append((os.path.join(real_dir, f), 0)) # 0 = Genuine
            
    for f in os.listdir(fake_dir):
        if f.endswith('.wav'):
            tasks.append((os.path.join(fake_dir, f), 1)) # 1 = Deepfake
            
    print(f"Found {len(tasks)} audio files total. Starting parallel feature extraction...")
    
    tab_features_list = []
    mel_spectrograms_list = []
    labels_list = []
    paths_list = []
    
    # Process in parallel using a process pool
    max_workers = min(os.cpu_count(), 8)
    print(f"Using {max_workers} CPU workers for parallel feature extraction.")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, path, label) for path, label in tasks]
        
        # Wrap in tqdm to show progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing audio"):
            tab_feat, mel_spec, label, file_path, success = future.result()
            if success:
                tab_features_list.append(tab_feat)
                mel_spectrograms_list.append(mel_spec)
                labels_list.append(label)
                paths_list.append(file_path)
            else:
                print(f"Failed to process file: {file_path}")
                
    res = (np.array(tab_features_list), np.array(mel_spectrograms_list), np.array(labels_list), paths_list)
    try:
        os.makedirs('models', exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(res, f)
        print(f"Saved feature cache to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save feature cache: {e}")
        
    return res

def plot_confusion_matrix(cm, classes, title, save_path):
    """
    Plots and saves a confusion matrix.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, lgb_probs, cnn_probs, ensemble_probs, save_path):
    """
    Plots and saves ROC curves for the models.
    """
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(8, 6))
    
    for probs, name in zip([lgb_probs, cnn_probs, ensemble_probs], ['LightGBM', 'CNN', 'Ensemble']):
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_cnn(model, train_loader, val_loader, epochs, lr, device, model_save_path):
    """
    Trains the PyTorch 2D CNN model and saves the best weight checkpoint.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    
    print("\nStarting PyTorch 2D CNN Training...")
    
    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_acc = correct_train / total_train
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct_val += (preds == y_batch).sum().item()
                total_val += y_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct_val / total_val
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Val Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model checkpoint to {model_save_path}")
            
    # Load best checkpoint before returning
    model.load_state_dict(torch.load(model_save_path))
    return model

def main():
    parser = argparse.ArgumentParser(description="Deepfake Audio Detection Training Pipeline")
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Path to dataset directory containing real and fake folders')
    parser.add_argument('--epochs', type=int, default=12, help='Number of epochs for CNN training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for CNN training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for CNN training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Create save folders
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # 1. Load data and extract features in parallel
    X_tab, X_mel, y, file_paths = load_dataset_parallel(args.dataset_dir)
    print(f"\nLoaded successfully: {len(y)} samples")
    print(f"Tabular features shape: {X_tab.shape}")
    print(f"Mel Spectrogram shape: {X_mel.shape}")
    
    # 2. Split dataset into Train (70%), Val (15%), Test (15%) splits
    # Stratify by labels to ensure perfect class balance in all sets
    indices = np.arange(len(y))
    idx_train, idx_temp = train_test_split(indices, test_size=0.30, random_state=args.seed, stratify=y)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=args.seed, stratify=y[idx_temp])
    
    print(f"\nData splits:")
    print(f"Train samples: {len(idx_train)} (Real: {np.sum(y[idx_train] == 0)}, Fake: {np.sum(y[idx_train] == 1)})")
    print(f"Val samples:   {len(idx_val)} (Real: {np.sum(y[idx_val] == 0)}, Fake: {np.sum(y[idx_val] == 1)})")
    print(f"Test samples:  {len(idx_test)} (Real: {np.sum(y[idx_test] == 0)}, Fake: {np.sum(y[idx_test] == 1)})")
    
    # Tabular splits
    X_train_tab, y_train = X_tab[idx_train], y[idx_train]
    X_val_tab, y_val = X_tab[idx_val], y[idx_val]
    X_test_tab, y_test = X_tab[idx_test], y[idx_test]
    
    # Mel Spectrogram splits
    X_train_mel = X_mel[idx_train]
    X_val_mel = X_mel[idx_val]
    X_test_mel = X_mel[idx_test]
    
    # 3. Train LightGBM Model
    print("\nTraining LightGBM Classifier...")
    lgb_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=args.seed,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_train_tab, y_train,
        eval_set=[(X_val_tab, y_val)]
    )
    
    # Save LightGBM model
    lgb_save_path = os.path.join('models', 'lightgbm_model.pkl')
    with open(lgb_save_path, 'wb') as f:
        pickle.dump(lgb_model, f)
    print(f"Saved trained LightGBM model to {lgb_save_path}")
    
    # 4. Train PyTorch CNN Model
    train_dataset = InMemoryDataset(X_train_mel, y_train)
    val_dataset = InMemoryDataset(X_val_mel, y_val)
    test_dataset = InMemoryDataset(X_test_mel, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device} for CNN training.")
    
    cnn_model = AudioCNN().to(device)
    cnn_save_path = os.path.join('models', 'cnn_model.pth')
    
    cnn_model = train_cnn(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        model_save_path=cnn_save_path
    )
    
    # 5. Evaluate on Test Split
    print("\nEvaluating models on Test Split...")
    
    # LightGBM Predictions
    lgb_probs = lgb_model.predict_proba(X_test_tab)[:, 1]
    lgb_preds = (lgb_probs >= 0.5).astype(int)
    
    # CNN Predictions
    cnn_model.eval()
    cnn_probs = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            logits = cnn_model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            if probs.ndim == 0:  # single element check
                probs = np.array([probs])
            cnn_probs.extend(probs)
    cnn_probs = np.array(cnn_probs)
    cnn_preds = (cnn_probs >= 0.5).astype(int)
    
    # Ensemble Predictions (Soft Voting / Average Probability)
    ensemble_probs = (lgb_probs + cnn_probs) / 2.0
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    
    # Compute metrics for all three settings
    results = {}
    for name, preds, probs in [('LightGBM', lgb_preds, lgb_probs), 
                                ('CNN', cnn_preds, cnn_probs), 
                                ('Ensemble', ensemble_preds, ensemble_probs)]:
        acc = accuracy_score(y_test, preds)
        eer = compute_eer(y_test, probs)
        f1 = f1_score(y_test, preds)
        
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        
        # Per-class recall (recall for genuine and fake)
        rec_genuine = tn / (tn + fp)  # Recall for label 0 (Real)
        rec_fake = tp / (tp + fn)     # Recall for label 1 (Fake)
        
        results[name] = {
            'Accuracy': acc,
            'EER': eer,
            'F1-Score': f1,
            'Recall (Genuine)': rec_genuine,
            'Recall (Deepfake)': rec_fake,
            'Confusion Matrix': cm
        }
        
    # Print Ablation / Comparison Table
    print("\n" + "="*80)
    print("                        MODEL ABLATION STUDY RESULTS")
    print("="*80)
    df_metrics = pd.DataFrame({
        model_name: {
            'Accuracy (%)': metrics['Accuracy'] * 100,
            'EER (%)': metrics['EER'] * 100,
            'F1-Score (%)': metrics['F1-Score'] * 100,
            'Recall Genuine (%)': metrics['Recall (Genuine)'] * 100,
            'Recall Deepfake (%)': metrics['Recall (Deepfake)'] * 100,
        } for model_name, metrics in results.items()
    }).T
    print(df_metrics.to_string(formatters={
        'Accuracy (%)': '{:,.2f}%'.format,
        'EER (%)': '{:,.2f}%'.format,
        'F1-Score (%)': '{:,.2f}%'.format,
        'Recall Genuine (%)': '{:,.2f}%'.format,
        'Recall Deepfake (%)': '{:,.2f}%'.format,
    }))
    print("="*80)
    
    # Verify target thresholds for validation verification
    ens_acc = results['Ensemble']['Accuracy']
    ens_eer = results['Ensemble']['EER']
    ens_f1 = results['Ensemble']['F1-Score']
    ens_rec_g = results['Ensemble']['Recall (Genuine)']
    ens_rec_f = results['Ensemble']['Recall (Deepfake)']
    
    print("\nEnsemble Target Threshold Check:")
    print(f"Overall Accuracy:  {ens_acc*100:.2f}% (Target: >= 80%)  - {'PASSED' if ens_acc >= 0.80 else 'FAILED'}")
    print(f"Equal Error Rate:  {ens_eer*100:.2f}% (Target: <= 12%)  - {'PASSED' if ens_eer <= 0.12 else 'FAILED'}")
    print(f"F1 Score:          {ens_f1*100:.2f}% (Target: >= 80%)  - {'PASSED' if ens_f1 >= 0.80 else 'FAILED'}")
    print(f"Recall Genuine:    {ens_rec_g*100:.2f}% (Target: >= 75%)  - {'PASSED' if ens_rec_g >= 0.75 else 'FAILED'}")
    print(f"Recall Deepfake:   {ens_rec_f*100:.2f}% (Target: >= 75%)  - {'PASSED' if ens_rec_f >= 0.75 else 'FAILED'}")
    
    # Save plots
    print("\nGenerating and saving validation plots...")
    plot_roc_curve(
        y_true=y_test,
        lgb_probs=lgb_probs,
        cnn_probs=cnn_probs,
        ensemble_probs=ensemble_probs,
        save_path=os.path.join('plots', 'roc_curves.png')
    )
    plot_confusion_matrix(
        cm=results['Ensemble']['Confusion Matrix'],
        classes=['Genuine', 'Deepfake'],
        title='Ensemble Confusion Matrix',
        save_path=os.path.join('plots', 'confusion_matrix.png')
    )
    print("Saved ROC curves to plots/roc_curves.png")
    print("Saved Confusion Matrix to plots/confusion_matrix.png")
    print("\nTraining Pipeline finished successfully!")

if __name__ == '__main__':
    main()
