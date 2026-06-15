import os
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from src.features import load_and_normalize_audio, extract_tabular_features, extract_mel_spectrogram
from src.models import AudioCNN

def load_models(lgb_path, cnn_path, device):
    """
    Loads the trained LightGBM and PyTorch CNN models.
    """
    # 1. Load LightGBM
    if not os.path.exists(lgb_path):
        raise FileNotFoundError(f"LightGBM model file not found at {lgb_path}. Please run train_pipeline.py first.")
    with open(lgb_path, 'rb') as f:
        lgb_model = pickle.load(f)
        
    # 2. Load PyTorch CNN
    if not os.path.exists(cnn_path):
        raise FileNotFoundError(f"CNN model weight file not found at {cnn_path}. Please run train_pipeline.py first.")
    cnn_model = AudioCNN()
    cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn_model.to(device)
    cnn_model.eval()
    
    return lgb_model, cnn_model

def predict_single(file_path, lgb_model, cnn_model, device):
    """
    Performs ensemble inference on a single audio file.
    Returns a dictionary with predictions, confidence, and model-specific probabilities.
    """
    # Load and normalize audio
    y, sr = load_and_normalize_audio(file_path)
    
    # Extract features
    tab_feat = extract_tabular_features(y, sr)
    mel_spec = extract_mel_spectrogram(y, sr)
    
    # LightGBM inference
    # Reshape features to shape (1, num_features)
    lgb_feat = tab_feat.reshape(1, -1)
    lgb_prob = lgb_model.predict_proba(lgb_feat)[0, 1]
    
    # PyTorch CNN inference
    # Shape: (1, 1, 128, 63)
    cnn_feat = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = cnn_model(cnn_feat)
        cnn_prob = torch.sigmoid(logits).cpu().numpy().item()
        
    # Ensemble probability (soft average of both models)
    ensemble_prob = (lgb_prob + cnn_prob) / 2.0
    
    # Classify based on 0.5 threshold
    # 0 = Genuine (Human), 1 = Deepfake (AI-Generated)
    if ensemble_prob >= 0.5:
        prediction = "Deepfake"
        confidence = ensemble_prob
    else:
        prediction = "Genuine"
        confidence = 1.0 - ensemble_prob
        
    return {
        "file_path": file_path,
        "prediction": prediction,
        "confidence": float(confidence),
        "ensemble_probability_fake": float(ensemble_prob),
        "lightgbm_probability_fake": float(lgb_prob),
        "cnn_probability_fake": float(cnn_prob)
    }

def main():
    parser = argparse.ArgumentParser(description="Deepfake Audio Detection Inference Script")
    parser.add_argument('--audio_path', type=str, help='Path to single audio WAV file')
    parser.add_argument('--csv_path', type=str, help='Path to input CSV containing audio file paths')
    parser.add_argument('--path_column', type=str, default='file_path', help='Column name containing audio paths in the input CSV')
    parser.add_argument('--output_path', type=str, help='Path to save predictions (JSON for single file, CSV for batch)')
    parser.add_argument('--lgb_model_path', type=str, default='models/lightgbm_model.pkl', help='Path to LightGBM model pickle')
    parser.add_argument('--cnn_model_path', type=str, default='models/cnn_model.pth', help='Path to CNN model weights')
    args = parser.parse_args()
    
    if not args.audio_path and not args.csv_path:
        parser.print_help()
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    try:
        lgb_model, cnn_model = load_models(args.lgb_model_path, args.cnn_model_path, device)
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return
        
    # Single file prediction
    if args.audio_path:
        if not os.path.exists(args.audio_path):
            print(f"Error: Audio file not found at {args.audio_path}")
            return
            
        result = predict_single(args.audio_path, lgb_model, cnn_model, device)
        
        # Output result
        print(json.dumps(result, indent=4))
        
        # Save JSON if requested
        if args.output_path:
            with open(args.output_path, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"Saved result JSON to {args.output_path}")
            
    # Batch prediction on CSV
    elif args.csv_path:
        if not os.path.exists(args.csv_path):
            print(f"Error: CSV file not found at {args.csv_path}")
            return
            
        df = pd.read_csv(args.csv_path)
        if args.path_column not in df.columns:
            print(f"Error: Column '{args.path_column}' not found in CSV. Available columns: {list(df.columns)}")
            return
            
        print(f"Found {len(df)} files to process in CSV.")
        
        predictions = []
        confidences = []
        ens_probs = []
        lgb_probs = []
        cnn_probs = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing audio list"):
            file_path = row[args.path_column]
            
            # Resolve relative paths relative to CSV file location if needed
            if not os.path.isabs(file_path):
                csv_dir = os.path.dirname(args.csv_path)
                resolved_path = os.path.join(csv_dir, file_path)
                if os.path.exists(resolved_path):
                    file_path = resolved_path
                    
            if not os.path.exists(file_path):
                predictions.append("Error (File Not Found)")
                confidences.append(0.0)
                ens_probs.append(-1.0)
                lgb_probs.append(-1.0)
                cnn_probs.append(-1.0)
                continue
                
            try:
                res = predict_single(file_path, lgb_model, cnn_model, device)
                predictions.append(res['prediction'])
                confidences.append(res['confidence'])
                ens_probs.append(res['ensemble_probability_fake'])
                lgb_probs.append(res['lightgbm_probability_fake'])
                cnn_probs.append(res['cnn_probability_fake'])
            except Exception as e:
                predictions.append(f"Error ({str(e)})")
                confidences.append(0.0)
                ens_probs.append(-1.0)
                lgb_probs.append(-1.0)
                cnn_probs.append(-1.0)
                
        # Add columns to dataframe
        df['predicted_label'] = predictions
        df['confidence'] = confidences
        df['ensemble_prob_fake'] = ens_probs
        df['lgb_prob_fake'] = lgb_probs
        df['cnn_prob_fake'] = cnn_probs
        
        # Save results
        out_path = args.output_path if args.output_path else args.csv_path.replace('.csv', '_predictions.csv')
        df.to_csv(out_path, index=False)
        print(f"Saved batch prediction results to {out_path}")

if __name__ == '__main__':
    main()
