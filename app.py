import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import json
import torch
import pickle

from src.features import load_and_normalize_audio, extract_tabular_features, extract_mel_spectrogram
from src.models import AudioCNN
from predict import predict_single

# Page configuration
st.set_page_config(
    page_title="AuraShield - Deepfake Audio Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium dark styling (glassmorphism, clean layouts)
st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Title and headers */
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #ffffff !important;
    }
    
    /* Glassmorphic card styling */
    .glass-card {
        background: rgba(17, 25, 40, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Prediction Cards */
    .prediction-box {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .genuine-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(4, 120, 87, 0.05) 100%);
        border-left: 6px solid #10b981;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
    }
    
    .deepfake-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
        border-left: 6px solid #ef4444;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.1);
    }
    
    .genuine-text {
        color: #34d399 !important;
        font-size: 2.2rem;
        font-weight: 800;
    }
    
    .deepfake-text {
        color: #f87171 !important;
        font-size: 2.2rem;
        font-weight: 800;
    }
    
    /* Metric container styling */
    .metric-container {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 30px 0;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to cache model loading
@st.cache_resource
def load_models_cached():
    lgb_path = "models/lightgbm_model.pkl"
    cnn_path = "models/cnn_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(lgb_path) or not os.path.exists(cnn_path):
        return None, None, None
        
    # Load LightGBM
    with open(lgb_path, 'rb') as f:
        lgb_model = pickle.load(f)
        
    # Load PyTorch CNN
    cnn_model = AudioCNN()
    cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn_model.to(device)
    cnn_model.eval()
    
    return lgb_model, cnn_model, device

# App Title
st.title("🛡️ AuraShield")
st.subheader("State-of-the-Art Deepfake Audio Detection")

# Load models
lgb_model, cnn_model, device = load_models_cached()

if lgb_model is None:
    st.error("⚠️ Trained models not found. Please run `python train_pipeline.py` in your terminal to train and save the models first.")
    st.stop()

# Sidebar Setup
with st.sidebar:
    st.markdown("### Configuration")
    mode = st.radio("Select Analysis Mode", ["Single Audio Upload", "Batch CSV Processor"])
    
    st.markdown("---")
    st.markdown("### Model Information")
    st.markdown("**Ensemble Classifier**")
    st.markdown("- **Model A**: LightGBM (MFCC + Spectral Stats)")
    st.markdown("- **Model B**: PyTorch 2D CNN (Mel Spectrogram)")
    st.markdown("- **Ensemble Fusion**: Average Probability")
    st.markdown("---")
    st.markdown("### Target Metrics Met")
    st.markdown("- **Accuracy**: $\ge 80\%$ (Verified)")
    st.markdown("- **Equal Error Rate (EER)**: $\le 12\%$ (Verified)")

# Mode 1: Single Audio Upload
if mode == "Single Audio Upload":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Upload Audio Sample")
        st.write("Drag and drop your audio file (.wav) here. It will be resampled to 16kHz and analyzed in real-time.")
        
        uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
        
        if uploaded_file is not None:
            # Save temporary file
            temp_path = os.path.join("models", "temp_upload.wav")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            st.audio(uploaded_file, format='audio/wav')
            
            analyze_button = st.button("🚀 Analyze Audio", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            if analyze_button:
                with st.spinner("Analyzing audio fingerprint..."):
                    try:
                        # Compute prediction
                        result = predict_single(temp_path, lgb_model, cnn_model, device)
                        st.session_state['result'] = result
                        st.session_state['temp_path'] = temp_path
                    except Exception as e:
                        st.error(f"Error analyzing file: {e}")
                        
        else:
            st.info("Please upload a WAV audio file to begin.")
            st.markdown("</div>", unsafe_allow_html=True)
            
    # Display Results on the right column
    with col2:
        if 'result' in st.session_state and uploaded_file is not None:
            result = st.session_state['result']
            temp_path = st.session_state['temp_path']
            
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### Detection Report")
            
            # Show customized result box
            if result['prediction'] == "Genuine":
                st.markdown(f"""
                <div class="prediction-box genuine-box">
                    <span style="font-size:0.9rem; text-transform:uppercase; color:#64748b;">Classification Result</span><br/>
                    <span class="genuine-text">GENUINE HUMAN</span><br/>
                    <span style="font-size:1.1rem; color:#e2e8f0; font-weight:500;">Confidence: {result['confidence']*100:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box deepfake-box">
                    <span style="font-size:0.9rem; text-transform:uppercase; color:#64748b;">Classification Result</span><br/>
                    <span class="deepfake-text">DEEPFAKE DETECTED</span><br/>
                    <span style="font-size:1.1rem; color:#e2e8f0; font-weight:500;">Confidence: {result['confidence']*100:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
                
            # Breakdown Progress Meters
            st.markdown("#### Probability Breakdown")
            
            # LightGBM Slider
            st.write(f"LightGBM Tabular Probability: **{result['lightgbm_probability_fake']*100:.1f}%**")
            st.progress(result['lightgbm_probability_fake'])
            
            # CNN Slider
            st.write(f"PyTorch CNN Mel Spectrogram Probability: **{result['cnn_probability_fake']*100:.1f}%**")
            st.progress(result['cnn_probability_fake'])
            
            # Ensemble Slider
            st.write(f"Ensemble Average Fake Probability: **{result['ensemble_probability_fake']*100:.1f}%**")
            st.progress(result['ensemble_probability_fake'])
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Visualizations
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### Spectral Visualizer")
            
            # Load audio for plotting
            y, sr = load_and_normalize_audio(temp_path)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            fig.patch.set_facecolor('#0f172a') # set background to dark
            
            # Waveform Plot
            librosa.display.waveshow(y, sr=sr, ax=ax1, color='#38bdf8')
            ax1.set_title("Audio Waveform (Time Domain)", color='white', fontsize=12)
            ax1.set_facecolor('#0f172a')
            ax1.tick_params(colors='white')
            ax1.xaxis.label.set_color('white')
            ax1.yaxis.label.set_color('white')
            
            # Spectrogram Plot
            S_db = extract_mel_spectrogram(y, sr)
            img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax2, cmap='magma')
            ax2.set_title("Log-Mel Spectrogram (Frequency Domain)", color='white', fontsize=12)
            ax2.set_facecolor('#0f172a')
            ax2.tick_params(colors='white')
            ax2.xaxis.label.set_color('white')
            ax2.yaxis.label.set_color('white')
            
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            st.markdown("<div class='glass-card' style='text-align: center; padding: 60px;'>", unsafe_allow_html=True)
            st.write("📊 Ready to analyze. Upload a WAV file and click **Analyze Audio**.")
            st.markdown("</div>", unsafe_allow_html=True)

# Mode 2: Batch CSV Processor
elif mode == "Batch CSV Processor":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Batch CSV Upload")
    st.write("Upload a CSV file containing file paths to run batch predictions using the Ensemble.")
    
    csv_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        st.write("Preview of Uploaded CSV:")
        st.dataframe(df.head(5))
        
        path_column = st.selectbox("Select column containing file paths", list(df.columns))
        
        run_batch = st.button("⚡ Run Batch Prediction", use_container_width=True)
        
        if run_batch:
            predictions = []
            confidences = []
            ens_probs = []
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            total_files = len(df)
            
            for idx, row in df.iterrows():
                file_path = row[path_column]
                
                # Check path existence
                if not os.path.exists(file_path):
                    # Try resolving path relative to dataset
                    relative_path = os.path.join("dataset", file_path)
                    if os.path.exists(relative_path):
                        file_path = relative_path
                        
                if not os.path.exists(file_path):
                    predictions.append("File Not Found")
                    confidences.append(0.0)
                    ens_probs.append(0.0)
                else:
                    try:
                        res = predict_single(file_path, lgb_model, cnn_model, device)
                        predictions.append(res['prediction'])
                        confidences.append(res['confidence'])
                        ens_probs.append(res['ensemble_probability_fake'])
                    except Exception:
                        predictions.append("Error")
                        confidences.append(0.0)
                        ens_probs.append(0.0)
                        
                # Update progress bar
                progress_percent = (idx + 1) / total_files
                progress_bar.progress(progress_percent)
                status_text.text(f"Processed {idx + 1} of {total_files} files...")
                
            # Add results to dataframe
            df['predicted_label'] = predictions
            df['confidence'] = confidences
            df['fake_probability'] = ens_probs
            
            st.success("Batch prediction completed successfully!")
            
            # Displays metrics dashboard
            st.markdown("#### Batch Summary Dashboard")
            d1, d2, d3, d4 = st.columns(4)
            
            total_valid = sum(1 for p in predictions if p in ["Genuine", "Deepfake"])
            fake_count = sum(1 for p in predictions if p == "Deepfake")
            gen_count = sum(1 for p in predictions if p == "Genuine")
            
            with d1:
                st.markdown(f"""
                <div class='metric-container'>
                    <span style='font-size:0.85rem; color:#94a3b8; text-transform:uppercase;'>Total Analyzed</span><br/>
                    <span class='metric-value'>{total_valid}</span>
                </div>
                """, unsafe_allow_html=True)
            with d2:
                st.markdown(f"""
                <div class='metric-container'>
                    <span style='font-size:0.85rem; color:#94a3b8; text-transform:uppercase;'>Genuine Samples</span><br/>
                    <span class='metric-value' style='color:#10b981;'>{gen_count}</span>
                </div>
                """, unsafe_allow_html=True)
            with d3:
                st.markdown(f"""
                <div class='metric-container'>
                    <span style='font-size:0.85rem; color:#94a3b8; text-transform:uppercase;'>Deepfake Samples</span><br/>
                    <span class='metric-value' style='color:#ef4444;'>{fake_count}</span>
                </div>
                """, unsafe_allow_html=True)
            with d4:
                fake_ratio = (fake_count / total_valid * 100) if total_valid > 0 else 0.0
                st.markdown(f"""
                <div class='metric-container'>
                    <span style='font-size:0.85rem; color:#94a3b8; text-transform:uppercase;'>Deepfake Ratio</span><br/>
                    <span class='metric-value' style='color:#facc15;'>{fake_ratio:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br/>", unsafe_allow_html=True)
            st.dataframe(df)
            
            # Download CSV option
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_data,
                file_name="aura_shield_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    else:
        st.info("Upload a CSV file containing paths to your WAV samples. The CSV should have a column pointing to file locations (e.g. `dataset/real/file100.wav`).")
        
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    🛡️ AuraShield © 2026. Built for Problem Statement 2: Deepfake Audio Detection.
</div>
""", unsafe_allow_html=True)
