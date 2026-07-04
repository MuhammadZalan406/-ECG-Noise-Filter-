import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ecg_processor import ECGProcessor
import time

st.set_page_config(
    page_title="ECG Noise Filter - Zalan Data Scientist",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700;800&display=swap');
    
    .stApp {
        background: #0a0a12;
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main {
        background: 
            radial-gradient(ellipse at 20% 50%, rgba(52, 211, 153, 0.04) 0%, transparent 60%),
            radial-gradient(ellipse at 80% 50%, rgba(96, 165, 250, 0.04) 0%, transparent 60%),
            #0a0a12;
        min-height: 100vh;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 20px;
        padding: 1.8rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(239, 68, 68, 0.02), transparent, rgba(52, 211, 153, 0.02), transparent);
        animation: rotateBorder 15s linear infinite;
    }
    
    .glass-card:hover {
        border-color: rgba(239, 68, 68, 0.08);
        transform: translateY(-4px);
    }
    
    .glass-card > * {
        position: relative;
        z-index: 1;
    }
    
    @keyframes rotateBorder {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .header-container {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
        position: relative;
        z-index: 1;
    }
    
    .name-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(52, 211, 153, 0.15));
        border: 1px solid rgba(239, 68, 68, 0.15);
        border-radius: 50px;
        padding: 0.4rem 1.8rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: rgba(255,255,255,0.5);
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    .name-badge span { color: #34d399; font-weight: 700; }
    
    .title-main {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ef4444, #34d399, #60a5fa, #ef4444);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientFlow 4s ease-in-out infinite;
        letter-spacing: 2px;
        margin-bottom: 0.2rem;
    }
    
    @keyframes gradientFlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .title-sub {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.15);
        letter-spacing: 8px;
        text-transform: uppercase;
        font-weight: 300;
    }
    
    .divider-glow {
        width: 120px;
        height: 2px;
        margin: 0.5rem auto;
        background: linear-gradient(90deg, transparent, #ef4444, #34d399, transparent);
        border-radius: 2px;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.1);
    }
    
    .section-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        color: rgba(255,255,255,0.4);
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(255,255,255,0.03);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .metric-modern {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        padding: 1.2rem 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-modern .label {
        font-size: 0.6rem;
        color: rgba(255,255,255,0.15);
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    
    .metric-modern .value {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    
    .value-red { color: #ef4444; }
    .value-green { color: #34d399; }
    .value-blue { color: #60a5fa; }
    .value-orange { color: #f59e0b; }
    
    .stButton > button {
        font-weight: 600;
        border: none;
        border-radius: 14px;
        padding: 0.7rem 1.2rem;
        font-size: 0.85rem;
        transition: all 0.3s ease;
        width: 100%;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
    }
    
    .btn-process > button {
        background: linear-gradient(135deg, #ef4444, #34d399) !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.15);
    }
    
    .btn-process > button:hover {
        box-shadow: 0 8px 35px rgba(239, 68, 68, 0.25);
    }
    
    .btn-reset > button {
        background: rgba(255,255,255,0.03) !important;
        color: rgba(255,255,255,0.3) !important;
        border: 1px solid rgba(255,255,255,0.03) !important;
    }
    
    .success-glow {
        background: rgba(52, 211, 153, 0.03);
        border: 1px solid rgba(52, 211, 153, 0.06);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        animation: pulseGlow 2s ease-in-out infinite;
    }
    
    @keyframes pulseGlow {
        0%, 100% { border-color: rgba(52, 211, 153, 0.06); }
        50% { border-color: rgba(52, 211, 153, 0.15); }
    }
    
    .footer-modern {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        border-top: 1px solid rgba(255,255,255,0.02);
        margin-top: 1.5rem;
    }
    
    .footer-modern .text {
        color: rgba(255,255,255,0.02);
        font-size: 0.5rem;
        letter-spacing: 6px;
        text-transform: uppercase;
        font-family: 'Orbitron', sans-serif;
    }
    
    .stSlider > div > div > div {
        background: rgba(255,255,255,0.03) !important;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #ef4444, #34d399) !important;
    }
    
    ::-webkit-scrollbar {
        width: 4px;
        height: 4px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.01);
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ef4444, #34d399);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-container">
    <div class="name-badge">
         <span>Zalan</span> - Data Scientist
    </div>
    <div class="title-main">ECG Noise Filter</div>
    <div class="title-sub">Wearable Tech - Signal Processing - Heartbeat Detection</div>
    <div class="divider-glow"></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Configuration")
    
    duration = st.slider(
        "Duration (seconds)",
        min_value=2,
        max_value=10,
        value=5,
        step=1
    )
    
    heart_rate = st.slider(
        "Heart Rate (BPM)",
        min_value=40,
        max_value=120,
        value=72,
        step=2
    )
    
    st.markdown("---")
    
    noise_level = st.slider(
        "Noise Level",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.05
    )
    
    filter_type = st.selectbox(
        "Filter Type",
        ["Bandpass + Median", "Moving Average"]
    )
    
    if filter_type == "Bandpass + Median":
        lowcut = st.slider("Lowcut Frequency (Hz)", 0.1, 5.0, 0.5, 0.1)
        highcut = st.slider("Highcut Frequency (Hz)", 10, 100, 35, 5)
    else:
        filter_window = st.slider("Filter Window Size", 3, 15, 7, 2)
    
    peak_height_factor = st.slider(
        "Peak Sensitivity",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="Higher = fewer false peaks, Lower = more peaks detected"
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("Process", use_container_width=True, key="process")
    with col2:
        reset_btn = st.button("Reset", use_container_width=True, key="reset")

if reset_btn:
    for key in ['processor', 'processed', 'detected_hr']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

if process_btn:
    with st.spinner("Processing ECG signal..."):
        time.sleep(0.5)
        
        processor = ECGProcessor(seed=42)
        
        ecg, timestamps = processor.generate_ecg(duration=duration, heart_rate=heart_rate)
        noisy = processor.add_noise(ecg, noise_level=noise_level, seed=42)
        
        if filter_type == "Bandpass + Median":
            filtered = processor.bandpass_filter(noisy, lowcut=lowcut, highcut=highcut)
        else:
            filtered = processor.moving_average_filter(noisy, window_size=filter_window)
        
        detected_peaks, detected_hr = processor.detect_peaks_adaptive(
            filtered, 
            height_factor=peak_height_factor,
            min_distance=0.4
        )
        
        processor.raw_signal = ecg
        processor.noisy_signal = noisy
        processor.filtered_signal = filtered
        processor.peaks = detected_peaks
        processor.timestamps = timestamps
        
        st.session_state['processor'] = processor
        st.session_state['processed'] = True
        st.session_state['detected_hr'] = detected_hr
        st.session_state['heart_rate'] = heart_rate
        st.session_state['peak_height_factor'] = peak_height_factor
        
        st.success("ECG processing complete!")
        st.rerun()

if 'processed' in st.session_state and st.session_state['processed']:
    processor = st.session_state['processor']
    detected_hr = st.session_state.get('detected_hr', 0)
    heart_rate = st.session_state.get('heart_rate', 72)
    
    true_peaks = processor.true_peaks
    detected_peaks = processor.peaks
    
    precision, recall, f1 = processor.calculate_accuracy(true_peaks, detected_peaks)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-modern">
            <div class="label">Heart Rate</div>
            <div class="value value-red">{heart_rate} BPM</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-modern">
            <div class="label">Detected Rate</div>
            <div class="value value-green">{detected_hr:.2f} BPM</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-modern">
            <div class="label">F1 Score</div>
            <div class="value value-blue">{f1:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-modern">
            <div class="label">Peaks Detected</div>
            <div class="value value-orange">{len(detected_peaks)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="success-glow" style="margin-top: 0.5rem;">
        <div style="color: rgba(255,255,255,0.3); font-size: 0.85rem;">
            Signal processed - {heart_rate} BPM - F1: {f1:.1f}% - Peaks: {len(detected_peaks)}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">ECG Signal Analysis</div>', unsafe_allow_html=True)
    
    fig = processor.visualize_signals()
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Peak Detection Data</div>', unsafe_allow_html=True)
    
    if len(detected_peaks) > 0:
        peak_data = {
            'Peak Index': detected_peaks[:20],
            'Time (s)': [processor.timestamps[i] for i in detected_peaks[:20]],
            'Amplitude': [processor.filtered_signal[i] for i in detected_peaks[:20]]
        }
        df = pd.DataFrame(peak_data)
        st.dataframe(df, width='stretch', hide_index=True)
    else:
        st.warning("No peaks detected!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Detection Metrics</div>', unsafe_allow_html=True)
    
    col_met1, col_met2, col_met3 = st.columns(3)
    with col_met1:
        st.metric("Precision", f"{precision:.1f}%")
    with col_met2:
        st.metric("Recall", f"{recall:.1f}%")
    with col_met3:
        st.metric("F1 Score", f"{f1:.1f}%")
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="glass-card">
        <div style="text-align: center; padding: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;"></div>
            <div style="color: rgba(255,255,255,0.08); font-size: 0.8rem;">Click "Process" to start ECG analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-modern">
    <div class="text">ECG Noise Filter - Signal Processing - 2026</div>
</div>
""", unsafe_allow_html=True)
