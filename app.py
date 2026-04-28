import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import tempfile
import os
from PIL import Image
import plotly.graph_objects as go
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Video Forgery Detector",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-size: 20px;
        padding: 10px;
    }
    .stButton > button:hover {
        background-color: #ff0000;
        color: white;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .error-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .info-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎥 Video Forgery Detection System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📌 About")
    st.info(
        """
        This tool uses Deep Learning (ResNet50) to detect 
        tampering in videos. It analyzes frame-by-frame 
        to identify potential forgery.
        
        **Features:**
        - Real-time video analysis
        - Frame-by-frame inspection
        - Confidence scoring
        - Detailed visualizations
        """
    )
    
    st.header("⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,  # Changed from 0.5 to 0.3 for better sensitivity
        step=0.05,
        help="Minimum confidence to classify as forgery (Lower = more sensitive)"
    )
    
    frame_interval = st.slider(
        "Frame Sampling Rate",
        min_value=1,
        max_value=30,
        value=5,
        help="Analyze every Nth frame (higher = faster but less accurate)"
    )
    
    st.header("📊 Statistics")
    st.markdown("---")
    st.metric("Model", "ResNet50")
    st.metric("Input Size", "240x320")  # Fixed to match model
    st.metric("Framework", "TensorFlow 2.x")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'flv'],
        help="Supported formats: MP4, AVI, MOV, MKV, FLV"
    )

with col2:
    st.header("🎯 Detection Mode")
    detection_mode = st.radio(
        "Select analysis type:",
        ["Quick Scan (Faster)", "Detailed Analysis (Slower)"],
        index=0
    )
    show_frames = st.checkbox("Show analyzed frames", value=True)

# Function to load model with caching
@st.cache_resource
def load_forgery_model():
    try:
        model = load_model('ResNet50_Model/forgery_model.hdf5')
        # Display model input shape for debugging
        st.success(f"✅ Model loaded successfully! Input shape: {model.input_shape}")
        return model
    except Exception as e:
        st.warning(f"Model file not found or error: {e}\nUsing demonstration mode with random predictions.")
        return None

# Function to preprocess frame - FIXED for model input size
def preprocess_frame(frame):
    # Resize to match model's expected input (240, 320)
    frame = cv2.resize(frame, (320, 240))  # (width, height)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = preprocess_input(frame)
    return frame

# Function to analyze video
def analyze_video(video_path, model, frame_interval, confidence_threshold):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps
    
    predictions = []
    frames_analyzed = []
    frame_numbers = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    analyzed_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Sample every Nth frame
        if frame_count % frame_interval == 0:
            status_text.text(f"Analyzing frame {frame_count}/{total_frames}...")
            
            # Preprocess frame
            processed_frame = preprocess_frame(frame)
            processed_frame = np.expand_dims(processed_frame, axis=0)
            
            # Make prediction
            if model:
                pred = model.predict(processed_frame, verbose=0)
                # Handle different output formats
                if len(pred[0]) == 1:
                    probability = float(pred[0][0])
                else:
                    # For binary classification, take the positive class probability
                    probability = float(pred[0][1]) if len(pred[0]) > 1 else float(pred[0][0])
            else:
                # Demo mode - random prediction
                probability = np.random.random()
            
            is_forgery = probability >= confidence_threshold
            
            predictions.append({
                'frame': frame_count,
                'probability': probability,
                'is_forgery': is_forgery,
                'timestamp': frame_count / fps
            })
            
            if show_frames and len(frames_analyzed) < 10:  # Limit displayed frames
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_analyzed.append(frame_rgb)
                frame_numbers.append(frame_count)
            
            analyzed_count += 1
            progress_bar.progress(min(frame_count / total_frames, 1.0))
        
        frame_count += 1
    
    cap.release()
    status_text.text("✅ Analysis complete!")
    
    return predictions, frames_analyzed, frame_numbers, total_frames, fps, duration

# Function to display results - IMPROVED verdict logic
def display_results(predictions, total_frames, fps, duration, confidence_threshold):
    forgery_frames = [p for p in predictions if p['is_forgery']]
    authentic_frames = [p for p in predictions if not p['is_forgery']]
    
    # Calculate additional metrics
    forgery_ratio = len(forgery_frames) / len(predictions) if len(predictions) > 0 else 0
    max_probability = max([p['probability'] for p in predictions]) if predictions else 0
    avg_probability = np.mean([p['probability'] for p in predictions]) if predictions else 0
    
    # Summary metrics
    st.markdown("### 📊 Results Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Frames", len(predictions))
    with col2:
        st.metric("Forgery Frames", len(forgery_frames), 
                  delta=f"{(forgery_ratio)*100:.1f}%")
    with col3:
        st.metric("Peak Probability", f"{max_probability*100:.1f}%")
    with col4:
        st.metric("Avg Probability", f"{avg_probability*100:.1f}%")
    with col5:
        st.metric("Video Duration", f"{duration:.2f}s")
    
    # IMPROVED: More sensitive verdict logic
    st.markdown("### 🎯 Detection Verdict")
    
    # Multiple criteria for better detection
    is_highly_tampered = max_probability >= 0.7 or forgery_ratio > 0.4
    is_suspicious = max_probability >= 0.5 or forgery_ratio > 0.2 or avg_probability > 0.35
    is_likely_forged = max_probability >= 0.6 or forgery_ratio > 0.3
    
    if is_highly_tampered or is_likely_forged:
        st.markdown(
            f"""
            <div class="error-box">
                <h3>⚠️ VERDICT: VIDEO TAMPERED / FORGED</h3>
                <p><b>Detection evidence:</b><br>
                • {forgery_ratio*100:.1f}% of frames show tampering<br>
                • Peak forgery probability: {max_probability*100:.1f}%<br>
                • Average forgery probability: {avg_probability*100:.1f}%</p>
                <p><b>Recommendation:</b> This video shows clear signs of manipulation.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    elif is_suspicious:
        st.markdown(
            f"""
            <div class="info-box">
                <h3>⚠️ VERDICT: SUSPICIOUS - POSSIBLE TAMPERING</h3>
                <p><b>Detection evidence:</b><br>
                • {forgery_ratio*100:.1f}% of frames show anomalies<br>
                • Peak confidence: {max_probability*100:.1f}%<br>
                • Average confidence: {avg_probability*100:.1f}%</p>
                <p><b>Recommendation:</b> Manual review strongly recommended.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="success-box">
                <h3>✅ VERDICT: LIKELY AUTHENTIC</h3>
                <p>Only {forgery_ratio*100:.1f}% of frames show minimal signs of tampering<br>
                Peak confidence: {max_probability*100:.1f}%</p>
                <p><b>Recommendation:</b> Video appears authentic.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Probability chart
    st.markdown("### 📈 Forgery Probability Over Time")
    
    fig = go.Figure()
    
    timestamps = [p['timestamp'] for p in predictions]
    probabilities = [p['probability'] for p in predictions]
    colors = ['red' if p['is_forgery'] else 'green' for p in predictions]
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=probabilities,
        mode='lines+markers',
        name='Forgery Probability',
        line=dict(color='blue', width=2),
        marker=dict(size=6, color=colors),
        hovertemplate='<b>Time: %{x:.2f}s</b><br>' +
                      'Probability: %{y:.3f}<br>' +
                      'Status: %{text}<extra></extra>',
        text=['⚠️ Tampered' if p['is_forgery'] else '✅ Authentic' for p in predictions]
    ))
    
    # Add threshold line
    fig.add_hline(y=confidence_threshold, line_dash="dash", 
                  line_color="red", annotation_text=f"Threshold ({confidence_threshold})")
    
    # Add max probability annotation
    max_idx = np.argmax(probabilities)
    fig.add_annotation(
        x=timestamps[max_idx],
        y=probabilities[max_idx],
        text=f"Peak: {probabilities[max_idx]*100:.1f}%",
        showarrow=True,
        arrowhead=1,
        arrowcolor="red"
    )
    
    fig.update_layout(
        title="Frame-by-Frame Forgery Detection",
        xaxis_title="Time (seconds)",
        yaxis_title="Forgery Probability",
        yaxis_range=[0, 1],
        hovermode='closest',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    with st.expander("📋 View Detailed Frame Analysis"):
        df = pd.DataFrame(predictions)
        df['timestamp'] = df['timestamp'].apply(lambda x: f"{x:.2f}s")
        df['probability'] = df['probability'].apply(lambda x: f"{x:.3f} ({float(x)*100:.1f}%)")
        df['is_forgery'] = df['is_forgery'].apply(lambda x: "⚠️ TAMPERED" if x else "✅ AUTHENTIC")
        df = df[['frame', 'timestamp', 'probability', 'is_forgery']]
        df.columns = ['Frame #', 'Timestamp (s)', 'Probability', 'Status']
        st.dataframe(df, use_container_width=True, height=400)

# Main execution
if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    # Display file info
    st.info(f"📹 Processing: **{uploaded_file.name}** ({(uploaded_file.size/1024/1024):.2f} MB)")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_forgery_model()
    
    # Determine frame interval based on detection mode
    if detection_mode == "Quick Scan (Faster)经商":
        frame_interval = max(frame_interval, 10)
        st.info("⚡ Quick Scan mode enabled - analyzing fewer frames for speed")
    else:
        frame_interval = min(frame_interval, 3)
        st.info("🔍 Detailed Analysis mode enabled - analyzing more frames for accuracy")
    
    # Analyze video
    with st.spinner("Analyzing video frames..."):
        predictions, frames_analyzed, frame_numbers, total_frames, fps, duration = analyze_video(
            video_path, model, frame_interval, confidence_threshold
        )
    
    # Display results
    display_results(predictions, total_frames, fps, duration, confidence_threshold)
    
    # Display sample frames
    if show_frames and frames_analyzed:
        st.markdown("### 🖼️ Sample Analyzed Frames")
        st.caption("First 10 analyzed frames from the video")
        cols = st.columns(min(len(frames_analyzed), 5))
        for idx, (col, frame, frame_num) in enumerate(zip(cols, frames_analyzed, frame_numbers)):
            with col:
                # Resize frame for display
                frame_display = cv2.resize(frame, (320, 240))
                st.image(frame_display, caption=f"Frame {frame_num}", use_container_width=True)
    
    # Download results
    results_df = pd.DataFrame(predictions)
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Complete Analysis Report (CSV)",
        data=csv,
        file_name=f"forgery_detection_{uploaded_file.name.split('.')[0]}.csv",
        mime="text/csv"
    )
    
    # Cleanup
    os.unlink(video_path)
    
    # Success message
    st.balloons()
    
else:
    # Show instructions when no video is uploaded
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎬 How it works
        1. Upload a video file
        2. Our AI analyzes each frame
        3. Get instant forgery detection results
        4. Download detailed report
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 What we detect
        - Frame tampering
        - Object insertion/removal
        - Temporal inconsistencies
        - Artifacts and noise patterns
        """)
    
    with col3:
        st.markdown("""
        ### 💡 Tips
        - Use clear, high-quality videos
        - **Lower threshold (0.2-0.3) for sensitive detection**
        - Longer videos take more time
        - Export results for evidence
        """)
    
    # Demo video option
    st.markdown("---")
    st.markdown("### 🎯 Try with a sample")
    st.info("Upload a forged video and set Confidence Threshold to 0.3 for best results")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>Powered by ResNet50 Deep Learning Model | Video Forgery Detection System</p>
    <p>⚠️ For best results, use Confidence Threshold around 0.3 for forged videos</p>
</div>
""", unsafe_allow_html=True)