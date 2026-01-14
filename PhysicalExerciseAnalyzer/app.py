import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os
import time
import json
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.pose_detection import PoseDetector
    from src.squat_checker import SquatFormChecker
    from src.pushup_checker import PushUpFormChecker
    from src.pullup_checker import PullUpFormChecker
    from src.dip_checker import DipFormChecker
    from src.video_analyzer import ExerciseAnalyzer
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please make sure all dependencies are installed: pip install -r requirements.txt")
    st.stop()

st.set_page_config(
    page_title="Exercise Form Analyzer",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; }
    div[data-testid="stMetric"] { padding: 0.5rem 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: rgba(0, 0, 0, 0.1); border-radius: 10px; padding: 10px 20px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: rgba(0, 0, 0, 0.2); }
    div[data-testid="stExpander"] { background-color: rgba(0, 0, 0, 0.05); border-radius: 10px; border: 1px solid rgba(0, 0, 0, 0.1); }
    h1, h2, h3 { font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding: 0.3rem 0 0.2rem 0;'>
    <h1 style='font-size: 2rem; margin: 0;'>💪 Exercise Form Analyzer</h1>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    exercise_type = st.selectbox(
        "Select Exercise",
        ["Squat", "Push-up", "Pull-up", "Dip"],
        label_visibility="collapsed"
    )
    st.session_state['exercise_type'] = exercise_type

if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'rep_count' not in st.session_state:
    st.session_state.rep_count = 0


def display_exercise_info(exercise: str):
    if exercise == "Squat":
        with st.expander("📚 Squat Form Guide - Click to expand", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style='background: #f0f9ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3b82f6;'>
                    <h4 style='color: #1e40af; margin-top: 0;'>✅ Proper Form</h4>
                    <ul style='line-height: 1.8; color: #1e3a8a;'>
                        <li>🦶 Feet shoulder-width apart</li>
                        <li>👟 Toes slightly pointed out</li>
                        <li>💪 Chest up, core engaged</li>
                        <li>🎯 Knees track over toes</li>
                        <li>📏 Lower until thighs parallel</li>
                        <li>⬆️ Push through heels</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style='background: #fef2f2; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ef4444;'>
                    <h4 style='color: #991b1b; margin-top: 0;'>❌ Common Mistakes</h4>
                    <ul style='line-height: 1.8; color: #7f1d1d;'>
                        <li>⚠️ Knees cave inward</li>
                        <li>⚠️ Knees go too far forward</li>
                        <li>⚠️ Excessive forward lean</li>
                        <li>⚠️ Not going deep enough</li>
                        <li>⚠️ Rising on toes</li>
                        <li>⚠️ Uneven weight distribution</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    elif exercise == "Push-up":
        with st.expander("📚 Push-up Form Guide - Click to expand", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style='background: #f0f9ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3b82f6;'>
                    <h4 style='color: #1e40af; margin-top: 0;'>✅ Proper Form</h4>
                    <ul style='line-height: 1.8; color: #1e3a8a;'>
                        <li>🤲 Hands shoulder-width apart</li>
                        <li>📏 Body in straight line</li>
                        <li>💪 Core engaged throughout</li>
                        <li>⬇️ Lower chest to ground</li>
                        <li>📐 Elbows at 45° angle</li>
                        <li>⬆️ Full extension at top</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style='background: #fef2f2; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ef4444;'>
                    <h4 style='color: #991b1b; margin-top: 0;'>❌ Common Mistakes</h4>
                    <ul style='line-height: 1.8; color: #7f1d1d;'>
                        <li>⚠️ Hips sagging</li>
                        <li>⚠️ Hips too high (piking)</li>
                        <li>⚠️ Elbows flaring out</li>
                        <li>⚠️ Not going deep enough</li>
                        <li>⚠️ Head not neutral</li>
                        <li>⚠️ Incomplete range of motion</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    elif exercise == "Pull-up":
        with st.expander("📚 Pull-up Form Guide - Click to expand", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style='background: #f0f9ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3b82f6;'>
                    <h4 style='color: #1e40af; margin-top: 0;'>✅ Proper Form</h4>
                    <ul style='line-height: 1.8; color: #1e3a8a;'>
                        <li>🤲 Hands shoulder-width apart</li>
                        <li>💪 Start from full hang</li>
                        <li>⬆️ Pull chin above bar</li>
                        <li>📏 Control the movement</li>
                        <li>⬇️ Lower with control</li>
                        <li>🎯 Minimize body swing</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style='background: #fef2f2; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ef4444;'>
                    <h4 style='color: #991b1b; margin-top: 0;'>❌ Common Mistakes</h4>
                    <ul style='line-height: 1.8; color: #7f1d1d;'>
                        <li>⚠️ Not pulling high enough</li>
                        <li>⚠️ Too much swinging/kipping</li>
                        <li>⚠️ Not full extension at bottom</li>
                        <li>⚠️ Using momentum</li>
                        <li>⚠️ Uneven shoulder movement</li>
                        <li>⚠️ Incomplete range of motion</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    else:  # Dip
        with st.expander("📚 Dip Form Guide - Click to expand", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style='background: #f0f9ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3b82f6;'>
                    <h4 style='color: #1e40af; margin-top: 0;'>✅ Proper Form</h4>
                    <ul style='line-height: 1.8; color: #1e3a8a;'>
                        <li>💪 Start with arms extended</li>
                        <li>⬇️ Lower until 90° elbow angle</li>
                        <li>🎯 Keep elbows close to body</li>
                        <li>📏 Maintain upright torso</li>
                        <li>⬆️ Press back to start</li>
                        <li>🔒 Full lockout at top</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style='background: #fef2f2; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ef4444;'>
                    <h4 style='color: #991b1b; margin-top: 0;'>❌ Common Mistakes</h4>
                    <ul style='line-height: 1.8; color: #7f1d1d;'>
                        <li>⚠️ Not going deep enough</li>
                        <li>⚠️ Elbows flaring out</li>
                        <li>⚠️ Excessive forward lean</li>
                        <li>⚠️ Shoulders shrugging up</li>
                        <li>⚠️ Using momentum/bouncing</li>
                        <li>⚠️ Incomplete extension at top</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)


def detect_exercise_type(landmarks_dict):
    """
    Detect the most likely exercise type based on body position.
    
    Args:
        landmarks_dict: Dictionary of landmark positions with named keys
        
    Returns:
        Tuple of (suggested_exercise, confidence_message)
    """
    try:
        # Extract key landmarks
        left_shoulder = landmarks_dict['left_shoulder']
        right_shoulder = landmarks_dict['right_shoulder']
        left_hip = landmarks_dict['left_hip']
        right_hip = landmarks_dict['right_hip']
        left_knee = landmarks_dict['left_knee']
        right_knee = landmarks_dict['right_knee']
        left_ankle = landmarks_dict['left_ankle']
        right_ankle = landmarks_dict['right_ankle']
        left_wrist = landmarks_dict['left_wrist']
        right_wrist = landmarks_dict['right_wrist']
        left_elbow = landmarks_dict['left_elbow']
        right_elbow = landmarks_dict['right_elbow']
        
        # Calculate average positions (Y: lower values = higher on screen)
        avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        avg_hip_y = (left_hip['y'] + right_hip['y']) / 2
        avg_knee_y = (left_knee['y'] + right_knee['y']) / 2
        avg_ankle_y = (left_ankle['y'] + right_ankle['y']) / 2
        avg_wrist_y = (left_wrist['y'] + right_wrist['y']) / 2
        avg_elbow_y = (left_elbow['y'] + right_elbow['y']) / 2
        
        avg_wrist_x = (left_wrist['x'] + right_wrist['x']) / 2
        avg_shoulder_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        avg_hip_x = (left_hip['x'] + right_hip['x']) / 2
        
        torso_height = abs(avg_shoulder_y - avg_hip_y)
        shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
        body_height = abs(avg_shoulder_y - avg_ankle_y)
        
        is_horizontal = torso_height < shoulder_width * 0.8
        
        avg_knee_ankle_vertical = abs(avg_knee_y - avg_ankle_y)
        avg_knee_ankle_horizontal = abs((left_knee['x'] + right_knee['x'])/2 - (left_ankle['x'] + right_ankle['x'])/2)
        
        lower_legs_vertical = avg_knee_ankle_vertical > torso_height * 0.4
        lower_legs_horizontal = avg_knee_ankle_vertical < torso_height * 0.2
        
        knee_ankle_vertical_ratio = avg_knee_ankle_vertical / (torso_height + 0.001)
        legs_bent_perpendicular = knee_ankle_vertical_ratio < 0.15
        
        hands_above_shoulders = avg_wrist_y < avg_shoulder_y - 0.08
        hands_near_or_above_shoulders = avg_wrist_y < avg_shoulder_y + 0.05
        hands_below_hips = avg_wrist_y > avg_hip_y
        hands_beside_torso = abs(avg_wrist_x - avg_shoulder_x) > shoulder_width * 0.8
        
        feet_on_ground = avg_ankle_y > avg_hip_y + torso_height * 0.3
        feet_visible = avg_ankle_y > avg_knee_y
        
        elbows_above_shoulders = avg_elbow_y < avg_shoulder_y
        elbows_beside_body = abs(avg_wrist_x - avg_hip_x) < shoulder_width * 1.5
        
        scores = {
            'Pull-up': 0,
            'Dip': 0,
            'Push-up': 0,
            'Squat': 0
        }
        
        if elbows_above_shoulders:
            scores['Pull-up'] += 4
        if hands_above_shoulders:
            scores['Pull-up'] += 4
        # Also accept hands near shoulder level if elbows are up (pull position)
        elif hands_near_or_above_shoulders and elbows_above_shoulders:
            scores['Pull-up'] += 3
        # Vertical body with feet off ground (hanging) - key differentiator from squat
        if not is_horizontal and not feet_on_ground:
            scores['Pull-up'] += 5
        # Hands not beside torso (arms overhead, not at sides)
        if not hands_beside_torso:
            scores['Pull-up'] += 2
        # Additional check: if feet are clearly off ground and body is vertical
        if avg_ankle_y < avg_knee_y + torso_height * 0.3:
            scores['Pull-up'] += 3
        # Strong penalty only if BOTH vertical legs AND feet firmly on ground (standing)
        if lower_legs_vertical and feet_on_ground:
            scores['Pull-up'] -= 5
            
        # Dip detection: vertical body, hands beside/below shoulders, arms supporting body, feet NOT firmly on ground
        if not is_horizontal:
            scores['Dip'] += 1
        if not hands_above_shoulders and not hands_below_hips:
            scores['Dip'] += 2
        if hands_beside_torso:
            scores['Dip'] += 3
        # Key difference from squat: feet should be raised/not on ground AND lower legs not vertical
        if not feet_on_ground and avg_ankle_y < avg_hip_y + torso_height:
            scores['Dip'] += 3
        # Lower legs horizontal or bent perpendicular (90° at knee) = dip position
        if lower_legs_horizontal or legs_bent_perpendicular:
            scores['Dip'] += 3
        if elbows_beside_body:
            scores['Dip'] += 1
            
        # Push-up detection: horizontal body, hands on ground
        if is_horizontal:
            scores['Push-up'] += 4
        if abs(avg_wrist_y - avg_shoulder_y) < torso_height * 0.3:
            scores['Push-up'] += 2
        if not feet_on_ground:
            scores['Push-up'] += 1
        # Push-ups can have horizontal lower legs (plank position)
        if lower_legs_horizontal:
            scores['Push-up'] += 1
            
        # Squat detection: standing upright, feet on ground, VERTICAL LOWER LEGS (key difference!)
        if not is_horizontal:
            scores['Squat'] += 1
        # CRITICAL: Squats require vertical lower legs (shins upright)
        if lower_legs_vertical and not legs_bent_perpendicular:
            scores['Squat'] += 3  # Reduced from 4
        if feet_on_ground and feet_visible:
            scores['Squat'] += 3
        if not hands_above_shoulders and not hands_beside_torso:
            scores['Squat'] += 2
        if avg_wrist_y > avg_shoulder_y and avg_wrist_y < avg_knee_y:
            scores['Squat'] += 2
        if avg_ankle_y > avg_hip_y:
            scores['Squat'] += 1
        # Penalize if legs are horizontal (perpendicular/parallel to ground)
        if lower_legs_horizontal or legs_bent_perpendicular:
            scores['Squat'] -= 5  # Strong penalty
        # Strong penalty if hands are clearly above head (pull-up position)
        if hands_above_shoulders or elbows_above_shoulders:
            scores['Squat'] -= 6
        # Penalize if feet are clearly off the ground
        if not feet_on_ground:
            scores['Squat'] -= 4
            
        # Get best match
        max_score = max(scores.values())
        if max_score < 3:
            return None, "Could not confidently detect exercise type"
        
        detected = max(scores, key=scores.get)
        
        # Generate confidence message
        messages = {
            'Pull-up': f"Hands above head, body hanging (score: {scores['Pull-up']})",
            'Dip': f"Vertical body with hands supporting beside torso (score: {scores['Dip']})",
            'Push-up': f"Horizontal body position (score: {scores['Push-up']})",
            'Squat': f"Standing upright with feet on ground (score: {scores['Squat']})"
        }
        
        return detected, messages[detected]
        
    except Exception as e:
        return None, f"Detection error: {str(e)}"


def process_image(image, exercise_type, use_detected=False):
    """Process uploaded image for pose analysis."""
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Initialize components
    pose_detector = PoseDetector()
    
    # Detect pose first to get exercise type if needed
    landmarks, annotated_img = pose_detector.detect_pose(img_bgr)
    
    if landmarks is None:
        st.error("❌ No pose detected in image. Make sure your full body is visible.")
        pose_detector.close()
        return None, None
    
    # Convert landmarks to dictionary with named keys
    landmarks_dict = {}
    landmark_names = pose_detector.LANDMARK_NAMES
    for name, idx in landmark_names.items():
        if idx < len(landmarks):
            lm = landmarks[idx]
            landmarks_dict[name] = {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility if hasattr(lm, 'visibility') else 1.0
            }
    
    # Detect exercise type from pose
    detected_exercise, detection_message = detect_exercise_type(landmarks_dict)
    
    # Use detected exercise if requested and detection succeeded
    actual_exercise = detected_exercise if (use_detected and detected_exercise) else exercise_type
    
    # Initialize form checker based on actual exercise type
    if actual_exercise.lower() == 'squat':
        form_checker = SquatFormChecker()
    elif actual_exercise.lower() == 'pull-up':
        form_checker = PullUpFormChecker()
    elif actual_exercise.lower() == 'dip':
        form_checker = DipFormChecker()
    else:
        form_checker = PushUpFormChecker()
    
    # Analyze form
    analysis = form_checker.analyze_frame(landmarks_dict)
    
    # Add detection info to analysis
    analysis['detected_exercise'] = detected_exercise
    analysis['detection_message'] = detection_message
    analysis['actual_exercise'] = actual_exercise
    
    # Draw pose on image
    annotated_image = pose_detector.draw_landmarks(img_bgr, landmarks)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    pose_detector.close()
    
    return annotated_image, analysis


# Main content
tab1, tab2, tab3 = st.tabs(["📸 Image Analysis", "🎥 Video Analysis", "🖥️ Live Webcam"])

with tab1:
    st.markdown("""<h2>📸 Upload Exercise Photo</h2>""", unsafe_allow_html=True)
    display_exercise_info(exercise_type)
    
    uploaded_file = st.file_uploader("📁 Choose an image...", type=['png', 'jpg', 'jpeg'], help="Upload a photo showing your full body during exercise")
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""<h3>🖼️ Original Image</h3>""", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        
        # Process image
        with st.spinner("🔄 Analyzing pose and form..."):
            # First try with selected exercise
            annotated_image, analysis = process_image(image, exercise_type, use_detected=False)
        
        if annotated_image is not None and analysis is not None:
            # Check if detected exercise differs from selected
            if analysis.get('detected_exercise') and analysis['detected_exercise'] != exercise_type:
                st.warning(f"""
                    🤔 **Exercise Type Mismatch Detected!**
                    
                    You selected **{exercise_type}** but the pose looks like a **{analysis['detected_exercise']}**.
                    
                    *{analysis['detection_message']}*
                    
                    **Automatically using {analysis['detected_exercise']} analysis below.**
                """)
                # Re-analyze with detected exercise type
                annotated_image, analysis = process_image(image, exercise_type, use_detected=True)
            
            with col2:
                st.markdown("""<h3>🎯 Pose Detection</h3>""", unsafe_allow_html=True)
                st.image(annotated_image, use_container_width=True)
            
            # Display analysis results
            st.markdown("---")
            st.markdown("""<h2>📊 Form Analysis Results</h2>""", unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                score_color = "normal" if analysis['form_score'] >= 80 else "inverse"
                st.metric("💯 Form Score", f"{analysis['form_score']}%", 
                         delta="Good" if analysis['form_score'] >= 80 else "Needs Work")
            
            with col2:
                st.metric("🎯 Stage", analysis['stage'].upper())
            
            # Use actual exercise type from analysis (detected or selected)
            actual_ex = analysis.get('actual_exercise', exercise_type)
            
            if actual_ex == "Squat":
                with col3:
                    st.metric("🦵 Knee Angle", f"{analysis['knee_angle']:.1f}°")
                with col4:
                    st.metric("📐 Body Lean", f"{analysis['body_lean']:.1f}°")
            elif actual_ex == "Pull-up":
                with col3:
                    st.metric("💪 Elbow Angle", f"{analysis['elbow_angle']:.1f}°")
                with col4:
                    st.metric("📐 Body Swing", f"{analysis['body_swing']:.1f}°")
            elif actual_ex == "Dip":
                with col3:
                    st.metric("💪 Elbow Angle", f"{analysis['elbow_angle']:.1f}°")
                with col4:
                    st.metric("📐 Torso Lean", f"{analysis['torso_lean']:.1f}°")
            else:
                with col3:
                    st.metric("💪 Elbow Angle", f"{analysis['elbow_angle']:.1f}°")
                with col4:
                    st.metric("📏 Body Angle", f"{analysis['body_angle']:.1f}°")
            
            # Feedback with modern cards
            st.markdown("<br>", unsafe_allow_html=True)
            
            if analysis['form_score'] >= 80:
                st.markdown("""
                <div style='background: #d1fae5; padding: 1.5rem; border-radius: 15px; text-align: center; border: 2px solid #10b981;'>
                    <h3 style='color: #065f46; margin: 0;'>✅ Excellent Form!</h3>
                    <p style='color: #047857; margin-top: 0.5rem;'>Keep up the great work!</p>
                </div>
                """, unsafe_allow_html=True)
            elif analysis['form_score'] >= 60:
                st.markdown("""
                <div style='background: #fef3c7; padding: 1.5rem; border-radius: 15px; text-align: center; border: 2px solid #f59e0b;'>
                    <h3 style='color: #92400e; margin: 0;'>⚠️ Good Form</h3>
                    <p style='color: #b45309; margin-top: 0.5rem;'>Room for improvement - check feedback below</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: #fee2e2; padding: 1.5rem; border-radius: 15px; text-align: center; border: 2px solid #ef4444;'>
                    <h3 style='color: #991b1b; margin: 0;'>❌ Form Needs Work</h3>
                    <p style='color: #b91c1c; margin-top: 0.5rem;'>Focus on the corrections below</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Detailed feedback
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""<h4>✅ Form Check</h4>""", unsafe_allow_html=True)
                for feedback in analysis['feedback']:
                    st.markdown(f"""<div style='background: #f0f9ff; padding: 0.8rem; margin: 0.5rem 0; border-radius: 8px; border-left: 3px solid #3b82f6; color: #1e3a8a;'>{feedback}</div>""", unsafe_allow_html=True)
            
            with col2:
                if analysis['errors']:
                    st.markdown("""<h4>🔴 Issues to Address</h4>""", unsafe_allow_html=True)
                    for error in analysis['errors']:
                        st.markdown(f"""<div style='background: #fee2e2; padding: 0.8rem; margin: 0.5rem 0; border-radius: 8px; border-left: 3px solid #ef4444; color: #7f1d1d;'>{error}</div>""", unsafe_allow_html=True)
            
            # Angle details
            with st.expander("📐 Detailed Angle Measurements"):
                angles = analysis['angles']
                for joint, angle in angles.items():
                    st.write(f"**{joint.replace('_', ' ').title()}:** {angle:.2f}°")

with tab2:
    st.header("🎥 Video Analysis")
    st.markdown("Upload a video of yourself doing the exercise for detailed frame-by-frame analysis.")
    
    display_exercise_info(exercise_type)
    
    uploaded_video = st.file_uploader("Choose a video file...", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_video is not None:
        # Save uploaded video temporarily
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        st.video(uploaded_video)
        
        # Analysis options
        col1, col2 = st.columns(2)
        with col1:
            analyze_every_n_frames = st.slider("Analyze every N frames", 1, 10, 3, 
                                              help="Process every Nth frame for faster analysis")
        with col2:
            show_annotated = st.checkbox("Show annotated video", value=True)
        
        if st.button("🔍 Analyze Video", type="primary"):
            # Initialize analyzer
            analyzer = ExerciseAnalyzer(exercise_type=exercise_type.lower())
            
            # Process video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                st.error("Could not open video file")
            else:
                # Get video properties
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = total_frames / fps if fps > 0 else 0
                
                st.info(f"Video: {total_frames} frames, {fps} FPS, {duration:.1f} seconds")
                
                # Storage for analysis
                all_analyses = []
                annotated_frames = []
                frame_numbers = []
                detected_exercises = []  # Track detected exercise types
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                frame_count = 0
                processed_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every Nth frame
                    if frame_count % analyze_every_n_frames == 0:
                        # Analyze frame
                        annotated_frame, analysis = analyzer.process_frame(frame)
                        
                        if analysis is not None:
                            all_analyses.append(analysis)
                            frame_numbers.append(frame_count)
                            
                            # Detect exercise type from landmarks
                            landmarks, _ = analyzer.pose_detector.detect_pose(frame)
                            if landmarks is not None:
                                landmarks_dict = {}
                                for name, idx in analyzer.pose_detector.LANDMARK_NAMES.items():
                                    if idx < len(landmarks):
                                        lm = landmarks[idx]
                                        landmarks_dict[name] = {
                                            'x': lm.x,
                                            'y': lm.y,
                                            'z': lm.z,
                                            'visibility': lm.visibility if hasattr(lm, 'visibility') else 1.0
                                        }
                                detected_ex, _ = detect_exercise_type(landmarks_dict)
                                if detected_ex:
                                    detected_exercises.append(detected_ex)
                            
                            if show_annotated:
                                annotated_frames.append(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                        
                        processed_count += 1
                        
                        # Update progress
                        progress = frame_count / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%)")
                    
                    frame_count += 1
                
                cap.release()
                progress_bar.progress(1.0)
                status_text.text(f"✅ Analysis complete! Processed {processed_count} frames")
                
                # Clean up temp file
                import os
                os.unlink(video_path)
                
                if all_analyses:
                    st.success(f"✅ Analyzed {len(all_analyses)} frames successfully!")
                    
                    # Check detected exercise type
                    if detected_exercises:
                        from collections import Counter
                        exercise_counts = Counter(detected_exercises)
                        most_common_exercise, count = exercise_counts.most_common(1)[0]
                        detection_percentage = (count / len(detected_exercises)) * 100
                        
                        if most_common_exercise != exercise_type and detection_percentage > 30:
                            st.warning(f"""
                                🤔 **Exercise Type Mismatch Detected!**
                                
                                You selected **{exercise_type}** but {detection_percentage:.0f}% of frames look like **{most_common_exercise}**.
                                
                                Consider changing the exercise type in the sidebar for more accurate analysis.
                            """)
                        else:
                            st.info(f"✅ Exercise detection: {most_common_exercise.upper()} ({detection_percentage:.0f}% confidence)")
                    
                    # Calculate statistics
                    total_reps = max([a['rep_count'] for a in all_analyses]) if all_analyses else 0
                    avg_form_score = np.mean([a['form_score'] for a in all_analyses])
                    
                    if exercise_type == "Squat":
                        avg_knee_angle = np.mean([a['knee_angle'] for a in all_analyses])
                        avg_body_lean = np.mean([a['body_lean'] for a in all_analyses])
                    elif exercise_type == "Pull-up":
                        avg_elbow_angle = np.mean([a['elbow_angle'] for a in all_analyses])
                        avg_body_swing = np.mean([a['body_swing'] for a in all_analyses])
                    elif exercise_type == "Dip":
                        avg_elbow_angle = np.mean([a['elbow_angle'] for a in all_analyses])
                        avg_torso_lean = np.mean([a['torso_lean'] for a in all_analyses])
                    else:
                        avg_elbow_angle = np.mean([a['elbow_angle'] for a in all_analyses])
                        avg_body_angle = np.mean([a['body_angle'] for a in all_analyses])
                    
                    # Display summary
                    st.markdown("---")
                    st.header("📊 Analysis Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Reps", total_reps)
                    with col2:
                        st.metric("Avg Form Score", f"{avg_form_score:.1f}%")
                    with col3:
                        if exercise_type == "Squat":
                            st.metric("Avg Knee Angle", f"{avg_knee_angle:.1f}°")
                        else:
                            st.metric("Avg Elbow Angle", f"{avg_elbow_angle:.1f}°")
                    with col4:
                        if exercise_type == "Squat":
                            st.metric("Avg Body Lean", f"{avg_body_lean:.1f}°")
                        elif exercise_type == "Pull-up":
                            st.metric("Avg Body Swing", f"{avg_body_swing:.1f}°")
                        elif exercise_type == "Dip":
                            st.metric("Avg Torso Lean", f"{avg_torso_lean:.1f}°")
                        else:
                            st.metric("Avg Body Angle", f"{avg_body_angle:.1f}°")
                    
                    # Form score over time
                    st.subheader("📈 Form Score Over Time")
                    import pandas as pd
                    
                    df = pd.DataFrame({
                        'Frame': frame_numbers,
                        'Form Score': [a['form_score'] for a in all_analyses],
                        'Stage': [a['stage'] for a in all_analyses]
                    })
                    
                    st.line_chart(df.set_index('Frame')['Form Score'], use_container_width=True)
                    
                    # Angle measurements over time
                    st.subheader("📐 Angle Measurements")
                    
                    if exercise_type == "Squat":
                        angle_df = pd.DataFrame({
                            'Frame': frame_numbers,
                            'Knee Angle': [a['knee_angle'] for a in all_analyses],
                            'Hip Angle': [a['hip_angle'] for a in all_analyses],
                            'Body Lean': [a['body_lean'] for a in all_analyses]
                        })
                        st.line_chart(angle_df.set_index('Frame'), use_container_width=True)
                    elif exercise_type == "Pull-up":
                        angle_df = pd.DataFrame({
                            'Frame': frame_numbers,
                            'Elbow Angle': [a['elbow_angle'] for a in all_analyses],
                            'Body Swing': [a['body_swing'] for a in all_analyses]
                        })
                        st.line_chart(angle_df.set_index('Frame'), use_container_width=True)
                    elif exercise_type == "Dip":
                        angle_df = pd.DataFrame({
                            'Frame': frame_numbers,
                            'Elbow Angle': [a['elbow_angle'] for a in all_analyses],
                            'Torso Lean': [a['torso_lean'] for a in all_analyses]
                        })
                        st.line_chart(angle_df.set_index('Frame'), use_container_width=True)
                    else:
                        angle_df = pd.DataFrame({
                            'Frame': frame_numbers,
                            'Elbow Angle': [a['elbow_angle'] for a in all_analyses],
                            'Body Angle': [a['body_angle'] for a in all_analyses]
                        })
                        st.line_chart(angle_df.set_index('Frame'), use_container_width=True)
                    
                    # Common errors
                    st.subheader("⚠️ Most Common Issues")
                    all_errors = []
                    for analysis in all_analyses:
                        all_errors.extend(analysis['errors'])
                    
                    if all_errors:
                        from collections import Counter
                        error_counts = Counter(all_errors)
                        
                        for error, count in error_counts.most_common(5):
                            percentage = (count / len(all_analyses)) * 100
                            st.warning(f"**{error}** - Occurred in {percentage:.1f}% of frames ({count}/{len(all_analyses)})")
                    
                    # Best and worst frames
                    st.subheader("📸 Best vs Worst Form")
                    col1, col2 = st.columns(2)
                    
                    # Find best frame
                    best_idx = np.argmax([a['form_score'] for a in all_analyses])
                    best_frame_num = frame_numbers[best_idx]
                    best_score = all_analyses[best_idx]['form_score']
                    
                    # Find worst frame
                    worst_idx = np.argmin([a['form_score'] for a in all_analyses])
                    worst_frame_num = frame_numbers[worst_idx]
                    worst_score = all_analyses[worst_idx]['form_score']
                    
                    with col1:
                        st.success(f"**Best Form** (Frame {best_frame_num})")
                        st.metric("Score", f"{best_score:.1f}%")
                        if show_annotated and annotated_frames:
                            st.image(annotated_frames[best_idx], use_container_width=True)
                    
                    with col2:
                        st.error(f"**Worst Form** (Frame {worst_frame_num})")
                        st.metric("Score", f"{worst_score:.1f}%")
                        if show_annotated and annotated_frames:
                            st.image(annotated_frames[worst_idx], use_container_width=True)
                        st.write("**Issues:**")
                        for error in all_analyses[worst_idx]['errors']:
                            st.write(f"- {error}")
                    
                    # Show sample annotated frames
                    if show_annotated and annotated_frames:
                        st.subheader("📹 Sample Annotated Frames")
                        
                        # Show frames at different points
                        num_samples = min(6, len(annotated_frames))
                        sample_indices = np.linspace(0, len(annotated_frames)-1, num_samples, dtype=int)
                        
                        cols = st.columns(3)
                        for i, idx in enumerate(sample_indices):
                            with cols[i % 3]:
                                st.image(annotated_frames[idx], 
                                       caption=f"Frame {frame_numbers[idx]} - Score: {all_analyses[idx]['form_score']:.0f}%",
                                       use_container_width=True)
                    
                    # Overall recommendations
                    st.subheader("💡 Recommendations")
                    
                    if avg_form_score >= 85:
                        st.success("🌟 **Excellent form overall!** Keep up the great work.")
                    elif avg_form_score >= 70:
                        st.info("👍 **Good form!** Focus on the issues mentioned above for improvement.")
                    else:
                        st.warning("⚠️ **Form needs attention.** Review the common issues and work on corrections.")
                    
                    # Specific recommendations based on exercise
                    if exercise_type == "Squat":
                        if avg_knee_angle > 100:
                            st.warning("💡 Try to squat deeper - aim for thighs parallel to ground")
                        if avg_body_lean > 40:
                            st.warning("💡 Keep your chest more upright - engage your core")
                    elif exercise_type == "Push-up":
                        if avg_elbow_angle > 100:
                            st.warning("💡 Lower your chest closer to the ground")
                        if avg_body_angle < 165:
                            st.warning("💡 Engage your core to prevent hip sagging")
                    elif exercise_type == "Pull-up":
                        if avg_elbow_angle > 70:
                            st.warning("💡 Pull higher - aim to get your chin above the bar")
                        if avg_body_swing > 15:
                            st.warning("💡 Control your swing - use strict form")
                    elif exercise_type == "Dip":
                        if avg_elbow_angle > 100:
                            st.warning("💡 Lower deeper - aim for 90° elbow angle")
                        if avg_torso_lean > 30:
                            st.warning("💡 Keep your torso more upright - reduce forward lean")
                
                else:
                    st.error("❌ No pose detected in the video. Make sure your full body is visible.")

with tab3:
    st.header("🖥️ Live Webcam Analysis")
    st.info("📹 Click 'START' to begin webcam analysis. Make sure your full body is visible in the frame.")
    
    exercise_type = st.session_state.get('exercise_type', 'Squat')
    
    class ExerciseVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.pose_detector = PoseDetector()
            self.rep_count = 0
            self.last_stage = 'up'
            self.exercise_type = exercise_type
            
            if self.exercise_type.lower() == 'squat':
                self.form_checker = SquatFormChecker()
            elif self.exercise_type.lower() == 'pull-up':
                self.form_checker = PullUpFormChecker()
            elif self.exercise_type.lower() == 'push-up':
                self.form_checker = PushUpFormChecker()
            elif self.exercise_type.lower() == 'dip':
                self.form_checker = DipFormChecker()
            else:
                self.form_checker = SquatFormChecker()
        
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Detect pose
            landmarks, _ = self.pose_detector.detect_pose(img)
            
            if landmarks is None:
                # No pose detected - show message
                cv2.putText(img, "No pose detected - step back to show full body", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Build landmarks dictionary with normalized coordinates (0-1 range)
            landmarks_dict = {}
            for name, idx in self.pose_detector.LANDMARK_NAMES.items():
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    landmarks_dict[name] = {
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility if hasattr(lm, 'visibility') else 1.0
                    }
            
            # Analyze form
            analysis = self.form_checker.analyze_frame(landmarks_dict)
            
            # Update rep count
            current_stage = analysis['stage']
            if self.last_stage == 'down' and current_stage == 'up':
                self.rep_count += 1
            self.last_stage = current_stage
            analysis['rep_count'] = self.rep_count
            
            # Draw pose
            annotated = self.pose_detector.draw_landmarks(img, landmarks)
            
            # Draw analysis info
            h, w = annotated.shape[:2]
            
            # Top left info box - Exercise and Reps
            box_height = 120
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, 10), (320, box_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
            
            cv2.putText(annotated, f"Exercise: {self.exercise_type.upper()}",
                       (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Reps: {self.rep_count}", 
                       (20, 75), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Stage: {current_stage.upper()}", 
                       (20, 105), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            
            # Top right - Form score
            score = analysis['form_score']
            score_color = (0, 255, 0) if score >= 80 else (0, 165, 255) if score >= 60 else (0, 0, 255)
            overlay2 = annotated.copy()
            cv2.rectangle(overlay2, (w - 220, 10), (w - 10, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.7, annotated, 0.3, 0, annotated)
            
            cv2.putText(annotated, f"Form Score", 
                       (w - 205, 35), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"{score}%", 
                       (w - 205, 65), cv2.FONT_HERSHEY_DUPLEX, 1.2, score_color, 2, cv2.LINE_AA)
            
            # Feedback messages
            y_offset = 140
            for i, feedback in enumerate(analysis['feedback'][:3]):
                overlay3 = annotated.copy()
                cv2.rectangle(overlay3, (10, y_offset - 20), (min(w - 10, 600), y_offset + 8), (0, 0, 0), -1)
                cv2.addWeighted(overlay3, 0.6, annotated, 0.4, 0, annotated)
                cv2.putText(annotated, feedback, (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 255), 1, cv2.LINE_AA)
                y_offset += 35
            
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
    
    # Display form guide
    display_exercise_info(exercise_type)
    
    # IMPORTANT: Set session state BEFORE creating webrtc_streamer
    st.session_state.exercise_type = exercise_type
    
    # WebRTC configuration
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Create centered column for webcam
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Start webcam stream with smaller size
        webrtc_ctx = webrtc_streamer(
            key=f"exercise-analysis-{exercise_type.lower()}",
            video_processor_factory=ExerciseVideoProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720}
                }, 
                "audio": False
            },
            async_processing=True,
        )
    
    st.markdown("""
    **Instructions:**
    1. Click the **START** button above to activate your webcam
    2. Allow browser access to your camera when prompted
    3. Position yourself so your full body is visible
    4. Start exercising and watch the real-time feedback!
    """)
