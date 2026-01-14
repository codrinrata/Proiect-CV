# 💪 Exercise Form Analyzer

A real-time computer vision project that analyzes exercise form for **squats**, **push-ups**, **pull-ups**, and **dips** using pose estimation and biomechanical analysis.

## 🎯 Project Overview

This project uses **MediaPipe Pose** and computer vision techniques to:
1. Detect human body pose in real-time from webcam or images
2. Calculate joint angles and body measurements
3. Verify exercise form against biomechanical standards
4. Provide instant feedback on form errors
5. Count reps automatically

**Perfect for university computer vision courses** - demonstrates practical CV applications in sports and fitness!

## 🔬 Computer Vision Techniques Used

### Pose Estimation
- **MediaPipe Pose**: State-of-the-art ML model for body landmark detection
- **33 Landmark Detection**: Full body skeletal tracking
- **Real-time Processing**: Optimized for live video (30+ FPS)
- **3D Coordinates**: X, Y, Z position for each landmark

### Geometric Analysis
- **Angle Calculation**: Using vector mathematics and dot products
- **Joint Angle Measurement**: Knee, hip, elbow, body angles
- **Distance Calculation**: Euclidean distance between landmarks
- **Alignment Checking**: Vertical and horizontal body alignment
- **Body Lean Analysis**: Forward/backward tilt measurement

### Form Verification
- **Rule-based System**: Biomechanically sound criteria
- **Threshold Detection**: Angle ranges for proper form
- **Symmetry Analysis**: Left/right side comparison
- **Stage Detection**: Up/down position tracking
- **Rep Counting**: Automatic based on joint angles

## 📁 Project Structure

```
laundry-symbol-detector/  (renamed for exercise analysis)
├── app.py                          # Streamlit web interface
├── requirements.txt                # Python dependencies
├── analyze_video.py                # CLI video processor
├── test_pose.py                    # Test script
├── models/
│   └── pose_landmarker_heavy.task # MediaPipe model (31MB)
├── data/
│   └── symbol_database.json       # Exercise form criteria
├── src/
│   ├── pose_detection.py          # MediaPipe pose detector (Tasks API)
│   ├── squat_checker.py           # Squat form analyzer
│   ├── pushup_checker.py          # Push-up form analyzer
│   ├── pullup_checker.py          # Pull-up form analyzer
│   ├── dip_checker.py             # Dip form analyzer
│   ├── video_analyzer.py          # Real-time video processing
│   └── utils/
│       ├── angle_utils.py         # Angle calculation functions
│       └── visualization.py        # Drawing utilities
├── models/                         # (Saved features if needed)
├── notebooks/                      # Jupyter notebooks for analysis
└── outputs/                        # Saved analysis videos
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Webcam (for real-time analysis)
- OpenCV 4.8+
- MediaPipe 0.10+

### Installation

1. Navigate to the project directory:
```bash
cd "Proiect CV/laundry-symbol-detector"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

#### Option 1: Web Interface (Streamlit)
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

Features:
- Upload exercise photos for analysis
- Get detailed form feedback
- See pose landmarks visualized
- View angle measurements

#### Option 2: Real-time Webcam (Recommended)
```bash
cd src
python video_analyzer.py squat    # For squat analysis
python video_analyzer.py pushup   # For push-up analysis
python video_analyzer.py pullup   # For pull-up analysis
python video_analyzer.py dip      # For dip analysis
```

Controls:
- **'q'**: Quit
- **'r'**: Reset rep counter

## 💪 Supported Exercises

### 🏋️ Squats

**What We Check:**
- ✓ Knee angle (depth)
- ✓ Knee alignment (not over toes)
- ✓ Hip angle
- ✓ Forward lean
- ✓ Stance width
- ✓ Symmetry (left vs right)

**Form Criteria:**
- Knees should reach ~70-90° at bottom
- Knees stay behind toes
- Chest stays up (< 45° forward lean)
- Feet shoulder-width apart
- Even weight distribution

### 💪 Push-ups

**What We Check:**
- ✓ Elbow angle (depth)
- ✓ Body alignment (straight line)
- ✓ Hip position (no sagging/piking)
- ✓ Elbow flare angle
- ✓ Hand placement width
- ✓ Head/neck position

**Form Criteria:**
- Elbows reach ~70-90° at bottom
- Body angle stays ~180° (straight)
- Hands shoulder-width apart
- Elbows at ~45° to body
- Full range of motion

### 🤸 Pull-ups

**What We Check:**
- ✓ Elbow angle (full range)
- ✓ Chin height (above bar)
- ✓ Body swing control
- ✓ Full arm extension
- ✓ Shoulder symmetry

**Form Criteria:**
- Chin clears the bar at top
- Full extension at bottom (>160° elbows)
- Minimal swinging/kipping (< 15°)
- Controlled movement
- Even shoulder activation

### 🔽 Dips

**What We Check:**
- ✓ Depth (elbow angle)
- ✓ Elbow position (tucked)
- ✓ Torso lean
- ✓ Full extension at top
- ✓ Shoulder symmetry

**Form Criteria:**
- Elbows reach ~90° at bottom
- Elbows stay close to body
- Moderate forward lean (< 30°)
- Full lockout at top
- Controlled descent and ascent

## 📊 Output & Feedback

The system provides:

1. **Real-time Visual Feedback**
   - Pose skeleton overlay
   - Joint angle displays
   - Form score (0-100%)
   - Stage indicator (up/down)

2. **Specific Form Corrections**
   - "⚠️ Not deep enough"
   - "⚠️ Keep knees behind toes"
   - "⚠️ Hips sagging"
   - "✓ Good depth"
   - "✓ Good posture"

3. **Metrics**
   - Rep counter
   - Current angles
   - Form score percentage
   - FPS (frames per second)

## 🎓 Educational Value

This project demonstrates:

### Computer Vision Concepts
- **Pose Estimation**: How ML models detect human poses
- **Landmark Detection**: Identifying body keypoints
- **Real-time Processing**: Optimizing CV algorithms for video
- **Coordinate Systems**: Working with 2D and 3D coordinates

### Mathematical Concepts
- **Vector Mathematics**: Calculating angles from points
- **Trigonometry**: Angle computation using arccos/arctan
- **Euclidean Geometry**: Distance calculations
- **Linear Algebra**: Dot products and normalization

### Applied Biomechanics
- **Joint Angles**: Understanding human movement
- **Form Analysis**: Biomechanically sound criteria
- **Injury Prevention**: Detecting unsafe movement patterns
- **Performance Optimization**: Proper technique analysis

### Software Engineering
- **Modular Design**: Separate concerns (detection, analysis, visualization)
- **Real-time Systems**: Frame processing optimization
- **User Interface**: Clear feedback presentation
- **State Management**: Tracking exercise progress

## 🔧 Customization

### Adjust Form Thresholds

In `src/squat_checker.py`:
```python
self.MIN_KNEE_ANGLE = 70        # Minimum for full squat
self.MAX_FORWARD_LEAN = 45      # Maximum acceptable lean
self.KNEE_OVER_TOE_THRESHOLD = 0.05  # Knee position tolerance
```

In `src/pushup_checker.py`:
```python
self.MIN_ELBOW_ANGLE = 70       # Full push-up depth
self.MAX_HIP_SAG = 20           # Hip sag tolerance
```

### Add New Exercises

1. Create new form checker (e.g., `plank_checker.py`)
2. Define landmarks and angles to track
3. Set form criteria and thresholds
4. Add to `video_analyzer.py`

Example structure:
```python
class PlankFormChecker:
    def analyze_frame(self, landmarks):
        # Calculate body angle
        # Check for hip sag
        # Verify elbow position
        # Return analysis dict
```

## 📈 Performance Tips

- **Lighting**: Ensure good, even lighting
- **Camera Position**: Place camera to show full body
- **Background**: Plain background works best
- **Clothing**: Wear fitted clothing for better detection
- **Distance**: Stand 6-10 feet from camera

## 🤝 Contributing

Ideas for expansion:

- [ ] Add more exercises (lunges, planks, pull-ups)
- [ ] Implement workout tracking/history
- [ ] Add voice feedback
- [ ] Create exercise tutorials
- [ ] Build mobile app version
- [ ] Add rep quality scoring
- [ ] Implement workout plans

## 📄 License

This project is created for educational purposes.

## 🙏 Acknowledgments

- MediaPipe team for pose estimation model
- OpenCV community
- Computer Vision course materials
- Sports science and biomechanics research

## 📬 Support

For questions or issues, please open an issue on the repository.

---

**Built with ❤️ using Computer Vision for fitness and health**