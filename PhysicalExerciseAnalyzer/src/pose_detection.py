import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


class PoseDetector:
    
    def __init__(self, 
                 model_path: str = None,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(os.path.dirname(current_dir), 'models', 'pose_landmarker_heavy.task')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        
        self.LANDMARK_NAMES = {
            'nose': 0,
            'left_eye_inner': 1,
            'left_eye': 2,
            'left_eye_outer': 3,
            'right_eye_inner': 4,
            'right_eye': 5,
            'right_eye_outer': 6,
            'left_ear': 7,
            'right_ear': 8,
            'mouth_left': 9,
            'mouth_right': 10,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_pinky': 17,
            'right_pinky': 18,
            'left_index': 19,
            'right_index': 20,
            'left_thumb': 21,
            'right_thumb': 22,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot_index': 31,
            'right_foot_index': 32
        }
        
        self.POSE_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),
            (11, 12),
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
        ]
    
    def detect_pose(self, image: np.ndarray) -> Tuple[Optional[List], np.ndarray]:
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.detector.detect(mp_image)
        
        landmarks = None
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
        
        return landmarks, image.copy()
    
    def draw_landmarks(self, image: np.ndarray, landmarks, draw_connections: bool = True) -> np.ndarray:
        
        annotated_image = image.copy()
        if landmarks is None:
            return annotated_image
        
        h, w = image.shape[:2]
        
        # Draw connections
        if draw_connections:
            for connection in self.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    
                    start_x = int(start_point.x * w)
                    start_y = int(start_point.y * h)
                    end_x = int(end_point.x * w)
                    end_y = int(end_point.y * h)
                    
                    cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), 
                            (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)
        
        return annotated_image
    
    def get_landmark_coords(self, 
                           landmarks, 
                           landmark_name: str,
                           image_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Get pixel coordinates of a specific landmark.
        
        Args:
            landmarks: Pose landmarks from MediaPipe
            landmark_name: Name of the landmark (e.g., 'left_knee')
            image_shape: Shape of the image (height, width)
            
        Returns:
            Tuple of (x, y) pixel coordinates or None if landmark not found
        """
        if not landmarks or landmark_name not in self.LANDMARK_NAMES:
            return None
        
        idx = self.LANDMARK_NAMES[landmark_name]
        if idx >= len(landmarks):
            return None
        
        landmark = landmarks[idx]
        h, w = image_shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        
        return (x, y)
    
    def get_landmark(self, landmarks, landmark_name: str) -> Optional[Dict]:
        """
        Get a landmark by name.
        
        Args:
            landmarks: Pose landmarks from MediaPipe
            landmark_name: Name of the landmark
            
        Returns:
            Dictionary with x, y, z, visibility or None
        """
        if not landmarks or landmark_name not in self.LANDMARK_NAMES:
            return None
        
        idx = self.LANDMARK_NAMES[landmark_name]
        if idx >= len(landmarks):
            return None
        
        landmark = landmarks[idx]
        return {
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1.0
        }
    
    def close(self):
        """Release resources."""
        self.detector.close()
