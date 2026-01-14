import cv2
import numpy as np
from typing import Optional, Callable
import time

from .pose_detection import PoseDetector
from .squat_checker import SquatFormChecker
from .pushup_checker import PushUpFormChecker
from .pullup_checker import PullUpFormChecker
from .dip_checker import DipFormChecker


class ExerciseAnalyzer:
    
    def __init__(self, exercise_type: str = 'squat'):
        self.exercise_type = exercise_type
        self.pose_detector = PoseDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        if exercise_type == 'squat':
            self.form_checker = SquatFormChecker()
        elif exercise_type == 'pushup' or exercise_type == 'push-up':
            self.form_checker = PushUpFormChecker()
        elif exercise_type == 'pullup' or exercise_type == 'pull-up':
            self.form_checker = PullUpFormChecker()
        elif exercise_type == 'dip':
            self.form_checker = DipFormChecker()
        else:
            raise ValueError(f"Unknown exercise type: {exercise_type}")
        
        self.fps = 0
        self.last_time = time.time()
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        landmarks, _ = self.pose_detector.detect_pose(frame)
        
        if landmarks is None:
            return frame, None
        
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
        
        analysis = self.form_checker.analyze_frame(landmarks_dict)
        annotated_frame = self.pose_detector.draw_landmarks(frame, landmarks)
        annotated_frame = self._draw_annotations(annotated_frame, analysis, landmarks_dict)
        
        current_time = time.time()
        self.fps = 1 / (current_time - self.last_time)
        self.last_time = current_time
        
        return annotated_frame, analysis
    
    def _draw_annotations(self, frame: np.ndarray, analysis: dict, landmarks: dict) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # Draw rep counter
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Reps: {analysis['rep_count']}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, f"Stage: {analysis['stage'].upper()}", 
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw form score
        score = analysis['form_score']
        score_color = self._get_score_color(score)
        cv2.rectangle(frame, (w - 200, 10), (w - 10, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"Form: {score}%", 
                   (w - 190, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 2)
        
        # Draw feedback messages
        y_offset = 120
        for i, feedback in enumerate(analysis['feedback'][:5]):  # Show max 5 messages
            # Background
            (text_w, text_h), _ = cv2.getTextSize(
                feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (10, y_offset - 25), 
                         (20 + text_w, y_offset + 5), (0, 0, 0), -1)
            
            # Text color based on feedback type
            color = (0, 255, 0) if '✓' in feedback else (0, 165, 255)
            cv2.putText(frame, feedback, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 35
        
        # Draw key angles
        if self.exercise_type == 'squat':
            angle_text = f"Knee: {analysis['knee_angle']:.1f}°"
            cv2.putText(frame, angle_text, (w - 200, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            lean_text = f"Lean: {analysis['body_lean']:.1f}°"
            cv2.putText(frame, lean_text, (w - 200, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        elif self.exercise_type in ['pushup', 'push-up']:
            angle_text = f"Elbow: {analysis['elbow_angle']:.1f}°"
            cv2.putText(frame, angle_text, (w - 200, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            body_text = f"Body: {analysis['body_angle']:.1f}°"
            cv2.putText(frame, body_text, (w - 200, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        elif self.exercise_type in ['pullup', 'pull-up']:
            angle_text = f"Elbow: {analysis['elbow_angle']:.1f}°"
            cv2.putText(frame, angle_text, (w - 200, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            swing_text = f"Swing: {analysis['body_swing']:.1f}°"
            cv2.putText(frame, swing_text, (w - 200, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        elif self.exercise_type == 'dip':
            angle_text = f"Elbow: {analysis['elbow_angle']:.1f}°"
            cv2.putText(frame, angle_text, (w - 200, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            lean_text = f"Lean: {analysis['torso_lean']:.1f}°"
            cv2.putText(frame, lean_text, (w - 200, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _get_score_color(self, score: float) -> tuple:
        """Get color based on form score."""
        if score >= 80:
            return (0, 255, 0)  # Green
        elif score >= 60:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 0, 255)  # Red
    
    def run_webcam(self, camera_id: int = 0, 
                   on_frame: Optional[Callable] = None):
        """
        Run analysis on webcam feed.
        
        Args:
            camera_id: Camera device ID
            on_frame: Optional callback function for each frame
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"Starting {self.exercise_type} analysis...")
        print("Press 'q' to quit, 'r' to reset counter")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Process frame
            annotated_frame, analysis = self.process_frame(frame)
            
            # Call custom callback if provided
            if on_frame and analysis:
                on_frame(annotated_frame, analysis)
            
            # Display
            cv2.imshow(f'{self.exercise_type.title()} Form Analyzer', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.form_checker.reset_counter()
                print("Rep counter reset")
        
        cap.release()
        cv2.destroyAllWindows()
        self.pose_detector.close()
    
    def process_video(self, video_path: str, output_path: Optional[str] = None):
        """
        Process a video file.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, analysis = self.process_frame(frame)
            
            # Write to output if specified
            if writer:
                writer.write(annotated_frame)
            
            # Display (optional)
            cv2.imshow('Processing...', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if writer:
            writer.release()
            print(f"Saved output to: {output_path}")
        
        cv2.destroyAllWindows()
        self.pose_detector.close()
    
    def reset(self):
        """Reset the analyzer."""
        self.form_checker.reset_counter()
    
    def change_exercise(self, exercise_type: str):
        """Change the exercise type."""
        self.exercise_type = exercise_type
        if exercise_type == 'squat':
            self.form_checker = SquatFormChecker()
        elif exercise_type == 'pushup':
            self.form_checker = PushUpFormChecker()
        else:
            raise ValueError(f"Unknown exercise type: {exercise_type}")


def main():
    """Run standalone exercise analyzer."""
    import sys
    
    exercise = sys.argv[1] if len(sys.argv) > 1 else 'squat'
    
    if exercise not in ['squat', 'pushup']:
        print("Usage: python video_analyzer.py [squat|pushup]")
        sys.exit(1)
    
    analyzer = ExerciseAnalyzer(exercise_type=exercise)
    analyzer.run_webcam()


if __name__ == '__main__':
    main()
