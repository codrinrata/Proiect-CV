import numpy as np
from typing import Dict, List, Tuple
from .utils.angle_utils import (
    calculate_angle, calculate_vertical_alignment,
    calculate_horizontal_alignment, calculate_body_lean
)


class SquatFormChecker:
    
    def __init__(self):
        self.rep_count = 0
        self.current_stage = "up"
        self.feedback_messages = []
        
        self.MIN_KNEE_ANGLE = 70
        self.MAX_KNEE_ANGLE = 160
        self.MIN_HIP_ANGLE = 60
        self.MAX_FORWARD_LEAN = 45
        self.KNEE_OVER_TOE_THRESHOLD = 0.05
        
    def analyze_frame(self, landmarks: Dict[str, Dict]) -> Dict:
        self.feedback_messages = []
        
        # Extract key landmarks
        left_shoulder = (landmarks['left_shoulder']['x'], landmarks['left_shoulder']['y'])
        right_shoulder = (landmarks['right_shoulder']['x'], landmarks['right_shoulder']['y'])
        left_hip = (landmarks['left_hip']['x'], landmarks['left_hip']['y'])
        right_hip = (landmarks['right_hip']['x'], landmarks['right_hip']['y'])
        left_knee = (landmarks['left_knee']['x'], landmarks['left_knee']['y'])
        right_knee = (landmarks['right_knee']['x'], landmarks['right_knee']['y'])
        left_ankle = (landmarks['left_ankle']['x'], landmarks['left_ankle']['y'])
        right_ankle = (landmarks['right_ankle']['x'], landmarks['right_ankle']['y'])
        
        # Calculate angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        hip_angle = (left_hip_angle + right_hip_angle) / 2
        
        # Calculate body lean
        avg_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2,
                       (left_shoulder[1] + right_shoulder[1]) / 2)
        avg_hip = ((left_hip[0] + right_hip[0]) / 2,
                  (left_hip[1] + right_hip[1]) / 2)
        body_lean = calculate_body_lean(avg_shoulder, avg_hip)
        
        # Determine squat stage and count reps
        # Use both knee and hip angles for better detection
        previous_stage = self.current_stage
        
        # Down position: knees bent significantly OR hips lowered
        if knee_angle < 100 or hip_angle < 100:
            self.current_stage = "down"
        # Up position: both knees and hips extended
        elif knee_angle > 140 and hip_angle > 140:
            self.current_stage = "up"
        # Else: maintain current stage (transition zone)
        
        # Count rep when transitioning from down to up
        if previous_stage == "down" and self.current_stage == "up":
            self.rep_count += 1
        
        # Analyze form and provide feedback
        form_score = 100
        errors = []
        
        # Check depth
        if self.current_stage == "down":
            if knee_angle > self.MIN_KNEE_ANGLE + 20:
                errors.append("Go deeper! Squat to at least parallel")
                self.feedback_messages.append("⚠️ Not deep enough")
                form_score -= 20
            else:
                self.feedback_messages.append("✓ Good depth")
        
        # Check knee alignment
        knee_over_toe = self._check_knee_over_toe(
            (left_knee, right_knee),
            (left_ankle, right_ankle)
        )
        if knee_over_toe:
            errors.append("Knees too far forward over toes")
            self.feedback_messages.append("⚠️ Keep knees behind toes")
            form_score -= 25
        else:
            self.feedback_messages.append("✓ Good knee alignment")
        
        # Check forward lean
        if body_lean > self.MAX_FORWARD_LEAN:
            errors.append(f"Too much forward lean ({body_lean:.1f}°)")
            self.feedback_messages.append("⚠️ Keep chest up")
            form_score -= 20
        else:
            self.feedback_messages.append("✓ Good posture")
        
        # Check feet alignment (should be shoulder-width)
        feet_width = abs(left_ankle[0] - right_ankle[0])
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        
        if feet_width < shoulder_width * 0.8:
            errors.append("Feet too narrow - widen your stance")
            self.feedback_messages.append("⚠️ Widen stance")
            form_score -= 15
        elif feet_width > shoulder_width * 1.5:
            errors.append("Feet too wide")
            self.feedback_messages.append("⚠️ Narrow stance")
            form_score -= 10
        else:
            self.feedback_messages.append("✓ Good stance width")
        
        # Check symmetry
        angle_diff = abs(left_knee_angle - right_knee_angle)
        if angle_diff > 15:
            errors.append("Uneven knee bend - maintain symmetry")
            self.feedback_messages.append("⚠️ Uneven form")
            form_score -= 15
        
        form_score = max(0, form_score)
        
        return {
            'knee_angle': knee_angle,
            'hip_angle': hip_angle,
            'body_lean': body_lean,
            'stage': self.current_stage,
            'rep_count': self.rep_count,
            'form_score': form_score,
            'errors': errors,
            'feedback': self.feedback_messages,
            'angles': {
                'left_knee': left_knee_angle,
                'right_knee': right_knee_angle,
                'left_hip': left_hip_angle,
                'right_hip': right_hip_angle
            }
        }
    
    def _check_knee_over_toe(self, knees: Tuple, ankles: Tuple) -> bool:
        """
        Check if knees extend too far over toes.
        
        Args:
            knees: Tuple of (left_knee, right_knee) coordinates
            ankles: Tuple of (left_ankle, right_ankle) coordinates
        
        Returns:
            True if knees are too far forward
        """
        left_knee, right_knee = knees
        left_ankle, right_ankle = ankles
        
        # Check if knee x-position is significantly beyond ankle
        left_over = left_knee[0] - left_ankle[0]
        right_over = right_knee[0] - right_ankle[0]
        
        return (abs(left_over) > self.KNEE_OVER_TOE_THRESHOLD or 
                abs(right_over) > self.KNEE_OVER_TOE_THRESHOLD)
    
    def reset_counter(self):
        """Reset the rep counter."""
        self.rep_count = 0
        self.current_stage = "up"
    
    def get_tips(self) -> List[str]:
        """Get general squat form tips."""
        return [
            "Keep your feet shoulder-width apart",
            "Point toes slightly outward",
            "Keep your chest up and core tight",
            "Push through your heels",
            "Keep knees in line with toes",
            "Lower until thighs are parallel to ground",
            "Don't let knees cave inward",
            "Maintain neutral spine"
        ]
