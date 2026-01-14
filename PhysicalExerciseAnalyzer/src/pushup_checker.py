import numpy as np
from typing import Dict, List, Tuple
from .utils.angle_utils import (
    calculate_angle, calculate_horizontal_alignment,
    calculate_distance, calculate_body_lean
)


class PushUpFormChecker:
    
    def __init__(self):
        self.rep_count = 0
        self.current_stage = "up"
        self.feedback_messages = []
        
        self.MIN_ELBOW_ANGLE = 70
        self.MAX_ELBOW_ANGLE = 160
        self.MAX_HIP_SAG = 20
        self.MAX_HIP_PIKE = 15
        self.MIN_DEPTH_RATIO = 0.7
        
    def analyze_frame(self, landmarks: Dict[str, Dict]) -> Dict:
        self.feedback_messages = []
        
        # Extract key landmarks
        left_shoulder = (landmarks['left_shoulder']['x'], landmarks['left_shoulder']['y'])
        right_shoulder = (landmarks['right_shoulder']['x'], landmarks['right_shoulder']['y'])
        left_elbow = (landmarks['left_elbow']['x'], landmarks['left_elbow']['y'])
        right_elbow = (landmarks['right_elbow']['x'], landmarks['right_elbow']['y'])
        left_wrist = (landmarks['left_wrist']['x'], landmarks['left_wrist']['y'])
        right_wrist = (landmarks['right_wrist']['x'], landmarks['right_wrist']['y'])
        left_hip = (landmarks['left_hip']['x'], landmarks['left_hip']['y'])
        right_hip = (landmarks['right_hip']['x'], landmarks['right_hip']['y'])
        left_knee = (landmarks['left_knee']['x'], landmarks['left_knee']['y'])
        right_knee = (landmarks['right_knee']['x'], landmarks['right_knee']['y'])
        left_ankle = (landmarks['left_ankle']['x'], landmarks['left_ankle']['y'])
        right_ankle = (landmarks['right_ankle']['x'], landmarks['right_ankle']['y'])
        
        # Calculate angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        
        # Calculate body alignment (should be straight line)
        avg_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2,
                       (left_shoulder[1] + right_shoulder[1]) / 2)
        avg_hip = ((left_hip[0] + right_hip[0]) / 2,
                  (left_hip[1] + right_hip[1]) / 2)
        avg_ankle = ((left_ankle[0] + right_ankle[0]) / 2,
                    (left_ankle[1] + right_ankle[1]) / 2)
        
        # Calculate hip angle (should be ~180 for straight body)
        left_body_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
        right_body_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
        body_angle = (left_body_angle + right_body_angle) / 2
        
        # Determine push-up stage and count reps
        previous_stage = self.current_stage
        
        # Down position: elbows bent significantly
        if elbow_angle < 95:
            self.current_stage = "down"
        # Up position: elbows extended
        elif elbow_angle > 150:
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
            if elbow_angle > self.MIN_ELBOW_ANGLE + 30:
                errors.append("Go lower! Elbows should reach 90 degrees")
                self.feedback_messages.append("⚠️ Not deep enough")
                form_score -= 25
            else:
                self.feedback_messages.append("✓ Good depth")
        
        # Check body alignment (straight line from head to heels)
        if body_angle < 160:  # Hip sagging
            hip_sag = 180 - body_angle
            errors.append(f"Hips sagging ({hip_sag:.1f}°) - engage your core!")
            self.feedback_messages.append("⚠️ Hips sagging")
            form_score -= 30
        elif body_angle > 195:  # Hip piking (too high)
            errors.append("Hips too high - keep body straight")
            self.feedback_messages.append("⚠️ Hips too high")
            form_score -= 20
        else:
            self.feedback_messages.append("✓ Good body alignment")
        
        # Check elbow flare (should be close to body, not too wide)
        elbow_flare = self._check_elbow_flare(
            (left_shoulder, right_shoulder),
            (left_elbow, right_elbow)
        )
        if elbow_flare > 60:
            errors.append("Elbows too wide - keep them closer to body")
            self.feedback_messages.append("⚠️ Elbows flaring")
            form_score -= 15
        else:
            self.feedback_messages.append("✓ Good elbow position")
        
        # Check hand placement (should be shoulder-width)
        hand_width = abs(left_wrist[0] - right_wrist[0])
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        
        if hand_width < shoulder_width * 0.7:
            errors.append("Hands too narrow")
            self.feedback_messages.append("⚠️ Hands too close")
            form_score -= 10
        elif hand_width > shoulder_width * 1.4:
            errors.append("Hands too wide")
            self.feedback_messages.append("⚠️ Hands too wide")
            form_score -= 10
        else:
            self.feedback_messages.append("✓ Good hand width")
        
        # Check symmetry
        angle_diff = abs(left_elbow_angle - right_elbow_angle)
        if angle_diff > 20:
            errors.append("Uneven arm bend - maintain symmetry")
            self.feedback_messages.append("⚠️ Uneven form")
            form_score -= 15
        
        # Check head position (should be neutral)
        nose = (landmarks['nose']['x'], landmarks['nose']['y'])
        if abs(nose[1] - avg_shoulder[1]) > 0.15:  # Head too far from shoulders
            errors.append("Keep neck neutral - don't look up or down")
            self.feedback_messages.append("⚠️ Head position")
            form_score -= 10
        
        form_score = max(0, form_score)
        
        return {
            'elbow_angle': elbow_angle,
            'body_angle': body_angle,
            'stage': self.current_stage,
            'rep_count': self.rep_count,
            'form_score': form_score,
            'errors': errors,
            'feedback': self.feedback_messages,
            'angles': {
                'left_elbow': left_elbow_angle,
                'right_elbow': right_elbow_angle,
                'left_body': left_body_angle,
                'right_body': right_body_angle
            }
        }
    
    def _check_elbow_flare(self, shoulders: Tuple, elbows: Tuple) -> float:
        """
        Calculate elbow flare angle.
        
        Args:
            shoulders: Tuple of (left_shoulder, right_shoulder)
            elbows: Tuple of (left_elbow, right_elbow)
        
        Returns:
            Average elbow flare angle
        """
        left_shoulder, right_shoulder = shoulders
        left_elbow, right_elbow = elbows
        
        # Calculate angle between shoulder-elbow line and vertical
        left_flare = abs(np.arctan2(
            left_elbow[0] - left_shoulder[0],
            left_elbow[1] - left_shoulder[1]
        ) * 180 / np.pi)
        
        right_flare = abs(np.arctan2(
            right_elbow[0] - right_shoulder[0],
            right_elbow[1] - right_shoulder[1]
        ) * 180 / np.pi)
        
        return (left_flare + right_flare) / 2
    
    def reset_counter(self):
        """Reset the rep counter."""
        self.rep_count = 0
        self.current_stage = "up"
    
    def get_tips(self) -> List[str]:
        """Get general push-up form tips."""
        return [
            "Keep hands shoulder-width apart",
            "Maintain straight line from head to heels",
            "Engage your core throughout",
            "Lower until chest nearly touches ground",
            "Keep elbows at 45° angle to body",
            "Don't let hips sag or pike up",
            "Keep neck neutral (don't look up)",
            "Push through palms to full extension"
        ]
