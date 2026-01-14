import numpy as np
from typing import Dict, List, Tuple
from .utils.angle_utils import (
    calculate_angle, calculate_vertical_alignment,
    calculate_distance, calculate_body_lean
)


class DipFormChecker:
    
    def __init__(self):
        self.rep_count = 0
        self.current_stage = "up"
        self.feedback_messages = []
        
        self.MIN_ELBOW_ANGLE = 70
        self.MAX_ELBOW_ANGLE = 160
        self.MAX_FORWARD_LEAN = 30
        self.MIN_DEPTH_ANGLE = 90
        
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
        
        # Calculate elbow angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        
        # Calculate torso lean
        mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                       (left_shoulder[1] + right_shoulder[1]) / 2)
        mid_hip = ((left_hip[0] + right_hip[0]) / 2, 
                  (left_hip[1] + right_hip[1]) / 2)
        torso_lean = abs(calculate_body_lean(mid_shoulder, mid_hip))
        
        # Determine stage
        # Dip: small elbow angle = at bottom (down), large angle = at top (up)
        previous_stage = self.current_stage
        if avg_elbow_angle < 100:
            self.current_stage = "down"
        elif avg_elbow_angle > 150:
            self.current_stage = "up"
        # else: maintain current stage (in transition)
        
        # Count reps when transitioning from down to up
        if previous_stage == "down" and self.current_stage == "up":
            self.rep_count += 1
        
        # Check form
        errors = []
        form_score = 100
        
        # Check depth
        if self.current_stage == "down" and avg_elbow_angle > self.MIN_DEPTH_ANGLE + 15:
            errors.append("Go deeper - elbows should reach 90 degrees")
            self.feedback_messages.append("⚠️ Insufficient depth")
            form_score -= 20
        elif self.current_stage == "down":
            self.feedback_messages.append("✓ Good depth")
        
        # Check elbow position (should stay close to body)
        elbow_width = abs(left_elbow[0] - right_elbow[0])
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        if elbow_width > shoulder_width * 1.3:
            errors.append("Keep elbows closer to body")
            self.feedback_messages.append("⚠️ Elbows flaring out")
            form_score -= 15
        else:
            self.feedback_messages.append("✓ Elbows tucked")
        
        # Check torso lean
        if torso_lean > self.MAX_FORWARD_LEAN:
            errors.append("Excessive forward lean - stay more upright")
            self.feedback_messages.append("⚠️ Too much forward lean")
            form_score -= 15
        else:
            self.feedback_messages.append("✓ Good torso position")
        
        # Check full extension
        if self.current_stage == "up" and avg_elbow_angle < self.MAX_ELBOW_ANGLE - 10:
            errors.append("Fully extend arms at the top")
            self.feedback_messages.append("⚠️ Incomplete extension")
            form_score -= 10
        elif self.current_stage == "up":
            self.feedback_messages.append("✓ Full extension")
        
        # Check shoulder symmetry
        if abs(left_elbow_angle - right_elbow_angle) > 15:
            errors.append("Keep shoulders level")
            self.feedback_messages.append("⚠️ Uneven shoulders")
            form_score -= 10
        else:
            self.feedback_messages.append("✓ Level shoulders")
        
        form_score = max(0, min(100, form_score))
        
        return {
            'rep_count': self.rep_count,
            'stage': self.current_stage,
            'form_score': int(form_score),
            'elbow_angle': avg_elbow_angle,
            'torso_lean': torso_lean,
            'feedback': self.feedback_messages,
            'errors': errors,
            'angles': {
                'left_elbow': left_elbow_angle,
                'right_elbow': right_elbow_angle,
                'torso_lean': torso_lean
            }
        }
