import numpy as np
from typing import Dict, List, Tuple
from .utils.angle_utils import (
    calculate_angle, calculate_vertical_alignment,
    calculate_distance, calculate_body_lean
)


class PullUpFormChecker:
    
    def __init__(self):
        self.rep_count = 0
        self.current_stage = "down"
        self.feedback_messages = []
        
        self.MIN_ELBOW_ANGLE = 60
        self.MAX_ELBOW_ANGLE = 160
        self.MAX_SWING = 15
        self.MIN_CHIN_HEIGHT = 0.9
        
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
        nose = (landmarks['nose']['x'], landmarks['nose']['y'])
        
        # Calculate elbow angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        
        # Calculate body swing (deviation from vertical)
        mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                       (left_shoulder[1] + right_shoulder[1]) / 2)
        mid_hip = ((left_hip[0] + right_hip[0]) / 2, 
                  (left_hip[1] + right_hip[1]) / 2)
        body_swing = abs(calculate_body_lean(mid_shoulder, mid_hip))
        
        # Check chin height relative to hands
        avg_hand_y = (left_wrist[1] + right_wrist[1]) / 2
        chin_above_bar = nose[1] < avg_hand_y
        
        # Determine stage
        # Pull-up: small elbow angle = at top (up), large angle = at bottom (down)
        previous_stage = self.current_stage
        if avg_elbow_angle < 90:
            self.current_stage = "up"
        elif avg_elbow_angle > 140:
            self.current_stage = "down"
        # else: maintain current stage (in transition)
        
        # Count reps when transitioning from down to up
        if previous_stage == "down" and self.current_stage == "up":
            self.rep_count += 1
        
        # Check form
        errors = []
        form_score = 100
        
        # Check elbow angle at top
        if self.current_stage == "up" and avg_elbow_angle > self.MIN_ELBOW_ANGLE + 20:
            errors.append("Pull higher - elbows not bent enough")
            self.feedback_messages.append("⚠️ Pull higher")
            form_score -= 20
        elif self.current_stage == "up":
            self.feedback_messages.append("✓ Good pull height")
        
        # Check chin height
        if self.current_stage == "up":
            if chin_above_bar:
                self.feedback_messages.append("✓ Chin above bar")
            else:
                errors.append("Chin should be above the bar")
                self.feedback_messages.append("⚠️ Chin not high enough")
                form_score -= 15
        
        # Check body swing
        if body_swing > self.MAX_SWING:
            errors.append("Too much body swing - control the movement")
            self.feedback_messages.append("⚠️ Minimize swinging")
            form_score -= 15
        else:
            self.feedback_messages.append("✓ Controlled movement")
        
        # Check full extension at bottom
        if self.current_stage == "down" and avg_elbow_angle < self.MAX_ELBOW_ANGLE - 10:
            errors.append("Fully extend arms at the bottom")
            self.feedback_messages.append("⚠️ Extend arms fully")
            form_score -= 10
        elif self.current_stage == "down":
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
            'body_swing': body_swing,
            'chin_above_bar': chin_above_bar,
            'feedback': self.feedback_messages,
            'errors': errors,
            'angles': {
                'left_elbow': left_elbow_angle,
                'right_elbow': right_elbow_angle,
                'body_swing': body_swing
            }
        }
