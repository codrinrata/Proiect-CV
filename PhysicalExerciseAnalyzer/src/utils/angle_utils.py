import numpy as np
from typing import Tuple, List


def calculate_angle(point1: Tuple[float, float], 
                   point2: Tuple[float, float], 
                   point3: Tuple[float, float]) -> float:
    
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle)
    
    return angle_degrees


def calculate_angle_3d(point1: Tuple[float, float, float], 
                       point2: Tuple[float, float, float], 
                       point3: Tuple[float, float, float]) -> float:
    
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    p1 = np.array(point1)
    p2 = np.array(point2)
    return np.linalg.norm(p1 - p2)


def calculate_vertical_alignment(point1: Tuple[float, float], 
                                 point2: Tuple[float, float],
                                 tolerance: float = 0.05) -> bool:
    return abs(point1[0] - point2[0]) < tolerance


def calculate_horizontal_alignment(point1: Tuple[float, float], 
                                   point2: Tuple[float, float],
                                   tolerance: float = 0.05) -> bool:
    return abs(point1[1] - point2[1]) < tolerance


def calculate_slope(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    
    if dx == 0:
        return 90.0 if dy > 0 else -90.0
    
    angle = np.degrees(np.arctan(dy / dx))
    return angle


def is_point_between(point: Tuple[float, float],
                     line_start: Tuple[float, float],
                     line_end: Tuple[float, float],
                     tolerance: float = 0.1) -> bool:
    total_dist = calculate_distance(line_start, line_end)
    dist1 = calculate_distance(line_start, point)
    dist2 = calculate_distance(point, line_end)
    return abs((dist1 + dist2) - total_dist) < tolerance


def normalize_coordinates(coords: List[Tuple[float, float]], 
                         reference_length: float) -> List[Tuple[float, float]]:
    if reference_length == 0:
        return coords
    return [(x / reference_length, y / reference_length) for x, y in coords]


def calculate_center_of_mass(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    
    x_avg = sum(p[0] for p in points) / len(points)
    y_avg = sum(p[1] for p in points) / len(points)
    return (x_avg, y_avg)


def calculate_body_lean(shoulder: Tuple[float, float], hip: Tuple[float, float]) -> float:
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    angle = np.degrees(np.arctan2(dx, abs(dy)))
    return abs(angle)
