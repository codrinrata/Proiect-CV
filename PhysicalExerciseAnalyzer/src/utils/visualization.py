"""
Utility functions for the laundry symbol detector project.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def display_images(images: List[np.ndarray], titles: List[str], 
                   figsize: Tuple[int, int] = (15, 10), cmap: str = 'gray'):
    """
    Display multiple images in a grid.
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        figsize: Figure size (width, height)
        cmap: Colormap for grayscale images
    """
    n_images = len(images)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 2:
            axes[i].imshow(img, cmap=cmap)
        else:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def save_processing_steps(steps: dict, output_dir: str = 'outputs'):
    """
    Save all preprocessing steps as separate images.
    
    Args:
        steps: Dictionary of step name to image
        output_dir: Directory to save images
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for step_name, image in steps.items():
        filepath = os.path.join(output_dir, f'{step_name}.png')
        cv2.imwrite(filepath, image)
        print(f"Saved {step_name} to {filepath}")


def draw_shape_info(image: np.ndarray, contour: np.ndarray, 
                    features: dict, position: Tuple[int, int] = (10, 30)):
    """
    Draw shape information on image.
    
    Args:
        image: Image to draw on
        contour: Contour of the shape
        features: Dictionary of shape features
        position: Starting position for text
    
    Returns:
        Image with annotations
    """
    img = image.copy()
    
    # Draw contour
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    
    # Draw bounding box
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = position[1]
    
    for key, value in features.items():
        if isinstance(value, float):
            text = f"{key}: {value:.3f}"
        else:
            text = f"{key}: {value}"
        
        cv2.putText(img, text, (position[0], y_offset), 
                   font, 0.5, (255, 255, 255), 1)
        y_offset += 20
    
    return img


def create_comparison_grid(original: np.ndarray, processed: List[np.ndarray],
                          titles: List[str]) -> np.ndarray:
    """
    Create a comparison grid of original and processed images.
    
    Args:
        original: Original image
        processed: List of processed images
        titles: Titles for each processed image
    
    Returns:
        Grid image
    """
    n_images = len(processed) + 1
    all_images = [original] + processed
    all_titles = ['Original'] + titles
    
    # Resize all to same size
    target_size = (400, 400)
    resized = [cv2.resize(img, target_size) for img in all_images]
    
    # Create grid
    n_cols = 3
    n_rows = (n_images + n_cols - 1) // n_cols
    
    grid = np.zeros((n_rows * target_size[1], n_cols * target_size[0], 3), dtype=np.uint8)
    
    for i, img in enumerate(resized):
        row = i // n_cols
        col = i % n_cols
        
        # Convert grayscale to BGR if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        y_start = row * target_size[1]
        x_start = col * target_size[0]
        
        grid[y_start:y_start+target_size[1], x_start:x_start+target_size[0]] = img
        
        # Add title
        cv2.putText(grid, all_titles[i], (x_start + 10, y_start + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return grid


def annotate_detection(image: np.ndarray, bbox: Tuple[int, int, int, int],
                       label: str, confidence: float, color: Tuple[int, int, int] = (0, 255, 0)):
    """
    Annotate detection on image.
    
    Args:
        image: Image to annotate
        bbox: Bounding box (x, y, w, h)
        label: Label text
        confidence: Confidence score
        color: Color for box and text
    
    Returns:
        Annotated image
    """
    img = image.copy()
    x, y, w, h = bbox
    
    # Draw box
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    
    # Prepare text
    text = f"{label}: {confidence:.2%}"
    
    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background for text
    cv2.rectangle(img, (x, y - text_h - 10), (x + text_w, y), color, -1)
    
    # Draw text
    cv2.putText(img, text, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
    
    return img
