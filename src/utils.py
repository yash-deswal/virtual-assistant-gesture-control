"""
Utility Functions Module.

Contains common helper functions used across the project,
such as coordinate mapping, geometry calculations, drawing utilities,
and UI overlays.
"""

import math
import cv2
import time
import numpy as np

class FPSCounter:
    """Helper class to calculate and keep track of FPS."""
    def __init__(self, avg_frames=10):
        self.p_time = 0
        self.frame_times = []
        self.avg_frames = avg_frames

    def calculate_fps(self):
        """Calculates current FPS based on time elapsed with a moving average for stability."""
        c_time = time.time()
        if self.p_time > 0:
            self.frame_times.append(c_time - self.p_time)
            if len(self.frame_times) > self.avg_frames:
                self.frame_times.pop(0)
                
        self.p_time = c_time
        
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            return int(1 / avg_time) if avg_time > 0 else 0
        return 0

def draw_text_with_outline(frame, text, position, font_scale=0.7, color=(255, 255, 255), thickness=2):
    """
    Draws text with a black outline for better visibility against any background.
    """
    # Outline
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
    # Inner Text
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_overlay(frame, height=80, position='top'):
    """
    Draws a translucent overlay bar for clean UI status rendering.
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    if position == 'top':
        cv2.rectangle(overlay, (0, 0), (w, height), (0, 0, 0), -1)
    else:
        cv2.rectangle(overlay, (0, h - height), (w, h), (0, 0, 0), -1)
        
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

def draw_bounding_box(frame, bbox, color=(0, 255, 0), thickness=2, style='corners'):
    """
    Draws a bounding box on the frame.
    """
    if not bbox:
        return
        
    xmin, ymin, xmax, ymax = bbox
    
    if style == 'solid':
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
    elif style == 'corners':
        # Draw only the corners for a cleaner HUD look
        length = 20
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1) # Faint full box
        # Top-Left
        cv2.line(frame, (xmin, ymin), (xmin + length, ymin), color, thickness)
        cv2.line(frame, (xmin, ymin), (xmin, ymin + length), color, thickness)
        # Top-Right
        cv2.line(frame, (xmax, ymin), (xmax - length, ymin), color, thickness)
        cv2.line(frame, (xmax, ymin), (xmax, ymin + length), color, thickness)
        # Bottom-Left
        cv2.line(frame, (xmin, ymax), (xmin + length, ymax), color, thickness)
        cv2.line(frame, (xmin, ymax), (xmin, ymax - length), color, thickness)
        # Bottom-Right
        cv2.line(frame, (xmax, ymax), (xmax - length, ymax), color, thickness)
        cv2.line(frame, (xmax, ymax), (xmax, ymax - length), color, thickness)

def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    """
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

def get_keyboard_zones(width, height):
    """
    Calculate the bounding boxes for the virtual keyboard zones.
    Positioned lower in the frame to avoid face occlusion.
    """
    margin_y = int(height * 0.4) # Start zones 40% down the screen
    h_zone = (height - margin_y) // 2
    w3 = width // 3
    
    return [
        {"label": "Space", "rect": (0, margin_y, w3, margin_y + h_zone)},
        {"label": "Enter", "rect": (w3, margin_y, 2*w3, margin_y + h_zone)},
        {"label": "Backspace", "rect": (2*w3, margin_y, width, margin_y + h_zone)},
        {"label": "Vol -", "rect": (0, margin_y + h_zone, w3, height)},
        {"label": "Vol +", "rect": (w3, margin_y + h_zone, 2*w3, height)},
        {"label": "Screenshot", "rect": (2*w3, margin_y + h_zone, width, height)}
    ]

def draw_keyboard_zones(frame, zones, active_zone_label=None):
    """
    Draw interactive keyboard zones with a professional glass-morphism style.
    """
    overlay = frame.copy()
    for zone in zones:
        x1, y1, x2, y2 = zone["rect"]
        is_active = (zone["label"] == active_zone_label)
        
        # Color: Cyan if active, Dark Gray otherwise
        color = (255, 255, 0) if is_active else (50, 50, 50)
        
        if is_active:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
        
        # Center the text
        text_size = cv2.getTextSize(zone["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        tx = x1 + (x2 - x1 - text_size[0]) // 2
        ty = y1 + (y2 - y1 + text_size[1]) // 2
        
        text_color = (0, 0, 0) if is_active else (255, 255, 255)
        cv2.putText(frame, zone["label"], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
