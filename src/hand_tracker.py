"""
Hand Tracker Module.

Responsible for detecting hands and extracting landmarks using MediaPipe.
"""

import cv2
import mediapipe as mp

class HandTracker:
    """
    Encapsulates MediaPipe hand tracking functionality.
    """
    def __init__(self, max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75):
        """
        Initialize the HandTracker with MediaPipe configuration.
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence)
        )

    def process_frame(self, frame):
        """
        Process an image frame to detect hands and landmarks.
        
        Args:
            frame: BGR image frame from OpenCV.
            
        Returns:
            MediaPipe results object containing hand landmarks.
        """
        # MediaPipe expects RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results

    def get_landmark_positions(self, frame, results):
        """
        Extract landmark positions and calculate bounding box.
        """
        lm_list = []
        bbox = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, c = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                    x_min, y_min = min(x_min, cx), min(y_min, cy)
                    x_max, y_max = max(x_max, cx), max(y_max, cy)
                
                # Add some padding to the bounding box
                bbox = (x_min - 20, y_min - 20, x_max + 20, y_max + 20)
                break  # Only track the first hand
        return lm_list, bbox

    def draw_landmarks(self, frame, results):
        """
        Draw hand landmarks and connections on the frame.
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
