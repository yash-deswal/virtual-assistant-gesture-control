"""
Gesture Recognizer Module.

Responsible for translating hand landmarks into predefined gestures
with stable debouncing and noise filtering.
"""

from src.utils import calculate_distance

class GestureRecognizer:
    """
    Analyzes hand landmarks to identify specific gestures robustly.
    """
    def __init__(self, pinch_threshold=40, shaka_distance_threshold=140, thumb_tolerance=15):
        """
        Initialize the gesture recognizer.
        
        Args:
            pinch_threshold (int): Maximum distance between landmarks for a pinch.
            shaka_distance_threshold (int): Minimum distance between thumb and pinky for Shaka gesture.
            thumb_tolerance (int): Margin in pixels to tolerate thumb angle variations.
        """
        self.pinch_threshold = pinch_threshold
        self.shaka_distance_threshold = shaka_distance_threshold
        self.thumb_tolerance = thumb_tolerance
        
        # Gesture Debouncing State
        self.previous_raw_gesture = "UNKNOWN"
        self.stable_gesture = "UNKNOWN"
        self.debounce_counter = 0
        
        # Configuration for stability
        # Mode switches need more frames to confirm, clicks can be faster
        self.debounce_thresholds = {
            "MOVE MODE": 2,
            "LEFT CLICK": 1,
            "RIGHT CLICK": 1,
            "SCROLL MODE": 2,
            "DRAG MODE": 3,
            "PAUSE": 3,
            "KEYBOARD MODE": 2, # Lowered from 5 to 2 to make activation faster and more responsive
            "UNKNOWN": 1
        }

    def calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two landmark points.
        """
        return calculate_distance(point1[1:], point2[1:])

    def fingers_up(self, landmarks):
        """
        Determine which fingers are extended.
        """
        fingers = []
        
        # 1. Thumb
        # Compare index MCP (5) and pinky MCP (17) to determine hand orientation
        if landmarks[5][1] > landmarks[17][1]:
            # Right hand (selfie-view)
            fingers.append(1 if landmarks[4][1] < landmarks[3][1] + self.thumb_tolerance else 0)
        else:
            # Left hand (selfie-view)
            fingers.append(1 if landmarks[4][1] > landmarks[3][1] - self.thumb_tolerance else 0)

        # 2-5. Index, Middle, Ring, Pinky
        tips_pips = [(8, 6), (12, 10), (16, 14), (20, 18)]
        for tip, pip in tips_pips:
            # Tip cy < Pip cy indicates finger is up
            fingers.append(1 if landmarks[tip][2] < landmarks[pip][2] else 0)
            
        return fingers

    def _get_raw_gesture(self, landmarks, fingers):
        """
        Evaluate raw heuristic gesture state without debouncing.
        """
        thumb_index_dist = self.calculate_distance(landmarks[4], landmarks[8])
        index_middle_dist = self.calculate_distance(landmarks[8], landmarks[12])
        thumb_pinky_dist = self.calculate_distance(landmarks[4], landmarks[20])

        # F. PAUSE: Open palm (all fingers extended)
        if fingers == [1, 1, 1, 1, 1]:
            return "PAUSE", thumb_index_dist, index_middle_dist, thumb_pinky_dist

        # E. DRAG MODE: Closed fist
        if fingers == [0, 0, 0, 0, 0]:
            return "DRAG MODE", thumb_index_dist, index_middle_dist, thumb_pinky_dist

        # G. KEYBOARD MODE: Shaka sign
        # We relax the strict thumb x-coordinate check and rely on wide thumb-pinky spread.
        # Require Middle and Ring folded, Pinky up, and a wide physical distance between thumb and pinky.
        if fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:
            if thumb_pinky_dist > self.shaka_distance_threshold:
                return "KEYBOARD MODE", thumb_index_dist, index_middle_dist, thumb_pinky_dist

        # D. SCROLL MODE & C. RIGHT CLICK: Index + Middle fingers up
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            if index_middle_dist < self.pinch_threshold:
                return "RIGHT CLICK", thumb_index_dist, index_middle_dist, thumb_pinky_dist
            return "SCROLL MODE", thumb_index_dist, index_middle_dist, thumb_pinky_dist

        # A. MOVE MODE & B. LEFT CLICK: Index finger up only
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            if thumb_index_dist < self.pinch_threshold:
                return "LEFT CLICK", thumb_index_dist, index_middle_dist, thumb_pinky_dist
            return "MOVE MODE", thumb_index_dist, index_middle_dist, thumb_pinky_dist

        return "UNKNOWN", thumb_index_dist, index_middle_dist, thumb_pinky_dist

    def recognize_gesture(self, landmarks):
        """
        Identify a gesture with robust debouncing to prevent flickering.
        
        Args:
            landmarks (list): Hand landmark data.
            
        Returns:
            tuple: (Stable Gesture (str), Confidence (bool), metadata (dict))
        """
        if not landmarks or len(landmarks) < 21:
            self.stable_gesture = "UNKNOWN"
            return "UNKNOWN", False, {}

        fingers = self.fingers_up(landmarks)
        raw_gesture, thumb_index_dist, index_middle_dist, thumb_pinky_dist = self._get_raw_gesture(landmarks, fingers)
        
        metadata = {
            "fingers": fingers,
            "thumb_index_dist": thumb_index_dist,
            "index_middle_dist": index_middle_dist,
            "thumb_pinky_dist": thumb_pinky_dist,
            "raw_gesture": raw_gesture
        }

        # Debounce logic
        if raw_gesture == self.previous_raw_gesture:
            self.debounce_counter += 1
        else:
            self.debounce_counter = 0
            self.previous_raw_gesture = raw_gesture

        required_frames = self.debounce_thresholds.get(raw_gesture, 2)
        
        if self.debounce_counter >= required_frames:
            self.stable_gesture = raw_gesture

        # Consider it a confident match if stable_gesture isn't UNKNOWN
        match = (self.stable_gesture != "UNKNOWN")

        return self.stable_gesture, match, metadata
