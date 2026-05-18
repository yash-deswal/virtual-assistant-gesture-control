"""
Tests for Gesture Control modules.
Updated for Phase 7 final validation and robust edge case handling.
"""

import pytest
import numpy as np
import time
from src.hand_tracker import HandTracker
from src.gesture_recognizer import GestureRecognizer

def test_hand_tracker_invalid_frame():
    """
    Test processing an invalid or malformed frame to ensure it doesn't crash the pipeline.
    """
    tracker = HandTracker()
    
    # Test completely empty array
    frame = np.array([])
    with pytest.raises(Exception):
        # OpenCV cvtColor will raise an error on empty arrays, 
        # which is caught and handled safely in main.py's try-except block
        tracker.process_frame(frame)
        
    # Test blank black frame (valid shape but no hands)
    valid_blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = tracker.process_frame(valid_blank_frame)
    lm_list, bbox = tracker.get_landmark_positions(valid_blank_frame, results)
    
    assert lm_list == []
    assert bbox is None

def test_gesture_recognizer_debounce():
    """
    Test the gesture recognizer's debounce logic to ensure stability against flickering.
    """
    recognizer = GestureRecognizer()
    
    # Simulate a fist gesture for DRAG MODE
    landmarks = [(i, 0, 0) for i in range(21)]
    landmarks[5] = (5, 200, 200)
    landmarks[17] = (17, 100, 200)
    landmarks[3] = (3, 150, 200)
    landmarks[4] = (4, 250, 200)
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        landmarks[pip] = (pip, 0, 50)
        landmarks[tip] = (tip, 0, 100)
        
    # DRAG MODE requires 3 frames to become stable
    gesture, match, meta = recognizer.recognize_gesture(landmarks)
    assert gesture == "UNKNOWN" # Frame 1
    
    gesture, match, meta = recognizer.recognize_gesture(landmarks)
    assert gesture == "UNKNOWN" # Frame 2
    
    gesture, match, meta = recognizer.recognize_gesture(landmarks)
    assert gesture == "DRAG MODE" # Frame 3, stable state reached
    assert match

def test_gesture_recognizer_shaka():
    """
    Test KEYBOARD MODE (Shaka) gesture validation.
    """
    recognizer = GestureRecognizer(shaka_distance_threshold=100)
    landmarks = [(i, 0, 0) for i in range(21)]
    # Right hand: index MCP right of pinky MCP
    landmarks[5] = (5, 200, 200)
    landmarks[17] = (17, 100, 200)
    
    # Thumb wide left
    landmarks[3] = (3, 200, 200)
    landmarks[4] = (4, 50, 200)  # Very far left
    
    # Index, Middle, Ring down
    for tip, pip in [(8, 6), (12, 10), (16, 14)]:
        landmarks[pip] = (pip, 0, 50)
        landmarks[tip] = (tip, 0, 100)
        
    # Pinky up and wide right
    landmarks[18] = (18, 0, 100)
    landmarks[20] = (20, 250, 50) # Far right, up
    
    # Distance thumb(50,200) to pinky(250,50) is hypot(200, 150) = 250 > 100
    
    # Test requires 2 frames for debounce threshold of KEYBOARD MODE
    gesture, match, meta = recognizer.recognize_gesture(landmarks)
    assert gesture == "UNKNOWN" # Frame 1
    
    gesture, match, meta = recognizer.recognize_gesture(landmarks)
    assert gesture == "KEYBOARD MODE" # Frame 2, stable state reached
    assert match
    assert meta["thumb_pinky_dist"] == 250.0

def test_mouse_controller_ema_smoothing():
    """
    Test Exponential Moving Average cursor smoothing.
    """
    from src.mouse_controller import MouseController
    controller = MouseController(frame_width=640, frame_height=480, margin=0, smoothing_alpha=0.5)
    
    controller.screen_width = 1000
    controller.screen_height = 1000
    
    controller.curr_x = 0
    controller.curr_y = 0
    
    controller.move_cursor((320, 240))
    # Target mapped = (500, 500)
    # EMA formula: (0.5 * 500) + (0.5 * 0) = 250
    assert abs(controller.curr_x - 250.0) < 1.0
    
    controller.move_cursor((320, 240))
    # EMA formula: (0.5 * 500) + (0.5 * 250) = 375
    assert abs(controller.curr_x - 375.0) < 1.0

def test_mouse_controller_out_of_bounds():
    """
    Test that mapping correctly clamps coordinates safely inside the screen bounds.
    """
    from src.mouse_controller import MouseController
    controller = MouseController(frame_width=640, frame_height=480, margin=100)
    controller.screen_width = 1920
    controller.screen_height = 1080
    
    # Send wild negative coordinates
    x, y = controller._map_to_screen(-1000, -5000)
    assert x == 0
    assert y == 0
    
    # Send wild positive coordinates
    x, y = controller._map_to_screen(5000, 9000)
    assert x == 1920 - 1
    assert y == 1080 - 1

def test_mouse_controller_drag_safety():
    """
    Test safe drag starting and stopping state management.
    """
    from src.mouse_controller import MouseController
    controller = MouseController()
    
    assert not controller.is_dragging
    controller.drag_start()
    assert controller.is_dragging
    controller.drag_start() # Double call shouldn't break state
    assert controller.is_dragging
    controller.drag_stop()
    assert not controller.is_dragging

def test_keyboard_controller_cooldown():
    """
    Test that KeyboardController prevents spamming keys.
    """
    from src.keyboard_controller import KeyboardController
    controller = KeyboardController(cooldown=0.5)
    
    controller.activate_keyboard_mode()
    
    # First action should succeed
    success = controller.trigger_action("FakeKey")
    assert success
    
    # Immediate second action should fail due to cooldown
    success2 = controller.trigger_action("FakeKey")
    assert not success2
    
    # Simulate time passing (manual override for test)
    controller.last_action_time = time.time() - 1.0
    success3 = controller.trigger_action("FakeKey")
    assert success3
