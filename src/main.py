"""
Main entry point for the Virtual Assistant & Gesture Control application.
Professional presentation-ready setup with refined UI, optimized performance,
and robust failsafe mechanics for live demonstrations.
"""

import sys
import time
import cv2
import numpy as np
from src.hand_tracker import HandTracker
from src.gesture_recognizer import GestureRecognizer
from src.mouse_controller import MouseController
from src.keyboard_controller import KeyboardController
from src.utils import (
    FPSCounter, draw_text_with_outline, draw_bounding_box, 
    draw_overlay, get_keyboard_zones, draw_keyboard_zones
)

def main():
    print("Initializing Virtual Assistant & Gesture Control (Phase 7 Validation)...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam on startup.")
        sys.exit(1)
        
    # Configure high-res frame if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    
    # Initialize Core Modules
    hand_tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
    gesture_recognizer = GestureRecognizer(pinch_threshold=45)
    mouse_controller = MouseController(
        frame_width=frame_width, 
        frame_height=frame_height, 
        margin=150, 
        smoothing_alpha=0.35 # Balanced responsiveness and smoothness
    )
    keyboard_controller = KeyboardController(cooldown=0.5)
    fps_counter = FPSCounter(avg_frames=15)
    
    # State Management
    is_keyboard_mode = False
    last_mode_switch_time = 0
    mode_switch_cooldown = 1.5
    frames_without_hand = 0
    
    print("System Online. Press 'q' in the window to exit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Warning: Camera frame read failed. Attempting recovery...")
                # Create a blank visual warning frame
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                draw_text_with_outline(frame, "CAMERA ERROR - Reconnecting...", (50, frame_height // 2), color=(0, 0, 255), font_scale=1.5)
                cv2.imshow("Gesture Control UI", frame)
                
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    break
                    
                # Re-initialize camera
                cap.release()
                cap = cv2.VideoCapture(0)
                continue
                
            frame = cv2.flip(frame, 1)

            # Processing
            try:
                results = hand_tracker.process_frame(frame)
                lm_list, bbox = hand_tracker.get_landmark_positions(frame, results)
            except Exception as e:
                print(f"Tracking exception caught: {e}")
                lm_list, bbox = [], None

            gesture_name = "UNKNOWN"
            confidence = False
            active_action = "Idle"
            active_zone_label = None
            
            # --- ACTION ROUTING ---
            if lm_list:
                frames_without_hand = 0
                gesture_name, confidence, metadata = gesture_recognizer.recognize_gesture(lm_list)
                
                if confidence:
                    index_finger_tip = lm_list[8][1:]
                    
                    # 1. Global Mode Switching
                    if gesture_name == "KEYBOARD MODE":
                        if time.time() - last_mode_switch_time > mode_switch_cooldown:
                            is_keyboard_mode = not is_keyboard_mode
                            last_mode_switch_time = time.time()
                            
                            if is_keyboard_mode:
                                keyboard_controller.activate_keyboard_mode()
                                mouse_controller.drag_stop()
                                active_action = "KEYBOARD MODE ACTIVATED"
                            else:
                                keyboard_controller.deactivate_keyboard_mode()
                                active_action = "MOUSE MODE ACTIVATED"
                    
                    # 2. Global Pause State
                    elif gesture_name == "PAUSE":
                        mouse_controller.drag_stop()
                        mouse_controller.reset_scroll()
                        active_action = "System Paused"

                    # 3. Keyboard Mode Actions
                    elif is_keyboard_mode:
                        mouse_controller.drag_stop()
                        mouse_controller.reset_scroll()
                        
                        cx, cy = index_finger_tip
                        zones = get_keyboard_zones(frame_width, frame_height)
                        
                        for zone in zones:
                            x1, y1, x2, y2 = zone["rect"]
                            if x1 <= cx <= x2 and y1 <= cy <= y2:
                                active_zone_label = zone["label"]
                                break
                                
                        active_action = f"Hover: {active_zone_label}" if active_zone_label else "Keyboard Idle"
                        
                        if gesture_name == "LEFT CLICK" and active_zone_label:
                            if keyboard_controller.trigger_action(active_zone_label):
                                active_action = f"Triggered: {active_zone_label}"

                    # 4. Mouse Mode Actions
                    else:
                        if gesture_name == "MOVE MODE":
                            mouse_controller.move_cursor(index_finger_tip)
                            mouse_controller.drag_stop()
                            mouse_controller.reset_scroll()
                            active_action = "Moving"
                            
                        elif gesture_name == "LEFT CLICK":
                            mouse_controller.move_cursor(index_finger_tip)
                            mouse_controller.left_click()
                            active_action = "Left Click"
                            
                        elif gesture_name == "RIGHT CLICK":
                            mouse_controller.right_click()
                            active_action = "Right Click"
                            
                        elif gesture_name == "SCROLL MODE":
                            mouse_controller.scroll(index_finger_tip[1])
                            active_action = "Scrolling"
                            
                        elif gesture_name == "DRAG MODE":
                            mouse_controller.drag_start(index_finger_tip)
                            active_action = "Dragging"
                else:
                    # Low confidence state fallback
                    mouse_controller.drag_stop()
                    mouse_controller.reset_scroll()
            else:
                # No hand detected
                frames_without_hand += 1
                mouse_controller.drag_stop()
                mouse_controller.reset_scroll()
                gesture_name = "None"
            
            # --- UI RENDERING ---
            # 1. Base HUD Overlays
            draw_overlay(frame, height=60, position='top')
            draw_overlay(frame, height=50, position='bottom')
            
            # 2. Contextual UI Elements
            if lm_list:
                hand_tracker.draw_landmarks(frame, results)
                # Modern corner bounding box
                draw_bounding_box(frame, bbox, color=(0, 255, 255), thickness=2, style='corners')
                
                if is_keyboard_mode:
                    zones = get_keyboard_zones(frame_width, frame_height)
                    draw_keyboard_zones(frame, zones, active_zone_label)
                else:
                    margin_box = mouse_controller.get_margin_box()
                    draw_bounding_box(frame, margin_box, color=(255, 0, 255), thickness=1, style='solid')
            
            # 3. Warning States
            if frames_without_hand > 45:
                # Show tracking lost warning if hand is absent for ~1.5 seconds at 30 FPS
                draw_text_with_outline(frame, "TRACKING LOST - Show hand to resume", 
                                       (frame_width // 2 - 250, frame_height // 2), 
                                       color=(0, 0, 255), font_scale=1.0)
            
            # 4. Status Text
            mode_text = "KEYBOARD MODE" if is_keyboard_mode else "MOUSE MODE"
            mode_color = (0, 255, 255) if is_keyboard_mode else (255, 100, 255)
            
            # Top Bar: Mode and Action
            draw_text_with_outline(frame, f"Mode: {mode_text}", (20, 40), color=mode_color, font_scale=0.8)
            draw_text_with_outline(frame, f"Action: {active_action}", (frame_width // 2 - 100, 40), color=(255, 255, 255))
            
            # Bottom Bar: Gesture and FPS
            fps = fps_counter.calculate_fps()
            draw_text_with_outline(frame, f"Gesture: {gesture_name}", (20, frame_height - 15), color=(0, 255, 0))
            
            # Temporary debug output for Shaka distance
            if lm_list:
                shaka_dist = int(metadata.get('thumb_pinky_dist', 0))
                fingers_str = str(metadata.get('fingers', []))
                debug_str = f"Shaka Dist: {shaka_dist} | Fingers: {fingers_str}"
                draw_text_with_outline(frame, debug_str, (300, frame_height - 15), color=(200, 200, 200), font_scale=0.6)
                
            draw_text_with_outline(frame, f"FPS: {fps}", (frame_width - 150, frame_height - 15), color=(255, 255, 255))
            
            # Show output
            cv2.imshow("Gesture Control UI", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit requested by user.")
                break
                
    except KeyboardInterrupt:
        print("\nShutdown signal received.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
    finally:
        print("Releasing system resources safely...")
        try:
            if 'mouse_controller' in locals() and mouse_controller.is_dragging:
                mouse_controller.drag_stop()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
