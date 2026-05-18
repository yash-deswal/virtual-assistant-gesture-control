"""
Mouse Controller Module.

Responsible for executing virtual mouse movements using PyAutoGUI.
Optimized for smooth, jitter-free cursor tracking using Exponential Moving Averages (EMA).
"""

import pyautogui
import time
import numpy as np

class MouseController:
    """
    Handles virtual mouse operations with smoothing and safety constraints.
    """
    def __init__(self, frame_width=640, frame_height=480, margin=150, smoothing_alpha=0.3):
        """
        Initialize the mouse controller.
        
        Args:
            frame_width (int): Frame width.
            frame_height (int): Frame height.
            margin (int): Frame margin for bounding the movement area.
            smoothing_alpha (float): Smoothing factor (0.0 to 1.0). Lower is smoother but slower.
        """
        # Disable failsafe to allow reaching screen edges programmatically
        pyautogui.FAILSAFE = False
        
        self.screen_width, self.screen_height = pyautogui.size()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.margin = margin
        
        # Exponential Moving Average factor
        self.alpha = smoothing_alpha
        
        # Current smoothed coordinates
        self.curr_x = self.screen_width / 2
        self.curr_y = self.screen_height / 2
        
        # Cooldown management
        self.last_click_time = 0
        self.click_cooldown = 0.4 # Slightly longer cooldown to prevent double accidental clicks
        
        # States
        self.is_dragging = False
        self.last_scroll_y = None

    def _map_to_screen(self, x, y):
        """
        Map frame coordinates to screen coordinates.
        Clamps values to ensure cursor stays within screen bounds.
        """
        # Map x, y with margin
        screen_x = np.interp(x, (self.margin, self.frame_width - self.margin), (0, self.screen_width))
        screen_y = np.interp(y, (self.margin, self.frame_height - self.margin), (0, self.screen_height))
        
        # Clamp to screen size for safety
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))
        
        return screen_x, screen_y

    def move_cursor(self, index_finger_position):
        """
        Move the mouse cursor using Exponential Moving Average for smooth tracking.
        """
        x, y = index_finger_position
        target_x, target_y = self._map_to_screen(x, y)
        
        # Apply EMA smoothing formula: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
        self.curr_x = (self.alpha * target_x) + ((1.0 - self.alpha) * self.curr_x)
        self.curr_y = (self.alpha * target_y) + ((1.0 - self.alpha) * self.curr_y)
        
        pyautogui.moveTo(int(self.curr_x), int(self.curr_y))

    def left_click(self):
        """Perform a single left click with cooldown."""
        if time.time() - self.last_click_time > self.click_cooldown:
            pyautogui.click(button='left')
            self.last_click_time = time.time()

    def right_click(self):
        """Perform a single right click with cooldown."""
        if time.time() - self.last_click_time > self.click_cooldown:
            pyautogui.click(button='right')
            self.last_click_time = time.time()
        
    def scroll(self, current_y):
        """
        Perform scrolling based on relative vertical hand movement.
        """
        if self.last_scroll_y is not None:
            delta = self.last_scroll_y - current_y
            scroll_amount = int(delta * 2.5) # Scroll sensitivity
            
            if abs(scroll_amount) > 3:
                pyautogui.scroll(scroll_amount)
                
        self.last_scroll_y = current_y

    def reset_scroll(self):
        """Reset scroll baseline."""
        self.last_scroll_y = None

    def drag_start(self, position=None):
        """Start drag action."""
        if not self.is_dragging:
            pyautogui.mouseDown(button='left')
            self.is_dragging = True
            
        if position:
            self.move_cursor(position)

    def drag_stop(self):
        """Stop drag action cleanly."""
        if self.is_dragging:
            pyautogui.mouseUp(button='left')
            self.is_dragging = False

    def get_margin_box(self):
        """Return the active movement area boundaries for UI."""
        return (self.margin, self.margin, 
                self.frame_width - self.margin, self.frame_height - self.margin)
