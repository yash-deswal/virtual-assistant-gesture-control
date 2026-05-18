"""
Keyboard Controller Module.

Responsible for executing keyboard inputs based on recognized gestures.
"""

import time
from pynput.keyboard import Controller, Key

class KeyboardController:
    """
    Handles virtual keyboard operations.
    """
    def __init__(self, cooldown=0.6):
        """
        Initialize the keyboard controller.
        
        Args:
            cooldown (float): Minimum time between keystrokes to prevent spamming.
        """
        self.keyboard = Controller()
        self.is_active = False
        self.last_action_time = 0
        self.cooldown = cooldown

    def activate_keyboard_mode(self):
        """Enable keyboard mode."""
        self.is_active = True

    def deactivate_keyboard_mode(self):
        """Disable keyboard mode."""
        self.is_active = False

    def trigger_action(self, action_label):
        """
        Trigger a specific keyboard action based on label.
        
        Args:
            action_label (str): The label of the action to trigger.
            
        Returns:
            bool: True if triggered, False if on cooldown.
        """
        if time.time() - self.last_action_time < self.cooldown:
            return False
            
        try:
            if action_label == "Space":
                self.keyboard.press(Key.space)
                self.keyboard.release(Key.space)
            elif action_label == "Enter":
                self.keyboard.press(Key.enter)
                self.keyboard.release(Key.enter)
            elif action_label == "Backspace":
                self.keyboard.press(Key.backspace)
                self.keyboard.release(Key.backspace)
            elif action_label == "Vol -":
                self.keyboard.press(Key.media_volume_down)
                self.keyboard.release(Key.media_volume_down)
            elif action_label == "Vol +":
                self.keyboard.press(Key.media_volume_up)
                self.keyboard.release(Key.media_volume_up)
            elif action_label == "Screenshot":
                # macOS shortcut for screenshot: Cmd+Shift+3
                self.keyboard.press(Key.cmd)
                self.keyboard.press(Key.shift)
                self.keyboard.press('3')
                self.keyboard.release('3')
                self.keyboard.release(Key.shift)
                self.keyboard.release(Key.cmd)
        except Exception as e:
            print(f"Failed to trigger {action_label}: {e}")
            
        self.last_action_time = time.time()
        return True
