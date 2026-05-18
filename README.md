# Virtual Assistant & Gesture Control

## Overview
A modular gesture-controlled virtual mouse and keyboard system using Python 3.10, OpenCV, MediaPipe, PyAutoGUI, and pynput. This project allows you to control your computer using hand gestures.

## Planned Features
- Hand tracking and landmark detection
- Gesture recognition
- Virtual mouse control (move, click, drag)
- Virtual keyboard control
- Customizable gesture mapping

## Folder Structure
- `src/`: Source code for hand tracking, gesture recognition, and control.
- `assets/`: Reference images and other static assets.
- `tests/`: Unit tests.
- `env/`: Conda environment.

## Environment Setup
The project uses a conda environment.

To activate the environment on macOS M1:
```bash
conda activate ./env
```

## How to Run
```bash
python -m src.main
```
