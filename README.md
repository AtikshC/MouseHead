# MouseHead

MouseHead is a computer vision–based human–computer interaction system that enables hands-free mouse control using head movement and facial gestures. By leveraging real-time facial landmark detection through a standard webcam, MouseHead translates natural facial expressions into mouse actions.

This project explores alternative input methods for accessibility, experimentation, and novel interaction design.

It placed 3rd in a University sponsored Hackathon.

---

## Features

- Hands-free cursor control using head movement
- Tongue gesture for left-click
- Wink gesture for right-click
- Real-time facial landmark tracking
- Fully local processing (no data collection or cloud services)

---

## How It Works

MouseHead captures video input from a webcam and applies facial landmark detection to track key facial features. These landmarks are mapped to screen coordinates and mouse events:

- Head position controls cursor movement
- Tongue detection triggers left-click events
- Eye wink detection triggers right-click events

Mouse interactions are executed using system-level automation.

---

## Technology Stack

- Python
- OpenCV
- MediaPipe
- PyAutoGUI
- NumPy

---

## Installation

Upgrade pip:
```bash
pip install --upgrade pip
