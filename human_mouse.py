import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
import time

# -----------------------
# Screen & smoothing setup
# -----------------------
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CURSOR_X, CURSOR_Y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2


SMOOTHING = 0.1   
DEADZONE = 4       
MOVE_SCALE = 1.6   

history_length = 8
history = deque(maxlen=history_length)

# -----------------------
# Mediapipe setup
# -----------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,          
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------
# Webcam
# -----------------------
cap = cv2.VideoCapture(0)

# Neutral head position
neutral_head = None

# Fail-safe off
pyautogui.FAILSAFE = False

# -----------------------
# Utility functions
# -----------------------
def get_head_position(face_landmarks, w, h):
    # Use tip of nose as head center
    nose = face_landmarks.landmark[1]
    return int(nose.x * w), int(nose.y * h)

def is_mouth_open(face_landmarks, h):
    # Approximate mouth open by distance between upper and lower lip
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]
    distance = abs((top_lip.y - bottom_lip.y) * h)
    # You can tweak this threshold if it's too sensitive
    return distance > 18

def get_eye_open_ratio(face_landmarks):
    """
    Compute how open ONE eye is, normalized by face height.
    We'll use a single eye so wink detection is more consistent.
    Using landmarks 159 (upper) and 145 (lower) for one eye.
    """
    # Eye landmarks (one eye)
    upper = face_landmarks.landmark[159]
    lower = face_landmarks.landmark[145]

    eye_open = (lower.y - upper.y)

    # Normalize by face height so it works at different distances
    top_face = face_landmarks.landmark[10]   # forehead-ish
    bottom_face = face_landmarks.landmark[152]  # chin
    face_height = (bottom_face.y - top_face.y)

    if face_height <= 0:
        return 0.0

    return eye_open / face_height

# -----------------------
# Wink / right-click state
# -----------------------
EYE_CLOSED_THRESH = 0.018   # below this = definitely closed
EYE_OPEN_THRESH = 0.028     # above this = definitely open again
RIGHT_CLICK_COOLDOWN = 0.35

eye_state = "open"          # "open" or "closed"
last_right_click_time = 0.0

# -----------------------
# Main loop
# -----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # -----------------------
    # Face detection
    # -----------------------
    face_results = face_mesh.process(rgb)
    head_x, head_y = None, None
    mouth_open_flag = False
    eye_open_ratio = None

    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0]

        # Head position for mouse
        head_x, head_y = get_head_position(landmarks, w, h)
        if neutral_head is None:
            neutral_head = (head_x, head_y)

        # Mouth for left click
        mouth_open_flag = is_mouth_open(landmarks, h)

        # Eye open ratio for wink / right click
        eye_open_ratio = get_eye_open_ratio(landmarks)

    # -----------------------
    # Head -> cursor movement
    # -----------------------
    if head_x is not None and head_y is not None and neutral_head is not None:
        dx = head_x - neutral_head[0]
        dy = head_y - neutral_head[1]

        # Deadzone filter
        if abs(dx) < DEADZONE:
            dx = 0
        if abs(dy) < DEADZONE:
            dy = 0

        # Target position
        target_x = CURSOR_X + dx * MOVE_SCALE
        target_y = CURSOR_Y + dy * MOVE_SCALE

        # EMA smoothing
        CURSOR_X = CURSOR_X * (1 - SMOOTHING) + target_x * SMOOTHING
        CURSOR_Y = CURSOR_Y * (1 - SMOOTHING) + target_y * SMOOTHING

        # Clamp to screen
        CURSOR_X = max(0, min(SCREEN_WIDTH - 1, CURSOR_X))
        CURSOR_Y = max(0, min(SCREEN_HEIGHT - 1, CURSOR_Y))

        # Moving average smoothing
        history.append((CURSOR_X, CURSOR_Y))
        avg_x = int(sum(p[0] for p in history) / len(history))
        avg_y = int(sum(p[1] for p in history) / len(history))

        pyautogui.moveTo(avg_x, avg_y, duration=0.01)

    now = time.time()

    # -----------------------
    # Left click: mouth open (original behaviour)
    # -----------------------
    if face_results.multi_face_landmarks and mouth_open_flag:
        pyautogui.click(button="left")

    # -----------------------
    # Right click: wink with one eye
    # -----------------------
    if eye_open_ratio is not None:
        # Debug: draw eye ratio
        cv2.putText(frame, f"Eye ratio: {eye_open_ratio:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # If eye clearly closed and was open before → trigger right click
        if eye_open_ratio < EYE_CLOSED_THRESH and eye_state == "open":
            if now - last_right_click_time > RIGHT_CLICK_COOLDOWN:
                pyautogui.click(button="right")
                last_right_click_time = now
            eye_state = "closed"

        # Reset state once eye clearly open again
        elif eye_open_ratio > EYE_OPEN_THRESH:
            eye_state = "open"

    # -----------------------
    # Debug overlay
    # -----------------------
    cv2.putText(frame, "Head Mouse + Mouth Left Click + Wink Right Click",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.imshow("Head Mouse + Gestures", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
