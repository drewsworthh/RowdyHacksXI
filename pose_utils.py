# pose_utils.py
import math
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


# --- Utility Functions ---
def is_fist(hand_landmarks):
    tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    pips = [
        mp_hands.HandLandmark.THUMB_IP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP,
    ]

    curled = sum(1 for tip, pip in zip(tips, pips)
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
    return curled >= 3


def is_pointing(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    index_extended = index_tip.y < index_pip.y
    middle_curled = middle_tip.y > middle_pip.y - 0.05
    ring_curled = ring_tip.y > ring_pip.y - 0.05
    pinky_curled = pinky_tip.y > pinky_pip.y - 0.05

    return index_extended and middle_curled and ring_curled and pinky_curled


def hands_on_hips(pose_landmarks, wrist_landmark, side="left", threshold=0.2):
    if side == "left":
        hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    else:
        hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    dx = wrist_landmark.x - hip.x
    dy = wrist_landmark.y - hip.y
    distance = (dx**2 + dy**2) ** 0.5
    return distance < threshold


def arm_angle(shoulder, elbow, wrist):
    a = [shoulder.x - elbow.x, shoulder.y - elbow.y]
    b = [wrist.x - elbow.x, wrist.y - elbow.y]
    dot = a[0] * b[0] + a[1] * b[1]
    mag_a = math.sqrt(a[0] ** 2 + a[1] ** 2)
    mag_b = math.sqrt(b[0] ** 2 + b[1] ** 2)
    if mag_a * mag_b == 0:
        return 0
    return math.degrees(math.acos(dot / (mag_a * mag_b)))


def spongebob_pose(pose_landmarks, wrist_landmark, side="left", threshold=0.11):
    if not pose_landmarks:
        return False

    shoulder = pose_landmarks.landmark[
        mp_pose.PoseLandmark.LEFT_SHOULDER if side == "left" else mp_pose.PoseLandmark.RIGHT_SHOULDER
    ]
    elbow = pose_landmarks.landmark[
        mp_pose.PoseLandmark.LEFT_ELBOW if side == "left" else mp_pose.PoseLandmark.RIGHT_ELBOW
    ]
    wrist = pose_landmarks.landmark[
        mp_pose.PoseLandmark.LEFT_WRIST if side == "left" else mp_pose.PoseLandmark.RIGHT_WRIST
    ]

    near_hip = hands_on_hips(pose_landmarks, wrist_landmark, side, threshold)
    angle = arm_angle(shoulder, elbow, wrist)
    return near_hip and angle > 90