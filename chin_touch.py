import math
import mediapipe as mp
from pose_utils import (
    mp_pose, mp_hands
)
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


def chin_touch_pose(pose_landmarks, hand_landmarks, threshold=0.05):
    """
    Detects a 'chin-touch' gesture when the index finger tip is close to
    the approximate jawline (below the mouth).
    """
    if not pose_landmarks or not hand_landmarks:
        return False

    # --- Estimate chin midpoint just below the mouth ---
    mouth_left = pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
    mouth_right = pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]
    chin_x = (mouth_left.x + mouth_right.x) / 2
    chin_y = (mouth_left.y + mouth_right.y) / 2 + 0.07  # adjust if needed

    # --- Get index finger tip coordinates ---
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # --- Euclidean distance between finger and chin ---
    dx = index_tip.x - chin_x
    dy = index_tip.y - chin_y
    distance = math.sqrt(dx**2 + dy**2)

    # --- Trigger when close enough to chin area ---
    return distance < threshold