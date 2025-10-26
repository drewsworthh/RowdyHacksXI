import cv2
import mediapipe as mp
import math
# --- Initialize MediaPipe modules ---
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
    # get all tips and pips
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    # check extension of index and curling of rest
    index_extended = index_tip.y < index_pip.y

    middle_curled = middle_tip.y > middle_pip.y - 0.05
    ring_curled = ring_tip.y > ring_pip.y - 0.05
    pinky_curled = pinky_tip.y > pinky_pip.y - 0.05

    if index_extended and middle_curled and ring_curled and pinky_curled:
        return True
    
    return False
def hands_on_hips(pose_landmarks, wrist_landmark, side="left", threshold=0.1):
    """Returns True if wrist is near the hip joint."""
    if side == "left":
        hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    else:
        hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    dx = wrist_landmark.x - hip.x
    dy = wrist_landmark.y - hip.y
    distance = (dx**2 + dy**2) ** 0.5
    return distance < threshold

def arm_angle(shoulder, elbow, wrist):
    """Compute the elbow angle in degrees."""
    a = [shoulder.x - elbow.x, shoulder.y - elbow.y]
    b = [wrist.x - elbow.x, wrist.y - elbow.y]
    dot = a[0] * b[0] + a[1] * b[1]
    mag_a = math.sqrt(a[0] ** 2 + a[1] ** 2)
    mag_b = math.sqrt(b[0] ** 2 + b[1] ** 2)
    if mag_a * mag_b == 0:
        return 0
    angle = math.degrees(math.acos(dot / (mag_a * mag_b)))
    return angle

def spongebob_pose(pose_landmarks, wrist_landmark, side="left", threshold=0.15):
    """Returns True if fist is near hip and arm is obtuse (>90°)."""
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

# --- Create Pose and Hands detectors ---
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Setup camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value
LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value
RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value

print("Press ESC to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # --- Run both Pose and Hands models ---
    pose_results = pose.process(image_rgb)
    hand_results = hands.process(image_rgb)

    # Convert back to BGR for drawing
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # --- Draw filtered pose landmarks (no hands except wrists, no feet except ankles) ---
    if pose_results.pose_landmarks:
        pose_landmarks = pose_results.pose_landmarks.landmark

        # Calculate "lip line"
        mouth_left = pose_landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value]
        mouth_right = pose_landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value]
        lip_line_y = (mouth_left.y + mouth_right.y) / 2

        # Draw filtered body (no hand/foot beyond wrists/ankles)
        for connection in mp_pose.POSE_CONNECTIONS:
            start, end = connection

            # Skip all hand-related points except wrists
            skip_hand = [
                mp_pose.PoseLandmark.LEFT_THUMB.value, mp_pose.PoseLandmark.LEFT_INDEX.value,
                mp_pose.PoseLandmark.LEFT_PINKY.value, mp_pose.PoseLandmark.RIGHT_THUMB.value,
                mp_pose.PoseLandmark.RIGHT_INDEX.value, mp_pose.PoseLandmark.RIGHT_PINKY.value,
            ]
            skip_foot = [
                mp_pose.PoseLandmark.LEFT_HEEL.value, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
                mp_pose.PoseLandmark.RIGHT_HEEL.value, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
            ]
            if start in skip_hand or end in skip_hand or start in skip_foot or end in skip_foot:
                continue

            if start in skip_hand or end in skip_hand or start in skip_foot or end in skip_foot:
                continue

            # Draw remaining pose lines
            h, w, _ = image.shape
            p1, p2 = pose_landmarks[start], pose_landmarks[end]
            cv2.line(image, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (0, 255, 0), 2)

        # Draw only wrists and ankles as dots
        for index in [LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE]:
            lm = pose_landmarks[index]
            h, w, _ = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 6, (0, 255, 255), -1)

    # --- Draw hand landmarks normally ---
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Check for pointing above lip line
            if is_pointing(hand_landmarks) and pose_results.pose_landmarks:
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                if index_tip.y < lip_line_y - 0.05:
                    cv2.putText(image, "👉 Pointing above lip line!", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            if is_fist(hand_landmarks) and pose_results.pose_landmarks:
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                if spongebob_pose(pose_results.pose_landmarks, wrist, "left", 0.12) or \
                spongebob_pose(pose_results.pose_landmarks, wrist, "right", 0.12):
                    # Load SpongeBob image
                    spongebob_img = cv2.imread("spongebob.jpg")

                    if spongebob_img is not None:
                        # Resize overlay smaller (adjust size if needed)
                        overlay_h, overlay_w = 150, 150
                        spongebob_img = cv2.resize(spongebob_img, (overlay_w, overlay_h))

                        # Choose top-left corner for overlay
                        x_offset, y_offset = 30, 50

                        # Overlay SpongeBob image on top of camera feed
                        y1, y2 = y_offset, y_offset + spongebob_img.shape[0]
                        x1, x2 = x_offset, x_offset + spongebob_img.shape[1]

                        # Ensure overlay fits inside frame
                        if y2 <= image.shape[0] and x2 <= image.shape[1]:
                            alpha_s = 0.8  # transparency (0–1)
                            image[y1:y2, x1:x2] = cv2.addWeighted(
                                image[y1:y2, x1:x2], 1 - alpha_s, spongebob_img, alpha_s, 0
                            )


    # --- Show the combined result ---
    cv2.imshow('MediaPipe Pose (Wrists + Ankles Only for Hands/Feet)', cv2.flip(image, 1))

    # --- Exit on ESC ---
    if cv2.waitKey(5) & 0xFF == 27:
        break

# --- Clean up ---
cap.release()
cv2.destroyAllWindows()
