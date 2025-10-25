import cv2
import mediapipe as mp
# functions
def is_fist(hand_landmarks):
    """Returns True if all fingers are curled (fist)."""
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
    
    curled = 0
    for tip, pip in zip(tips, pips):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            curled += 1

    return curled >= 4  # at least 4 fingers curled

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

# --- Initialize MediaPipe modules ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

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

        for connection in mp_pose.POSE_CONNECTIONS:
            start, end = connection

            # Skip all hand-related points except wrists
            skip_hand_points = [
                mp_pose.PoseLandmark.LEFT_THUMB.value, mp_pose.PoseLandmark.LEFT_INDEX.value,
                mp_pose.PoseLandmark.LEFT_PINKY.value, mp_pose.PoseLandmark.RIGHT_THUMB.value,
                mp_pose.PoseLandmark.RIGHT_INDEX.value, mp_pose.PoseLandmark.RIGHT_PINKY.value,
            ]
            if start in skip_hand_points or end in skip_hand_points:
                continue

            # Skip foot-related points except ankles
            skip_foot_points = [
                mp_pose.PoseLandmark.LEFT_HEEL.value, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
                mp_pose.PoseLandmark.RIGHT_HEEL.value, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
            ]
            if start in skip_foot_points or end in skip_foot_points:
                continue

            # Draw remaining pose lines
            h, w, _ = image.shape
            start_point = pose_landmarks[start]
            end_point = pose_landmarks[end]
            x1, y1 = int(start_point.x * w), int(start_point.y * h)
            x2, y2 = int(end_point.x * w), int(end_point.y * h)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
            # --- Check if hand is pointing ---
            if is_pointing(hand_landmarks):
                cv2.putText(image, "POINTING!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print("👉 Pointing detected!")
            if pose_results.pose_landmarks and hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    if is_fist(hand_landmarks):
                        # get wrist position from the hand
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                        # choose which side (you can determine this via handedness if needed)
                        left_hip_near = hands_on_hips(pose_results.pose_landmarks, wrist, "left")
                        right_hip_near = hands_on_hips(pose_results.pose_landmarks, wrist, "right")

                        if left_hip_near or right_hip_near:
                            cv2.putText(image, "Fist near hip!", (30, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


    # --- Show the combined result ---
    cv2.imshow('MediaPipe Pose (Wrists + Ankles Only for Hands/Feet)', cv2.flip(image, 1))

    # --- Exit on ESC ---
    if cv2.waitKey(5) & 0xFF == 27:
        break

# --- Clean up ---
cap.release()
cv2.destroyAllWindows()
