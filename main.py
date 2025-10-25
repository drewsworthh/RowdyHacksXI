import cv2
import mediapipe as mp

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

    # --- Show the combined result ---
    cv2.imshow('MediaPipe Pose (Wrists + Ankles Only for Hands/Feet)', cv2.flip(image, 1))

    # --- Exit on ESC ---
    if cv2.waitKey(5) & 0xFF == 27:
        break

# --- Clean up ---
cap.release()
cv2.destroyAllWindows()
