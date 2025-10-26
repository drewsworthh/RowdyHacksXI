import cv2
import mediapipe as mp
import math
from pose_utils import (
    mp_pose, mp_hands, mp_drawing, mp_drawing_styles,
    is_fist, is_pointing, spongebob_pose
)
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
