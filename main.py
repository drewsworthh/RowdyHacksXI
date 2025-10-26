import cv2
import mediapipe as mp
import math
from pose_utils import (
    mp_pose, mp_hands, mp_drawing, mp_drawing_styles,
    is_fist, is_pointing, arm_angle, spongebob_pose
)
from chin_touch import chin_touch_pose
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

fedora_active = False
# --- Setup camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value
LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value
RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value

print("Press ESC to quit.")

spongebob_active = False  # <-- add this before your while loop
chin_active = False  # <-- NEW STATE FLAG
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
    
    # --- NEW UTILITY FUNCTION FOR FEDORA TIP POSE ---
    def tips_fedora_pose(pose_landmarks, wrist_landmark, side="left", threshold_y=0.1, angle_min=120):
        """
        Checks for a hat-tipping-like pose:
        1. Wrist is vertically close to the eye/ear area.
        2. Elbow angle is obtuse to suggest a raised/extended arm.
        """
        if not pose_landmarks:
            return False

        # 1. Wrist vertical proximity check (to the head/face)
        if side == "left":
            ear = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
            shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            pose_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        else: # right side
            ear = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
            shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            pose_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Check if the wrist is vertically aligned with the ear/eye area.
        # We'll check if the wrist's Y coordinate is between the shoulder and the ear.
        # This helps filter out hands on hips or hands fully overhead.
        y_min = ear.y - threshold_y / 2
        y_max = shoulder.y + threshold_y / 2
        is_vertical_proximity = y_min < wrist_landmark.y < y_max

        # 2. Elbow Angle Check (Obtuse/Wide)
        angle = arm_angle(shoulder, elbow, pose_wrist)
        is_obtuse_angle = angle > angle_min # Use a higher angle like 120 for a more "tippy" look.
        
        return is_vertical_proximity and is_obtuse_angle
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

    # (your pose and hand detection code...)

    show_spongebob = False  # reset for each frame
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
            # Get the wrist landmark from the hand detector results
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            # Check for pointing above lip line
            if is_pointing(hand_landmarks) and pose_results.pose_landmarks:
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                if index_tip.y < lip_line_y - 0.05:
                    cv2.putText(image, "👉 Pointing above lip line!", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            if not is_fist(hand_landmarks) and pose_results.pose_landmarks:
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                if spongebob_pose(pose_results.pose_landmarks, wrist, "left", 0.12) or \
                   spongebob_pose(pose_results.pose_landmarks, wrist, "right", 0.12):
                    show_spongebob = True
                    if not spongebob_active:  # only open window once
                        spongebob_img = cv2.imread("spongebob.jpg")
                        if spongebob_img is not None:
                            spongebob_img = cv2.resize(spongebob_img, (750, 600))
                            cv2.imshow("SpongeBob Pose!", spongebob_img)
                            spongebob_active = True
                        # --- Chin-touch pose detection ---
            if chin_touch_pose(pose_results.pose_landmarks, hand_landmarks):
                if not chin_active:
                    chin_img = cv2.imread("chin_touch.jpg")  # your image
                    if chin_img is not None:
                        chin_img = cv2.resize(chin_img, (400, 400))
                        cv2.imshow("Chin Touch Pose!", chin_img)
                        chin_active = True
            else:
                if chin_active:
                    cv2.destroyWindow("Chin Touch Pose!")
                    chin_active = False
            # --- NEW: Check for Tips Fedora Pose ---
            # We assume a closed/mostly closed hand (fist-like) for the tip.
            fedora_detected = False
            if is_fist(hand_landmarks):
                # Check against both sides:
                if tips_fedora_pose(pose_results.pose_landmarks, wrist, "left", 0.1, 110) or \
                tips_fedora_pose(pose_results.pose_landmarks, wrist, "right", 0.1, 110):
                    cv2.putText(image, "🎩 Tips Fedora!", (30, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    fedora_detected = True
        # --- Manage SpongeBob window state ---
        if not show_spongebob and spongebob_active:
            cv2.destroyWindow("SpongeBob Pose!")
            spongebob_active = False

            # Reset fedora_active if the pose is no longer detected
            if not fedora_detected:
                fedora_active = False
            else:
                fedora_active = True
                


    # --- Show the combined result ---
    cv2.imshow('MediaPipe Pose (Wrists + Ankles Only for Hands/Feet)', cv2.flip(image, 1))

    if cv2.waitKey(5) & 0xFF == 27:
        break

# --- Clean up ---
cap.release()
cv2.destroyAllWindows()
