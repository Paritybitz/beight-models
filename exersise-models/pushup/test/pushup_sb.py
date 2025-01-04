import cv2
import mediapipe as mp
import numpy as np

# =====================
# 1. ANGLE CALCULATION
# =====================
def calculate_angle(a, b, c):
    """
    Calculate angle between three points (a, b, c).
    b is considered the vertex point.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

# ======================
# 2. STAGE DETERMINATION
# ======================
def current_stage(angle):
    """
    Based on the elbow angle, determine if the user is 'Up', 'Down', or 'Mid'.
    Adjust thresholds as needed for your push-up form.
    """
    if angle >= 130:
        return 'Up'
    elif angle <= 90:
        return 'Down'
    else: 
        return "Mid"

def posture_check(shoulder, hip, knee, ankle):
    """
    Check if body is relatively straight by calculating hip and knee angles.
    This can help you give posture feedback.
    """
    hip_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)

    if 160 <= hip_angle <= 200 and 160 <= knee_angle <= 200:
        return "Straight", hip_angle, knee_angle
    else:
        return "Bent", hip_angle, knee_angle

# =============================
# 3. MAIN PUSH-UP TRACKER LOGIC
# =============================
def pushup_tracker():
    # Initialize Mediapipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Push-up counter & stage
    pushup_counter = 0
    pushup_stage = None   # can be 'Up' or 'Down'

    # Video capture
    cap = cv2.VideoCapture(0)

    width = 1280
    height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Convert color space for Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # 2) Convert color space back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # =============
        # 3) LANDMARKS
        # =============
        try:
            landmarks = results.pose_landmarks.landmark

            # Extract relevant points
            shoulder = [
                int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width),
                int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height)
            ]
            elbow = [
                int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width),
                int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height)
            ]
            wrist = [
                int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width),
                int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height)
            ]

            hip = [
                int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width),
                int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height)
            ]
            knee = [
                int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width),
                int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height)
            ]
            ankle = [
                int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width),
                int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height)
            ]

            # ===============
            # 4) CALCULATIONS
            # ===============
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            stage = current_stage(elbow_angle)
            posture, hip_angle, knee_angle = posture_check(shoulder, hip, knee, ankle)

            # ======================
            # 5) PUSH-UP COUNTER
            # ======================
            # - We consider a single repetition to be "Up -> Down -> Up"
            # - This means we only increment the counter upon transitioning
            #   from Down back to Up.
            if pushup_stage is None:
                # Initialize stage when first detected
                pushup_stage = stage

            if stage == "Down" and pushup_stage == "Up":
                # User has moved from Up to Down
                pushup_stage = "Down"

            elif stage == "Up" and pushup_stage == "Down":
                # User has moved from Down to Up => Complete rep
                pushup_counter += 1
                pushup_stage = "Up"

            # ======================
            # 6) DISPLAY ON SCREEN
            # ======================
            cv2.putText(image, f'Stage: {stage}',
                        (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'Posture: {}'.format(posture), 
                        (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(image, 'Elbow Angle: {:.2f}'.format(elbow_angle), 
                        (50, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'Hip Angle: {:.2f}'.format(hip_angle), 
                        (50, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'Knee Angle: {:.2f}'.format(knee_angle), 
                        (50, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(image, f'Push-Ups: {pushup_counter}',
                        (50, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

            # Render Mediapipe landmarks
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        except Exception as e:
            print(f"Exception: {e}")

        # 7) Show the output frame
        cv2.imshow('Push-Up Tracker', image)

        # 8) Break the loop on 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
