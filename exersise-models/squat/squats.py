import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (a, b, c).
    b is considered the 'center' point.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Angle in radians via difference of arc tangents
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # Adjust if angle is > 180
    if angle > 180.0:
        angle = 360.0 - angle

    return angle


def get_squat_stage(knee_angle):
    """
    Returns a string indicating the stage of the squat:
    - 'Up' if angle is near 170-180
    - 'Down' if angle is near 90 or lower
    - 'Mid' otherwise
    """
    if knee_angle > 150:  # adjust as needed
        return "Up"
    elif knee_angle < 100:  # adjust as needed
        return "Down"
    else:
        return "Mid"

def squat_tracker():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Mediapipe Pose
    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    # Squat counter & stage
    squat_count = 0
    squat_stage = None   # can be 'Up' or 'Down'

    # Start video capture
    cap = cv2.VideoCapture(0)

    # Optional: set capture resolution
    width, height = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # Convert back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract pose landmarks if available
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates of left hip, knee, and ankle
            # You can do the same for the right side or take an average if you want both legs
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

            # Calculate the knee angle (hip-knee-ankle)
            knee_angle = calculate_angle(hip, knee, ankle)

            # Determine the stage (Up, Down, or Mid)
            current_stage = get_squat_stage(knee_angle)

            # Initialize squat_stage if None
            if squat_stage is None:
                squat_stage = current_stage

            # Check transitions for counting
            if current_stage == "Down" and squat_stage == "Up":
                squat_stage = "Down"
            elif current_stage == "Up" and squat_stage == "Down":
                # Completed one full squat
                squat_count += 1
                squat_stage = "Up"

            # ==================
            # Display on screen
            # ==================
            cv2.putText(image, f'Knee Angle: {knee_angle:.1f}',
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.putText(image, f'Stage: {current_stage}',
                        (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(image, f'Squat Count: {squat_count}',
                        (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw Mediapipe landmarks
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        except Exception as e:
            print(f"Exception: {e}")

        # Show frame
        cv2.imshow('Squat Tracker', image)

        # Quit on 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()