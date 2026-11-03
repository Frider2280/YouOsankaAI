import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerOptions
import time




def camera():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Could not open camera")
        return

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=r"C:\Program Files\JetBrains\PycharmProjects\TvoiOsanka\pose_landmarker_full.task"
        ),
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    y_eyes_nice = None
    y_shoulders_nice = None

    while True:
        success, frame = video.read()

        if success:
            frame = cv2.flip(frame, 1)

            timestamp_ms = time.monotonic_ns() // 1_000_000
            size = frame.shape

            height = size[0]
            width = size[1]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                left_eye = result.pose_landmarks[0][2]
                right_eye = result.pose_landmarks[0][5]
                left_shoulder = result.pose_landmarks[0][11]
                right_shoulder = result.pose_landmarks[0][12]

                x_l_e = int(left_eye.x * width)
                y_l_e = int(left_eye.y * height)
                x_r_e = int(right_eye.x * width)
                y_r_e = int(right_eye.y * height)

                cv2.circle(frame, (x_l_e, y_l_e), 10, (0, 255, 0), -1)
                cv2.circle(frame, (x_r_e, y_r_e), 10, (0, 255, 0), -1)
                cv2.line(frame, (x_l_e, y_l_e), (x_r_e, y_r_e), (255, 255, 255), 2)

                x_l_s = int(left_shoulder.x * width)
                y_l_s = int(left_shoulder.y * height)
                x_r_s = int(right_shoulder.x * width)
                y_r_s = int(right_shoulder.y * height)

                cv2.circle(frame, (x_l_s, y_l_s), 10, (0, 255, 0), -1)
                cv2.circle(frame, (x_r_s, y_r_s), 10, (0, 255, 0), -1)
                cv2.line(frame, (x_l_s, y_l_s), (x_r_s, y_r_s), (255, 255, 255), 2)

                y_eyes = (left_eye.y + right_eye.y) / 2
                y_shoulders = (left_shoulder.y + right_shoulder.y) / 2


                if y_eyes_nice is not None and y_shoulders_nice is not None:
                    if y_eyes - y_eyes_nice > 0.05:
                        cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (255, 0, 0), 3)



            cv2.imshow('camera', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and result.pose_landmarks:
                print("Pt fix...")
                y_eyes_nice = y_eyes
                y_shoulders_nice = y_shoulders

            if key == ord('q') or key == 27:
                break




        else:
            break

    video.release()
    cv2.destroyAllWindows()
    landmarker.close()


camera()