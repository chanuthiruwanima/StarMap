import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

class GestureController():
    def __init__(self, model_path="hand_landmarker.task"):
            base_options = python.BaseOptions(model_asset_path=model_path)
            
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.7,
                min_tracking_confidence=0.7
            )

            self.landmarker = vision.HandLandmarker.create_from_options(options)

    def frame_processing(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return result
    
    def draw_hands(self, frame, detection_result):
         if not detection_result.hand_landmarks:
            return frame
         h, w, _ = frame.shape
         hand_landmarks = detection_result.hand_landmarks[0]
         for landmark in hand_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        
         return frame

    def close(self):
        self.landmarker.close()

    def camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
             print("Cannot open camera")
             return

        while True:
             ret,frame = cap.read()
             if not ret:
                  print("Cannot grab frame")
                  break

             frame = cv2.flip(frame,1)
             result = self.frame_processing(frame)
             frame = self.draw_hands(frame, result)

             cv2.imshow("Webcam Stream", frame)

             if cv2.waitKey(1) & 0xFF == ord('q'):
                  break         
        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
     controller = GestureController("hand_landmarker.task")
     try:
        controller.camera()
     finally:
        controller.close()