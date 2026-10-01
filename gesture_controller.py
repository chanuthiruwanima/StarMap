import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import math

#loading the pre-trained model and setting contstraints
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

            self.prev_palm_x = None
            self.prev_palm_y = None

    def frame_processing(self, frame):
        #coverts the BGR capture of cv2 into RGB for processing with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        #Recording timestamps for detect_for_video
        timestamp_ms = int(time.time() * 1000)
        #returns hand landmark coordinates
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return result
    
    def draw_hands(self, frame, detection_result):
         if not detection_result.hand_landmarks:
            return frame
         h, w, _ = frame.shape
         hand_landmarks = detection_result.hand_landmarks[0]
         
         for landmark in hand_landmarks:
            #coordinate de-normalization
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            #draw green dot at landmark
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
           #rendering of landmarks
             frame = cv2.flip(frame,1)
             result = self.frame_processing(frame)
             frame = self.draw_hands(frame, result)

             if result.hand_landmarks and len(result.hand_landmarks)>0:
                 #testing
                 hand_landmarks = result.hand_landmarks[0]
                 zoom_distance = self.get_zoom_distance(hand_landmarks)
                 pan_dx,pan_dy = self.get_change_palm(hand_landmarks)

                 print (f"Zoom:{zoom_distance:.4f}" )
                 print (f"Pan_dx:{pan_dx:.4f}  Pan_dy:{pan_dy:.4f}" )
             else:
                 #reset tracking history if palm is no longer detected
                 self.prev_palm_x = None
                 self.prev_palm_y = None
                 print("No hand detected")

             cv2.imshow("Webcam Stream", frame)

             if cv2.waitKey(1) & 0xFF == ord('q'):
                  break         
        cap.release()
        cv2.destroyAllWindows()

    def get_zoom_distance(self, hand_landmarks):
        thumb = hand_landmarks[4]
        index = hand_landmarks[8]

        #calculate distance between thumb and index finger
        distance = math.hypot(index.x - thumb.x, index.y - thumb.y)
        return distance

    def get_change_palm(self, hand_landmarks):
        palm_center = hand_landmarks[9]
        curr_x, curr_y = palm_center.x, palm_center.y
        pan_dx, pan_dy = 0,0 

        #calculating movement of the palm to pan display
        if self.prev_palm_x is not None and self.prev_palm_y is not None: 
            pan_dx = curr_x - self.prev_palm_x
            pan_dy = curr_y - self.prev_palm_y

        #recentering the origin of the palm to the new coordinates
        self.prev_palm_x = curr_x
        self.prev_palm_y = curr_y

        return pan_dx, pan_dy



if __name__=="__main__":
     controller = GestureController("hand_landmarker.task")
     try:
        controller.camera()

     finally:
        controller.close()