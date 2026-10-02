import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import math
import pygame
import random



class GestureController():
    def __init__(self, model_path="hand_landmarker.task"):
            #loading the pre-trained model and setting contstraints
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

class StarField():
    def __init__(self):
        pygame.init()
        self.width = 500
        self.height = 200
        self.screen = pygame.display.set_mode((self.width,self.height))
        pygame.display.set_caption("Starmap")

        #camera tracker variables
        self.cam_x = 0.0 
        self.cam_y = 0.0 
        self.zoom = 1.0

        #creating random stars
        self.stars = []
        for star in range (200):
            star_x = random.uniform(-1500,1500)
            star_y = random.uniform(-1500, 1500)
            brightness = random.randint(150,255)
            self.stars.append((star_x,star_y,brightness))

    def update_camera(self, pan_dx, pan_dy, zoom_distance):
        #pan sensitivity
        self.cam_x += pan_dx * 800.0
        self.cam_y += pan_dy * 800.0

        #clamping zoom level
        zoom_sensitivity = 0.05
        if zoom_distance!=1.0:
            target_zoom = self.zoom + (zoom_distance - 0.15)*zoom_sensitivity
            self.zoom = max(0.2, min(target_zoom, 5.0))

    def render(self):
        self.screen.fill((5,5,12))

        for star_x, star_y, brightness in self.stars:
            #transforming coordinates to map onto the screen
            screen_x = int((star_x-self.cam_x)*self.zoom + (self.width/2.0))
            screen_y = int((star_y-self.cam_y)*self.zoom + (self.height/2.0))

            #check if star is inside screen viewport
            if 0<= screen_x and screen_x < self.width and 0<= screen_y and screen_y < self.height:
                #star radius dependent on zoom scale
                radius = max (1, int(2*self.zoom))
                color = (brightness, brightness, brightness)
                #render star
                pygame.draw.circle(self.screen, color, (screen_x, screen_y), radius)

        #tracking circle at center
        pygame.draw.circle(self.screen, (0,255,0), (int(self.width/2.0), int(self.height/2.0)), 4,1 )

        pygame.display.flip()

    def close(self):
        pygame.quit()

def StarMap():
    controller = GestureController("hand_landmarker.task")
    view = StarField()

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return

    clock = pygame.time.Clock()
    running = True

    try: 
        while running: 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not running:
                break

            ret,frame = cap.read()
            if not ret:
                print("Cannot grab frame")
                break
             
         #rendering of landmarks
            frame = cv2.flip(frame,1)
            result = controller.frame_processing(frame)
            frame = controller.draw_hands(frame, result)

            if result.hand_landmarks and len(result.hand_landmarks)>0:
                 #testing
                 hand_landmarks = result.hand_landmarks[0]
                 zoom_distance = controller.get_zoom_distance(hand_landmarks)
                 pan_dx,pan_dy = controller.get_change_palm(hand_landmarks)

                 print (f"Zoom:{zoom_distance:.4f}" )
                 print (f"Pan_dx:{pan_dx:.4f}  Pan_dy:{pan_dy:.4f}" )
            else:
                 #reset tracking history if palm is no longer detected
                 controller.prev_palm_x = None
                 controller.prev_palm_y = None
                 pan_dx = 0.0
                 pan_dy= 0.0
                 zoom_distance = 1.0
                 print("No hand detected")

            cv2.imshow("Webcam Stream", frame)
            view.update_camera(pan_dx,pan_dy, zoom_distance)
            view.render()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                  break     

            clock.tick(60) 

    finally:
        cap.release()
        cv2.destroyAllWindows()
        controller.close()
        view.close()

if __name__=="__main__":
    StarMap()