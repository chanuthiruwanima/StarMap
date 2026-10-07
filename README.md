### StarMap 
## Gesture-Controlled Planetarium

A gesture-controlled star map planetarium built with **Python**, **OpenCV**, **MediaPipe**, and **Pygame**.
Click 'q' on your keyboard/ the close button of the pygame window to quit the program 

Pan and zoom to navigate through the sky.
Lock onto stars and discover information regarding their magnitude, celestial coordinates in RA and DEC, and colour.

## Gesture Controls
- Move Palm: Pan camera across the sky
- Thumb & Index Pinch/Expand: Zoom in / Zoom out
- Center Crosshair Alignment: Lock target onto a star to open HUD info card

## To install

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/gesture-planetarium.git](https://github.com/your-username/gesture-planetarium.git)
   cd gesture-planetarium

2. **Install dependencies**
   ```bash
   pip install opencv-python mediapipe pygame

3. **Download the MediaPipe Model:**
   Download the pre-trained hand_landmarker.task model file from MediaPipe
   [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

4. **Run the main application script:**
     ```bash
     python starmap.py
     
