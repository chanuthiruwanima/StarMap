### StarMap 
## Gesture-Controlled Planetarium

An interactive, gesture-controlled 2D star map planetarium built with **Python**, **OpenCV**, **MediaPipe**, and **Pygame**.
Navigate celestial star catalogs using real-time webcam hand tracking—pan across the night sky, zoom into stellar clusters, and lock onto bright stars to display an astronomical HUD overlay.

## Features
- **Gesture Camera Control**: Real-time panning and zooming powered by MediaPipe hand landmarker detection.
- **Astronomical Catalog Projection**: Maps Right Ascension (RA) and Declination (Dec) celestial coordinates into a 2D viewport.
- **Visual Star Magnitudes**: Renders star radius and spectral color temperatures based on catalog visual magnitudes.
- **Constellation Vector Overlays**: Connects named star pairs to illustrate major constellations (e.g., Orion, Summer Triangle, Southern Cross).
- **Targeting Reticle & Info HUD**: Locks onto stars near the screen center, rendering a high-tech HUD overlay with magnitude, RA, and Dec data.

## Prerequisites & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/gesture-planetarium.git](https://github.com/your-username/gesture-planetarium.git)
   cd gesture-planetarium

2. **Install dependencies**
   ```bash
   pip install opencv-python mediapipe pygame

3. **Download the MediaPipe Model:**
   Download the pre-trained hand_landmarker.task model file from MediaPipe and place it in the project root directory:
   [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

4. **Run the main application script:**
     ```bash
     python starmap.py
     
## Gesture Controls
- Move Palm: Pan camera across Right Ascension and Declination
- Thumb & Index Pinch/Expand: Zoom in / Zoom out
- Center Crosshair Alignment: Lock target onto a star to open HUD info card

## Project Architecture 
├── main.py                   # Main loop & integration logic
├── hand_landmarker.task      # MediaPipe model file
└── README.md                 # Project documentation
