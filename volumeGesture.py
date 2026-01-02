import cv2
import time
import math
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class HandVolumeController:
    """Hand gesture-based volume control using MediaPipe and Pycaw."""
    
    def __init__(self, model_path="hand_landmarker.task"):
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # If model_path is just a filename, look in script directory
        if not os.path.isabs(model_path):
            model_path = os.path.join(script_dir, model_path)
        
        # Audio setup
        self.setup_audio()
        
        # MediaPipe setup
        self.setup_hand_detector(model_path)
        
        # Camera setup
        self.setup_camera()
        
        # Control parameters
        self.smooth_volume = self.volume.GetMasterVolumeLevel()
        self.smoothing_factor = 0.2
        self.min_distance = 40
        self.max_distance = 200
        
        # UI parameters
        self.prev_time = 0
        
    def setup_audio(self):
        """Initialize system audio controls."""
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None
            )
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            self.min_vol, self.max_vol, _ = self.volume.GetVolumeRange()
            print("✓ Audio system initialized")
        except Exception as e:
            print(f"❌ Audio initialization failed: {e}")
            raise
    
    def setup_hand_detector(self, model_path):
        """Initialize MediaPipe hand detector."""
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            print(f"📂 Loading model from: {model_path}")
            
            base_options = BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.hand_detector = vision.HandLandmarker.create_from_options(options)
            print("✓ Hand detector initialized")
        except Exception as e:
            print(f"❌ Hand detector initialization failed: {e}")
            raise
    
    def setup_camera(self, width=1280, height=720):
        """Initialize camera with specified resolution."""
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
            print("❌ Webcam not accessible")
            raise RuntimeError("Cannot access webcam")
        print("✓ Camera initialized")
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    
    def update_volume(self, distance):
        """Update system volume based on finger distance."""
        target_vol = np.interp(
            distance,
            [self.min_distance, self.max_distance],
            [self.min_vol, self.max_vol]
        )
        
        # Smooth volume changes
        self.smooth_volume += (target_vol - self.smooth_volume) * self.smoothing_factor
        
        try:
            self.volume.SetMasterVolumeLevel(self.smooth_volume, None)
        except Exception as e:
            print(f"⚠ Volume update failed: {e}")
    
    def draw_ui(self, frame, distance=None):
        """Draw UI elements on the frame."""
        h, w = frame.shape[:2]
        
        if distance is not None:
            # Volume bar
            vol_bar = np.interp(distance, [self.min_distance, self.max_distance], [400, 150])
            vol_percent = np.interp(distance, [self.min_distance, self.max_distance], [0, 100])
            
            cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (0, 255, 0), -1)
            cv2.putText(
                frame, f"{int(vol_percent)}%",
                (40, 430), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )
        
        # FPS counter
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if self.prev_time else 0
        self.prev_time = curr_time
        
        cv2.putText(
            frame, f"FPS: {int(fps)}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 2
        )
        
        # Instructions
        cv2.putText(
            frame, "Pinch to control volume | ESC to exit",
            (w // 2 - 250, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1
        )
    
    def process_hand(self, frame, hand_landmarks):
        """Process detected hand and update volume."""
        h, w = frame.shape[:2]
        
        # Get thumb tip (4) and index tip (8)
        thumb = hand_landmarks[4]
        index = hand_landmarks[8]
        
        x1, y1 = int(thumb.x * w), int(thumb.y * h)
        x2, y2 = int(index.x * w), int(index.y * h)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Draw landmarks
        cv2.circle(frame, (x1, y1), 10, (255, 0, 255), -1)
        cv2.circle(frame, (x2, y2), 10, (255, 0, 255), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
        
        # Calculate distance and update volume
        distance = self.calculate_distance((x1, y1), (x2, y2))
        self.update_volume(distance)
        
        return distance
    
    def run(self):
        """Main control loop."""
        print("\n🎮 Hand Gesture Volume Control Active")
        print("   Pinch thumb and index finger to control volume")
        print("   Press ESC to exit\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠ Failed to read frame")
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                
                # Detect hands
                result = self.hand_detector.detect(mp_image)
                
                distance = None
                if result.hand_landmarks:
                    distance = self.process_hand(frame, result.hand_landmarks[0])
                
                # Draw UI
                self.draw_ui(frame, distance)
                
                # Display
                cv2.imshow("Hand Gesture Volume Control", frame)
                
                # Exit on ESC
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                    
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Resources released")


if __name__ == "__main__":
    try:
        controller = HandVolumeController()
        controller.run()
    except Exception as e:
        print(f"❌ Failed to start: {e}")