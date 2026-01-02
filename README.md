# Volume-control-Gesture
Hand Gesture Volume Control — a real-time webcam app that recognizes hand gestures to adjust system volume.
# Hand Gesture Volume Control

A real-time hand gesture recognition application that controls your system volume using pinch gestures. Uses MediaPipe for hand detection and Pycaw for audio control.

## Features

- 🎥 Real-time hand gesture recognition using MediaPipe
- 🔊 Intuitive pinch gesture to control system volume
- 📊 Visual volume bar and percentage display
- 🚀 Smooth volume transitions using exponential smoothing
- 📱 FPS counter for performance monitoring
- ⌨️ Easy exit with ESC key

## Requirements

- Python 3.8+
- Windows OS (Pycaw is Windows-specific)
- Webcam

## Installation

### 1. Clone or download the project

```bash
git clone <repository-url>
cd Hand\ Gesture\ Volume\ Control
```

### 2. Install dependencies

```bash
pip install opencv-python mediapipe pycaw comtypes numpy
```

### 3. Download MediaPipe Hand Landmarker Model

Download the hand landmarker model file (`hand_landmarker.task`) from [MediaPipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/overview) and place it in the same directory as `volumeGesture.py`.

## Usage

### Run the application

```bash
python volumeGesture.py
```

### Controls

1. **Pinch gesture**: Bring your thumb and index finger closer or farther apart to decrease or increase volume
   - Fingers close together = Lower volume
   - Fingers far apart = Higher volume
2. **Exit**: Press `ESC` to quit the application

## How It Works

### Hand Detection
- Uses MediaPipe's HandLandmarker to detect 21 hand landmarks in real-time
- Tracks the position of the thumb tip (landmark 4) and index finger tip (landmark 8)

### Volume Control
- Calculates the Euclidean distance between thumb and index fingertips
- Maps the distance to your system's volume range
- Applies exponential smoothing for smooth, natural volume transitions

### Visualization
- Green circle marks the midpoint between thumb and index finger
- Purple circles mark the thumb and index finger tips
- On-screen volume bar shows current volume percentage
- FPS counter displays real-time performance

## Configuration

Adjust these parameters in the `__init__` method of `HandVolumeController`:

```python
self.smoothing_factor = 0.2  # Smoothness of volume transitions (0.0-1.0)
self.min_distance = 40       # Minimum distance for lowest volume (pixels)
self.max_distance = 200      # Maximum distance for highest volume (pixels)
```

## Troubleshooting

### Audio initialization failed
- Ensure your system has audio devices properly configured
- Try running the script as administrator

### Hand detector initialization failed
- Verify that `hand_landmarker.task` is in the same directory as the script
- Download the latest model from MediaPipe's official website

### Webcam not accessible
- Check if another application is using your webcam
- Try restarting the application
- Verify camera permissions

### Poor hand detection
- Ensure good lighting conditions
- Keep your hand fully visible in the frame
- Adjust `min_hand_detection_confidence` in the code if needed

## Project Structure

```
Hand Gesture Volume Control/
├── volumeGesture.py           # Main application
├── hand_landmarker.task       # MediaPipe model (download separately)
└── README.md                  # This file
```

## Technical Details

### Dependencies

- **OpenCV**: Video capture and frame processing
- **MediaPipe**: Hand landmark detection
- **Pycaw**: Windows audio interface control
- **NumPy**: Numerical operations
- **Comtypes**: COM library integration

### Performance

- Runs at 30-60 FPS depending on hardware
- Optimized for real-time processing with single-hand detection

## System Requirements

- **OS**: Windows 7 or later
- **CPU**: 2+ GHz processor
- **RAM**: 2GB minimum
- **Webcam**: Any standard USB webcam

## License

This project is provided as-is for educational and personal use.

## Future Enhancements

- [ ] Multi-hand gesture support
- [ ] Additional gestures for play/pause, next/previous
- [ ] Customizable gesture mappings
- [ ] Cross-platform support (macOS, Linux)
- [ ] Settings GUI for parameter adjustment

## Contributing

Feel free to fork, modify, and improve this project!

## Support

If you encounter any issues, please ensure:
1. All dependencies are correctly installed
2. The MediaPipe model file is in the correct location
3. Your webcam is working properly
4. You're running on Windows

---

**Created**: January 2026
