AI Animated Filter (Multi-Face Support)

This project uses OpenCV and MediaPipe to apply a "Thug Life" style glasses and mustache filter on multiple faces in real time using your webcam. The filters are applied based on facial landmarks, supporting up to 5 faces simultaneously.

 Features

- Real-time webcam feed.
- Detects up to 5 faces using MediaPipe FaceMesh
- Places transparent PNG glasses and mustache filters accurately using facial landmarks.
- Smooth overlay using alpha blending.

 Requirements

Install the following Python libraries before running the script:


pip install opencv-python mediapipe numpy


 Project Structure


glass/
├── main.py              # Main filter script
├── glasses.png          # Transparent thug glasses image
├── mustache.png         # Transparent mustache image
└── README.md            # Project documentation


 Usage

1. Clone the repository:


   git clone https://github.com/krish0827/AI-animated-filter.git
   cd AI-animated-filter


2. Make sure `glasses.png` and `mustache.png` are present in the same directory as `main.py`. Recommended image size: **1200x1200px** with transparent background.

3. Run the script:


   python main.py


4. Press `Esc` to quit the app.
