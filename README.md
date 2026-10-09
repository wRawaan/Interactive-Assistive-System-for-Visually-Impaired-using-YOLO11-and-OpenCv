# Interactive Assistive System for Visually Impaired Using YOLO11 and OpenCV

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8/11](https://img.shields.io/badge/Model-YOLO11%20%2F%20YOLOv8-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

An advanced, real-time assistive technology solution designed to improve the independence, safety, and spatial awareness of visually impaired individuals through intelligent computer vision and audio feedback.

---

## 🚀 Project Overview

The **Interactive Assistive System for Visually Impaired** processes live video feeds to interpret the user's surroundings and translates them into comprehensive audio-verbal descriptions. By combining deep learning object detection, monocular depth estimation, and color profiling, the system delivers rich contextual details about nearby objects.

### Core Architecture & Capabilities
* **Real-Time Object Detection:** Identifies everyday items, furniture, and people from a live webcam feed using a custom-trained YOLO model.
* **Monocular Depth Estimation:** Leverages the **MiDaS** model to compute the distance between the user and detected objects for spatial orientation.
* **Color Recognition:** Analyzes the dominant color of detected objects to provide richer verbal descriptors.
* **Height Approximation:** Calculates approximate physical heights relative to the scene.
* **Text-to-Speech (TTS) Feedback:** Uses `pyttsx3` to speak out real-time alerts containing object type, color, distance, and size.
* **Voice Command Integration:** Built-in `speech_recognition` allows hands-free control (starting and stopping detection via voice prompts).
* **Interactive Graphical User Interface (GUI):** A clean Tkinter interface offering manual overrides, dynamic confidence sliders, and live status displays.
* **Multithreaded Performance:** Ensures simultaneous object detection, audio feedback, and speech recognition without UI freezing.

---

## 🛠️ Features

1. Real-time object detection using a custom-trained YOLO model.
2. Accurate distance calculation and depth estimation using MiDaS.
3. Dominant color extraction for enhanced object descriptions.
4. Approximate object height calculations.
5. Automated text-to-speech feedback system.
6. Hands-free voice command support.
7. User-friendly Tkinter GUI with adjustable confidence thresholds.
8. Multithreaded execution for zero-lag performance.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
Open your terminal and run the following commands:
```bash
git clone [https://github.com/wRawaan/Interactive-Assistive-System-for-Virtually-Impaired-using-YOLO11-and-OpenCv.git](https://github.com/wRawaan/Interactive-Assistive-System-for-Virtually-Impaired-using-YOLO11-and-OpenCv.git)
cd Interactive-Assistive-System-for-Virtually-Impaired-using-YOLO11-and-OpenCV

### 2. Install Dependencies
Ensure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt

3. Run the Application
Launch the assistant by executing:

Bash
python vision_assistant.py
📁 Dataset
Due to file size constraints and repository optimization limits, the training and testing datasets are hosted externally.

Download Dataset from Google Drive

📊 Sample Output
The system outputs verbal text descriptions concurrently through audio (TTS) and displays them in the active GUI console:

"A Table of gray color, about 31 cm tall, is 0.5 feet away."

"A Mobile phone of black color, about 25 cm tall, is 0.5 feet away."

"A Bottle of black color, about 14 cm tall, is 0.5 feet away."

🤝 Acknowledgements
We express our gratitude to the open-source community and creators of the foundational tools powering this project:

Ultralytics for the YOLO object detection framework.

Intel ISL for the MiDaS depth estimation model.

OpenCV & NumPy for foundational computer vision and matrix operations.

SpeechRecognition & pyttsx3 for enabling voice interaction and speech synthesis.

Tkinter for providing a lightweight and functional GUI interface.

📝 License
This project is open-source and available under the MIT License.