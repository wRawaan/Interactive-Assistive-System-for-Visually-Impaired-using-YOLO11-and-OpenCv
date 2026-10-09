# INTERACTIVE ASSISTIVE SYSTEM FOR VISUALLY IMPAIRED USING YOLO11 AND OPENCV


import cv2
import numpy as np
import torch
import pyttsx3
import tkinter as tk
from tkinter import ttk
import speech_recognition as sr
from threading import Thread
from ultralytics import YOLO
import yaml
import time
import os

class VisionAssistant:
    def __init__(self):
        # Initialize variables
        self.detection_running = False
        self.stop_detection_flag = False
        self.stop_voice_thread = False
        self.last_spoken = ""
        self.confidence_threshold = 0.25
        self.last_announcement_time = 0
        self.announcement_interval = 3.0  # Announce every 3 seconds for one-by-one descriptions
        self.current_object_index = 0  # Track which object to announce
        
        # Load models
        self.load_models()
        
        # Setup TTS
        self.setup_tts()
        
        # Setup GUI
        self.setup_gui()
        
        # Start voice command listener
        self.start_voice_listener()
    
    def load_models(self):
        """Load YOLO and MiDaS models"""
        print("Loading models...")
        
        # Load YOLO model with error handling
        model_path = r"path to your model"
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            print("Using default YOLOv8 model instead...")
            self.yolo_model = YOLO('yolov8n.pt')  # Fallback to default model
        else:
            self.yolo_model = YOLO(model_path)
        
        # Load class names with error handling
        yaml_path = r"path to your data yaml "
        try:
            if os.path.exists(yaml_path):
                with open(yaml_path, "r") as stream:
                    data = yaml.safe_load(stream)
                    self.class_names = data["names"]
            else:
                raise FileNotFoundError("YAML file not found")
        except (FileNotFoundError, KeyError):
            # Fallback class names (COCO dataset classes)
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
            print("Using fallback COCO class names")
        
        # Load MiDaS model for depth estimation with error handling
        try:
            device = "cpu"
            model_type = "DPT_Large"
            self.midas_model = torch.hub.load("intel-isl/MiDaS", model_type)
            self.midas_model.to(device)
            self.midas_model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if model_type in ("DPT_Large", "DPT_Hybrid"):
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            
            self.depth_estimation_available = True
            print("MiDaS depth estimation model loaded successfully!")
            
        except Exception as e:
            print(f"Warning: Could not load MiDaS model: {e}")
            print("Continuing without depth estimation...")
            self.depth_estimation_available = False
            self.midas_model = None
            self.transform = None
            
        print("Models loaded successfully!")
    
    def setup_tts(self):
        """Setup text-to-speech engine"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            self.tts_available = True
            self.tts_busy = False  # Flag to prevent concurrent TTS calls
        except Exception as e:
            print(f"Warning: Could not initialize TTS engine: {e}")
            self.tts_available = False
            self.engine = None
            self.tts_busy = False
    
    def get_basic_color_name(self, rgb):
        """Get basic color name from RGB values"""
        colors = {
            'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255),
            'yellow': (255, 255, 0), 'orange': (255, 165, 0), 'purple': (128, 0, 128),
            'pink': (255, 192, 203), 'brown': (150, 75, 0), 'black': (0, 0, 0),
            'white': (255, 255, 255), 'gray': (128, 128, 128)
        }
        min_dist = float('inf')
        closest = "unknown"
        for name, c_rgb in colors.items():
            dist = np.linalg.norm(np.array(rgb) - np.array(c_rgb))
            if dist < min_dist:
                min_dist = dist
                closest = name
        return closest
    
    def estimate_distance_simple(self, box_area, frame_area):
        """Simple distance estimation based on bounding box size"""
        # Simple heuristic: larger objects are closer
        size_ratio = box_area / frame_area
        if size_ratio > 0.3:
            return "very close", 0.5
        elif size_ratio > 0.1:
            return "close", 1.5
        elif size_ratio > 0.05:
            return "medium distance", 3.0
        else:
            return "far", 6.0
    
    def estimate_height_realistic(self, box_height_pixels, distance_ft, label):
        """Estimate realistic height based on object type and distance"""
        # Rough pixel-to-real-world scaling based on distance
        # Closer objects appear larger in pixels
        base_scaling = 0.3 + (distance_ft * 0.1)  # Scale increases with distance
        
        if label.lower() == "person":
            # For people, use more realistic scaling
            # Average person height is 165-175 cm
            estimated_height = max(150, min(200, box_height_pixels * base_scaling))
        elif label.lower() in ["chair", "couch", "bench"]:
            # Furniture typically 80-120 cm
            estimated_height = max(40, min(120, box_height_pixels * base_scaling * 0.6))
        elif label.lower() in ["bottle", "cup"]:
            # Small objects 10-30 cm
            estimated_height = max(5, min(35, box_height_pixels * base_scaling * 0.2))
        elif label.lower() in ["laptop", "book", "keyboard"]:
            # Tech items 2-30 cm
            estimated_height = max(2, min(30, box_height_pixels * base_scaling * 0.15))
        elif label.lower() in ["car", "truck", "bus"]:
            # Vehicles 150-300 cm
            estimated_height = max(120, min(350, box_height_pixels * base_scaling * 1.2))
        else:
            # Generic objects
            estimated_height = max(10, min(100, box_height_pixels * base_scaling * 0.4))
        
        return round(estimated_height, 0)
    
    def run_detection(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.update_status("Error: Could not open camera")
            print("Error: Could not open camera")
            return
            
        self.last_spoken = ""
        self.last_announcement_time = 0

        while self.detection_running and not self.stop_detection_flag:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break

            frame_height, frame_width = frame.shape[:2]
            frame_area = frame_height * frame_width
            
            # Depth estimation (if available)
            depth_map = None
            if self.depth_estimation_available and self.midas_model is not None:
                try:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_batch = self.transform(img_rgb).to("cpu")
                    if input_batch.ndim == 3:
                        input_batch = input_batch.unsqueeze(0)
                    
                    with torch.no_grad():
                        prediction = self.midas_model(input_batch)
                        depth_map = prediction.squeeze().cpu().numpy()
                        # Normalize depth map
                        depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-6)
                except Exception as e:
                    print(f"Depth estimation error: {e}")
                    depth_map = None

            # YOLO object detection
            try:
                results = self.yolo_model(frame, verbose=False)
                objects_in_frame = []

                # Check if any detections exist
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        conf = float(box.conf[0])
                        if conf < self.confidence_threshold:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        label = self.class_names[cls_id] if cls_id < len(self.class_names) else f'class_{cls_id}'

                        # Ensure valid bounding box
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # Calculate box area
                        box_area = (x2 - x1) * (y2 - y1)
                        
                        # Extract object crop for color analysis
                        obj_crop = frame[y1:y2, x1:x2]
                        if obj_crop.size == 0:
                            continue

                        # Get dominant color from center region
                        h, w = obj_crop.shape[:2]
                        color_name = "unknown"
                        if h > 10 and w > 10:  # Ensure crop is large enough
                            cx1, cx2 = max(0, int(w * 0.3)), min(w, int(w * 0.7))
                            cy1, cy2 = max(0, int(h * 0.3)), min(h, int(h * 0.7))

                            obj_crop_rgb = cv2.cvtColor(obj_crop, cv2.COLOR_BGR2RGB)
                            center_crop_rgb = obj_crop_rgb[cy1:cy2, cx1:cx2]
                            
                            if center_crop_rgb.size > 0:
                                # Get average color more robustly
                                avg_color = np.mean(center_crop_rgb.reshape(-1, 3), axis=0)
                                color_name = self.get_basic_color_name(tuple(avg_color.astype(int)))

                        # Distance calculation
                        if depth_map is not None:
                            # Use MiDaS depth estimation
                            center_x = max(0, min((x1 + x2) // 2, depth_map.shape[1] - 1))
                            center_y = max(0, min((y1 + y2) // 2, depth_map.shape[0] - 1))
                            
                            # Sample area around center for better depth estimation
                            depth_values = []
                            for dy in range(-2, 3):
                                for dx in range(-2, 3):
                                    ny = max(0, min(center_y + dy, depth_map.shape[0] - 1))
                                    nx = max(0, min(center_x + dx, depth_map.shape[1] - 1))
                                    depth_values.append(depth_map[ny, nx])
                            
                            depth_val = np.median(depth_values)
                            
                            # Improved distance scaling for feet
                            DEPTH_SCALE_FEET = 10  # Direct scaling to feet
                            approx_dist_ft = round(abs(depth_val * DEPTH_SCALE_FEET), 1)
                            approx_dist_ft = max(0.5, min(20.0, approx_dist_ft))  # Reasonable bounds
                            
                            dist_description = f"{approx_dist_ft} feet"
                        else:
                            # Use simple distance estimation
                            dist_description, approx_dist_ft = self.estimate_distance_simple(box_area, frame_area)
                            dist_description = f"{approx_dist_ft} feet"

                        # Height estimation with realistic scaling
                        box_height_pixels = y2 - y1
                        approx_height_cm = self.estimate_height_realistic(box_height_pixels, approx_dist_ft, label)

                        # Generate description - ALWAYS in feet
                        if label.lower() == "person":
                            desc = f"A person, about {int(approx_height_cm)} cm tall, is {approx_dist_ft} feet away."
                            overlay_text = f"Person: {int(approx_height_cm)}cm, {approx_dist_ft}ft"
                        else:
                            desc = f"A {label} of {color_name} color, about {int(approx_height_cm)} cm tall, is {approx_dist_ft} feet away."
                            overlay_text = f"{label}: {color_name}, {int(approx_height_cm)}cm, {approx_dist_ft}ft"

                        print(f"[{label}] {desc}")
                        objects_in_frame.append(desc)

                        # Draw GREEN bounding box for ALL objects
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Bright green, thick line
                        
                        # Add background to text for better readability
                        text_size = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (x1, y2 + 5), (x1 + text_size[0] + 10, y2 + 30), (0, 0, 0), -1)
                        cv2.putText(frame, overlay_text, (x1 + 5, y2 + 23),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Bright green text
                        
                        # Add confidence score in green
                        conf_text = f"Conf: {conf:.2f}"
                        cv2.putText(frame, conf_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Bright green text

                # Text-to-Speech Output - Announce objects ONE BY ONE
                current_time = time.time()
                if (objects_in_frame and self.tts_available and not self.tts_busy and 
                    (current_time - self.last_announcement_time) > self.announcement_interval):
                    
                    if len(objects_in_frame) > 0:
                        # Cycle through objects one by one
                        if self.current_object_index >= len(objects_in_frame):
                            self.current_object_index = 0
                        
                        # Announce current object
                        message = objects_in_frame[self.current_object_index]
                        
                        # Add object number for context when multiple objects
                        if len(objects_in_frame) > 1:
                            message = f"Object {self.current_object_index + 1} of {len(objects_in_frame)}: {message}"
                        
                        try:
                            # Run TTS in separate thread to avoid blocking
                            def speak_async():
                                self.tts_busy = True
                                try:
                                    if self.engine:
                                        self.engine.say(message)
                                        self.engine.runAndWait()
                                except Exception as e:
                                    print(f"TTS Error: {e}")
                                finally:
                                    self.tts_busy = False
                            
                            Thread(target=speak_async, daemon=True).start()
                            self.last_announcement_time = current_time
                            self.current_object_index += 1  # Move to next object
                            self.last_spoken = message
                            
                        except Exception as e:
                            print(f"TTS Thread Error: {e}")
                            self.tts_busy = False
                
                # Display comprehensive frame information
                info_y = 30
                if objects_in_frame:
                    # Show total object count
                    count_text = f"Objects detected: {len(objects_in_frame)}"
                    cv2.putText(frame, count_text, (10, info_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    info_y += 30
                    
                    # Show current object being announced
                    if len(objects_in_frame) > 1:
                        current_obj_text = f"Announcing: Object {(self.current_object_index % len(objects_in_frame)) + 1} of {len(objects_in_frame)}"
                        cv2.putText(frame, current_obj_text, (10, info_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                print(f"Detection error: {e}")
                continue

            # Show the frame
            cv2.imshow("Vision Assistant", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.detection_running = False
        self.stop_detection_flag = False
        self.update_status("Detection stopped.")
    
    def start_detection(self):
        """Start detection"""
        if not self.detection_running:
            self.detection_running = True
            self.stop_detection_flag = False
            self.current_object_index = 0  # Reset object index
            Thread(target=self.run_detection, daemon=True).start()
            self.update_status("Detection running...")
            print("Detection started")
    
    def stop_detection(self):
        """Stop detection"""
        if self.detection_running:
            self.stop_detection_flag = True
            self.detection_running = False
            self.update_status("Detection stopped.")
            print("Detection stopped")
    
    def update_confidence(self, val):
        """Update confidence threshold"""
        self.confidence_threshold = float(val) / 100
        self.conf_label.config(text=f"Accuracy: {self.confidence_threshold:.2f}")
    
    def update_status(self, message):
        """Update status label"""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=message)
    
    def voice_listener(self):
        """Voice command listener"""
        try:
            recognizer = sr.Recognizer()
            mic = sr.Microphone()

            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
            print("Voice commands ready: 'start detection' or 'stop detection'")
            
        except Exception as e:
            print(f"Microphone setup error: {e}")
            return

        while not self.stop_voice_thread:
            try:
                with mic as source:
                    audio = recognizer.listen(source, phrase_time_limit=3, timeout=1)
                command = recognizer.recognize_google(audio).lower()
                print(f"Voice Command: {command}")

                if "start detection" in command and not self.detection_running:
                    self.start_detection()
                elif "stop detection" in command and self.detection_running:
                    self.stop_detection()
                    
            except sr.WaitTimeoutError:
                pass  # Continue listening
            except sr.UnknownValueError:
                pass  # Couldn't understand audio
            except Exception as e:
                print(f"Voice listener error: {e}")
                time.sleep(1)  # Brief pause on error
    
    def start_voice_listener(self):
        """Start voice command listener in background thread"""
        self.stop_voice_thread = False
        Thread(target=self.voice_listener, daemon=True).start()
    
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Vision Assistant - G20")
        self.root.geometry("350x250")
        self.root.resizable(False, False)
        
        # Configure style
        try:
            style = ttk.Style()
            style.theme_use('clam')
        except:
            pass  # Use default theme if clam is not available
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="Vision Assistant", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Start Detection Button
        self.start_btn = ttk.Button(main_frame, text="Start Detection", 
                                   command=self.start_detection)
        self.start_btn.pack(pady=(0, 10), fill='x')
        
        # Stop Detection Button  
        self.stop_btn = ttk.Button(main_frame, text="Stop Detection",
                                  command=self.stop_detection)
        self.stop_btn.pack(pady=(0, 20), fill='x')
        
        # Status Label
        self.status_label = ttk.Label(main_frame, text="Ready to start detection",
                                     font=('Arial', 10))
        self.status_label.pack(pady=(0, 20))
        
        # Accuracy Slider Frame
        slider_frame = ttk.Frame(main_frame)
        slider_frame.pack(fill='x', pady=(0, 10))
        
        # Accuracy Label
        ttk.Label(slider_frame, text="Detection Accuracy:", 
                 font=('Arial', 9)).pack(anchor='w')
        
        self.conf_slider = ttk.Scale(slider_frame, from_=1, to=100, 
                                    orient=tk.HORIZONTAL, 
                                    command=self.update_confidence)
        self.conf_slider.set(int(self.confidence_threshold * 100))
        self.conf_slider.pack(fill='x', pady=(5, 0))
        
        # Current Accuracy Display
        self.conf_label = ttk.Label(main_frame, 
                                   text=f"Current Accuracy: {self.confidence_threshold:.2f}",
                                   font=('Arial', 9))
        self.conf_label.pack(pady=(5, 0))
        
        # Instructions
        instructions = ttk.Label(main_frame, 
                               text="Use buttons or voice commands:\n'start detection' or 'stop detection'\nPress 'q' in camera window to quit",
                               font=('Arial', 8), foreground='gray',
                               justify='center')
        instructions.pack(pady=(20, 0))
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_voice_thread = True
        self.stop_detection()
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
        self.root.destroy()
    
    def run(self):
        """Start the application"""
        print("Vision Assistant Starting...")
        print("Group G20 - Interactive Assistive System for Visually Impaired")
        print("="*60)
        self.root.mainloop()

# Entry Point
if __name__ == "__main__":
    try:
        app = VisionAssistant()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")  # Keep console open to see errors
