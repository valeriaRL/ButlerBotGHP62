# face_recognition_module.py
import os
import time
import face_recognition
import cv2
import numpy as np
from pycoral.adapters.common import input_size, set_input
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

class FaceRecognizer:
    def __init__(self):
        # Initialize paths and settings
        self.default_model_dir = '/Users/tarawang/Downloads/examples-camera/all_models'
        self.default_model = 'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
        self.default_labels = '/Users/tarawang/Downloads/butlerbot/face_labels1.txt'
        
        # Output directory
        self.output_base_dir = "/Users/tarawang/Downloads/butlerbot"
        
        # Load known faces
        self.known_faces = self.load_known_faces()
        
        # Initialize the interpreter
        self.interpreter = None
        self.labels = None
        self.size = None
        self.initialize_detector()

    def load_known_faces(self):
        print("Loading reference faces...")
        known_faces = {}
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Define the reference images with their correct extensions
            reference_faces = {
                "Goli": "goli.png",
                "Craig": "craig.jpeg", 
                "Alex": "alex.jpg"
            }
            
            # Load each face encoding
            for name, filename in reference_faces.items():
                image_path = os.path.join(script_dir, filename)
                if os.path.exists(image_path):
                    known_faces[name] = self.get_first_face_encoding(image_path, name)
                else:
                    print(f"Warning: Reference image not found at {image_path}")
            
            # Create output directories if they don't exist
            for name in known_faces.keys():
                person_dir = os.path.join(self.output_base_dir, name.lower())
                os.makedirs(person_dir, exist_ok=True)
                
        except Exception as e:
            print(f"Error loading known faces: {str(e)}")
            
        return known_faces

    def get_first_face_encoding(self, image_file, name="unknown"):
        image = face_recognition.load_image_file(image_file)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            raise ValueError(f"No face found in {image_file} for {name}")
        return encodings[0]

    def initialize_detector(self):
        try:
            model_path = os.path.join(self.default_model_dir, self.default_model)
            labels_path = os.path.join(self.default_model_dir, self.default_labels)
            
            print(f"Loading model: {model_path}\nLoading labels: {labels_path}")
            self.interpreter = make_interpreter(model_path)
            self.interpreter.allocate_tensors()
            self.labels = read_label_file(labels_path)
            self.size = input_size(self.interpreter)
        except Exception as e:
            print(f"Error initializing face detector: {str(e)}")
            raise

    def process_frame(self, frame):
        if self.interpreter is None:
            print("Face detector not initialized")
            return frame
            
        # Create a copy of the frame to work with
        img = cv2.resize(frame, self.size)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            # Perform face detection
            set_input(self.interpreter, rgb)
            self.interpreter.invoke()
            objs = get_objects(self.interpreter, 0.1)[:3]  # threshold=0.1, top_k=3
            
            # Process each detected face
            for obj in objs:
                bbox = obj.bbox
                x0, y0 = int(bbox.xmin), int(bbox.ymin)
                x1, y1 = int(bbox.xmax), int(bbox.ymax)

                scale_x = frame.shape[1] / self.size[0]
                scale_y = frame.shape[0] / self.size[1]
                x0, x1 = int(x0 * scale_x), int(x1 * scale_x)
                y0, y1 = int(y0 * scale_y), int(y1 * scale_y)

                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(frame.shape[1], x1), min(frame.shape[0], y1)

                face_crop = frame[y0:y1, x0:x1]
                label_text = "Face"

                if face_crop.size != 0:
                    rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_crop)
                    
                    if len(encodings) > 0:
                        face_encoding = encodings[0]
                        matches = face_recognition.compare_faces(list(self.known_faces.values()), face_encoding)
                        distances = face_recognition.face_distance(list(self.known_faces.values()), face_encoding)

                        if any(matches):
                            best_match_index = np.argmin(distances)
                            name = list(self.known_faces.keys())[best_match_index]
                            distance = distances[best_match_index]
                            if distance < 0.6:
                                label_text = f'{name} ({distance:.2f})'
                                person_dir = os.path.join(self.output_base_dir, name.lower())
                                timestamp = time.strftime("%Y%m%d-%H%M%S")
                                save_path = os.path.join(person_dir, f'{name}_detected_{timestamp}.jpg')
                                cv2.imwrite(save_path, face_crop)
                            else:
                                label_text = f'Unknown ({distance:.2f})'
                        else:
                            label_text = "Unknown"
                    else:
                        label_text = "No encoding"

                # Draw rectangle and label
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x0, y0 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            
        return frame