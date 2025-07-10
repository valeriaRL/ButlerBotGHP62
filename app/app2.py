import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QStackedWidget, QFrame, QSizePolicy, QStyle, QButtonGroup)
from PyQt5.QtCore import Qt, QTimer, QSize, QRectF, QRect, QPoint
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtGui import QPainter, QPainterPath, QBrush, QColor
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque
import pyttsx3
import requests

from face_recognition_module import FaceRecognizer

class StartScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome")
        self.setGeometry(100, 100, 1000, 700)
        
        self.central_widget = QLabel()
        self.central_widget.setAlignment(Qt.AlignCenter)
        
        pixmap = QPixmap("/Users/tarawang/Downloads/butlerbot/app/start_image.png")
        if pixmap.isNull():
            self.central_widget.setText("Welcome to ButlerBot\n(Press Space to start)")
            self.central_widget.setStyleSheet("font-size: 24px;")
        else:
            self.central_widget.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        self.setCentralWidget(self.central_widget)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.close()
            self.launch_main_app()
            
    def launch_main_app(self):
        self.main_window = ButlerBotApp()
        self.main_window.show()

class RoundedCameraLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.radius = 0  
        self._pixmap = None  
        self.setMinimumSize(1, 1)
        
    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()  # Trigger a repaint
        
    def pixmap(self):
        return self._pixmap
        
    def paintEvent(self, event):
        path = QPainterPath()
        rect = self.rect()
        path.addRoundedRect(QRectF(rect), self.radius, self.radius)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setClipPath(path)
        
        painter.fillRect(rect, QBrush(QColor(0, 0, 0)))
        
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatioByExpanding, 
                Qt.SmoothTransformation
            )
            x = (scaled.width() - self.width()) / 2
            y = (scaled.height() - self.height()) / 2
            painter.drawPixmap(QPoint(int(-x), int(-y)), scaled)
        
        painter.end()

class ButlerBotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ButlerBot")
        self.setGeometry(100, 100, 1000, 700)
        
        self.cap = None
        self.face_recognizer = FaceRecognizer()
        self.face_recognition_enabled = False

        self.beverage_detection_enabled = False
        self.yolo_model = YOLO("/Users/tarawang/Downloads/butlerbot/drinkstuff/best.pt")
        self.class_names = self.yolo_model.model.names
        self.frame_queue = deque(maxlen=10)
        self.last_detected_drink = None
        self.last_print_time = 0  # Initialize this variable

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Initialize UI
        self.init_sidebar()
        self.init_main_area()
        
        # Show camera by default
        self.show_camera()
        
    def init_sidebar(self):
        # Sidebar frame
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(200)
        self.sidebar.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border: none;
            }
            QPushButton {
                background-color: transparent;
                color: white;
                font-size: 14px;
                text-align: left;
                padding: 12px 20px 12px 40px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
            QPushButton:pressed {
                background-color: #2980b9;
            }
            QPushButton::icon {
                filter: invert(100%);
            }
        """)
        
        # Sidebar layout
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 20, 0, 20)
        sidebar_layout.setSpacing(10)
        
        # Title
        title = QLabel("ButlerBot")
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
                padding: 20px;
            }
        """)
        sidebar_layout.addWidget(title)

        # Get the absolute path to the directory containing this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Menu buttons with icons
        self.menu_buttons = {}
        icon_size = QSize(32, 32)  # Size of icons in pixels
        
        # Camera button
        camera_btn = QPushButton("Camera")
        camera_icon_path = os.path.join(script_dir, "camera1.png")
        if os.path.exists(camera_icon_path):
            camera_btn.setIcon(QIcon(camera_icon_path))
        else:
            camera_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaRecord))
        camera_btn.setIconSize(icon_size)
        camera_btn.setCursor(Qt.PointingHandCursor)
        camera_btn.clicked.connect(self.show_camera)
        sidebar_layout.addWidget(camera_btn)
        self.menu_buttons["Camera"] = camera_btn
        
        # Sensor Data button
        sensor_btn = QPushButton("Sensor Data")
        sensor_icon_path = os.path.join(script_dir, "sensor1.png")
        if os.path.exists(sensor_icon_path):
            sensor_btn.setIcon(QIcon(sensor_icon_path))
        else:
            sensor_btn.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        sensor_btn.setIconSize(icon_size)
        sensor_btn.setCursor(Qt.PointingHandCursor)
        sensor_btn.clicked.connect(self.show_sensor_data)
        sidebar_layout.addWidget(sensor_btn)
        self.menu_buttons["Sensor Data"] = sensor_btn
        
        # Music button
        music_btn = QPushButton("Music")
        music_icon_path = os.path.join(script_dir, "music1.png")
        if os.path.exists(music_icon_path):
            music_btn.setIcon(QIcon(music_icon_path))
        else:
            music_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        music_btn.setIconSize(icon_size)
        music_btn.setCursor(Qt.PointingHandCursor)
        music_btn.clicked.connect(self.show_music)
        sidebar_layout.addWidget(music_btn)
        self.menu_buttons["Music"] = music_btn

        # Nutrition button
        nutrition_btn = QPushButton("Nutrition Information")
        nutrition_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogHelpButton))
        nutrition_btn.setIconSize(icon_size)
        nutrition_btn.setCursor(Qt.PointingHandCursor)
        nutrition_btn.clicked.connect(self.show_nutrition)
        sidebar_layout.addWidget(nutrition_btn)
        self.menu_buttons["Nutrition"] = nutrition_btn

        # Spacer
        sidebar_layout.addStretch()
        
        # Settings button
        settings_btn = QPushButton("Settings")
        settings_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        settings_btn.setIconSize(icon_size)
        settings_btn.setCursor(Qt.PointingHandCursor)
        settings_btn.clicked.connect(self.show_settings)
        sidebar_layout.addWidget(settings_btn)
        
        # Add sidebar to main layout
        self.main_layout.addWidget(self.sidebar)
        
    def init_main_area(self):
        # Main content area
        self.main_content = QFrame()
        self.main_content.setStyleSheet("""
            QFrame {
                background-color: #ecf0f1;
                border: none;
            }
        """)
        
        # Stacked widget for different views
        self.stacked_widget = QStackedWidget()
        
        # Camera view
        self.camera_view = QWidget()
        self.init_camera_view()
        self.stacked_widget.addWidget(self.camera_view)
        
        # Sensor view
        self.sensor_view = QWidget()
        self.init_sensor_view()
        self.stacked_widget.addWidget(self.sensor_view)
        
        # Music view
        self.music_view = QWidget()
        self.init_music_view()
        self.stacked_widget.addWidget(self.music_view)

        # Nutrition view
        self.nutrition_view = QWidget()
        self.init_nutrition_view()
        self.stacked_widget.addWidget(self.nutrition_view)
        
        # Settings view
        self.settings_view = QWidget()
        self.init_settings_view()
        self.stacked_widget.addWidget(self.settings_view)
        
        # Main content layout
        main_content_layout = QVBoxLayout(self.main_content)
        main_content_layout.setContentsMargins(0, 0, 0, 0)
        main_content_layout.addWidget(self.stacked_widget)
        
        # Shutdown button
        shutdown_btn = QPushButton("Shutdown")
        shutdown_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        shutdown_btn.setCursor(Qt.PointingHandCursor)
        shutdown_btn.setFixedWidth(100)
        shutdown_btn.clicked.connect(self.close)
        
        # Position shutdown button in bottom right
        shutdown_container = QWidget()
        shutdown_layout = QHBoxLayout(shutdown_container)
        shutdown_layout.addStretch()
        shutdown_layout.addWidget(shutdown_btn)
        shutdown_layout.setContentsMargins(0, 0, 20, 20)
        main_content_layout.addWidget(shutdown_container)
        
        # Add main content to main layout
        self.main_layout.addWidget(self.main_content)
        
    def init_camera_view(self):
        layout = QVBoxLayout(self.camera_view)
        layout.setAlignment(Qt.AlignCenter)
        
        # Create mode toggle buttons
        mode_selector = QWidget()
        mode_layout = QHBoxLayout(mode_selector)
        mode_layout.setContentsMargins(0, 0, 0, 20)
        mode_layout.setSpacing(10)
        
        # Button group for exclusive selection
        self.mode_group = QButtonGroup(self)
        
        # Regular camera button
        self.camera_mode_btn = QPushButton("Camera")
        self.camera_mode_btn.setCheckable(True)
        self.camera_mode_btn.setChecked(True)
        self.camera_mode_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #2980b9;
            }
        """)
        self.camera_mode_btn.clicked.connect(self.toggle_camera_mode)
        self.mode_group.addButton(self.camera_mode_btn)
        
        # Facial recognition button
        self.face_mode_btn = QPushButton("Facial Recognition")
        self.face_mode_btn.setCheckable(True)
        self.face_mode_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #2980b9;
            }
        """)
        self.face_mode_btn.clicked.connect(self.toggle_camera_mode)
        self.mode_group.addButton(self.face_mode_btn)

        # Beverage detection button
        self.beverage_mode_btn = QPushButton("Beverages")
        self.beverage_mode_btn.setCheckable(True)
        self.beverage_mode_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #2980b9;
            }
        """)
        self.beverage_mode_btn.clicked.connect(self.toggle_camera_mode)
        self.mode_group.addButton(self.beverage_mode_btn)

        # Add buttons to layout
        mode_layout.addStretch()
        mode_layout.addWidget(self.camera_mode_btn)
        mode_layout.addWidget(self.face_mode_btn)
        mode_layout.addWidget(self.beverage_mode_btn)
        mode_layout.addStretch()
        
        layout.addWidget(mode_selector)
        
        # Create container for the camera with shadow
        camera_container = QWidget()
        camera_container.setFixedSize(800, 600)  # Slightly larger than camera for shadow
        
        # Create rounded camera label
        self.camera_label = RoundedCameraLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFixedSize(800, 600)
        
        # Add to container
        container_layout = QVBoxLayout(camera_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)
        
        layout.addWidget(camera_container, alignment=Qt.AlignCenter)
        
        # Buttons container
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.addStretch()

        # Capture button
        capture_btn = QPushButton("Capture Image")
        capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #2c3e50;
                color: white;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        capture_btn.setCursor(Qt.PointingHandCursor)
        capture_btn.clicked.connect(self.capture_image)
        btn_layout.addWidget(capture_btn)

        # Get Nutrition Info button
        nutrition_btn = QPushButton("Get Nutrition Info")
        nutrition_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1e8449;
            }
        """)
        nutrition_btn.setCursor(Qt.PointingHandCursor)
        nutrition_btn.clicked.connect(self.get_nutrition_info_for_last_detection)
        btn_layout.addWidget(nutrition_btn)

        btn_layout.addStretch()
        layout.addWidget(btn_container)
        
        # Start camera
        self.start_camera()
        
    def toggle_camera_mode(self):
        """Toggle between regular camera and face recognition modes"""
        self.face_recognition_enabled = self.face_mode_btn.isChecked()
        self.beverage_detection_enabled = self.beverage_mode_btn.isChecked()
        
    def init_sensor_view(self):
        layout = QVBoxLayout(self.sensor_view)
        layout.setAlignment(Qt.AlignCenter)
        
        title = QLabel("Sensor Data")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)
        layout.addWidget(title, alignment=Qt.AlignCenter)
        
        # Add sensor data widgets here
        temp_label = QLabel("Temperature: 24.5°C")
        temp_label.setStyleSheet("font-size: 18px; color: #34495e;")
        layout.addWidget(temp_label, alignment=Qt.AlignCenter)
        
        humidity_label = QLabel("Humidity: 45%")
        humidity_label.setStyleSheet("font-size: 18px; color: #34495e;")
        layout.addWidget(humidity_label, alignment=Qt.AlignCenter)
        
        layout.addStretch()
        
    def init_music_view(self):
        layout = QVBoxLayout(self.music_view)
        layout.setAlignment(Qt.AlignCenter)
        
        title = QLabel("Music Player")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)
        layout.addWidget(title, alignment=Qt.AlignCenter)
        
        # Add music player widgets here
        song_label = QLabel("Now Playing: Nothing")
        song_label.setStyleSheet("font-size: 18px; color: #34495e;")
        layout.addWidget(song_label, alignment=Qt.AlignCenter)
        
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        
        prev_btn = QPushButton("⏮")
        play_btn = QPushButton("⏵")
        next_btn = QPushButton("⏭")
        
        for btn in [prev_btn, play_btn, next_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 24px;
                    background-color: #3498db;
                    color: white;
                    border-radius: 30px;
                    min-width: 60px;
                    min-height: 60px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
            """)
            btn.setCursor(Qt.PointingHandCursor)
            controls_layout.addWidget(btn)
        
        layout.addWidget(controls, alignment=Qt.AlignCenter)
        layout.addStretch()
    
    def init_nutrition_view(self):
        layout = QVBoxLayout(self.nutrition_view)
        layout.setAlignment(Qt.AlignTop)
    
        title = QLabel("Nutrition Information")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)
        layout.addWidget(title)
    
        # A text area to display nutrition data
        self.nutrition_text = QLabel("")
        self.nutrition_text.setWordWrap(True)
        self.nutrition_text.setStyleSheet("font-size: 16px; color: #34495e;")
        layout.addWidget(self.nutrition_text)
    
        layout.addStretch()
        
    def show_nutrition(self):
        self.stacked_widget.setCurrentWidget(self.nutrition_view)
        
    def get_nutrition_info_for_last_detection(self):
        if self.last_detected_drink:
            self.fetch_and_print_nutrition(self.last_detected_drink)
        else:
            self.nutrition_text.setText("No drink detected recently.")
            self.show_nutrition()

    def init_settings_view(self):
        layout = QVBoxLayout(self.settings_view)
        layout.setAlignment(Qt.AlignCenter)
        
        title = QLabel("Settings")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)
        layout.addWidget(title, alignment=Qt.AlignCenter)
        
        # Add settings widgets here
        setting1 = QLabel("Camera Settings")
        setting1.setStyleSheet("font-size: 18px; color: #34495e;")
        layout.addWidget(setting1, alignment=Qt.AlignCenter)
        
        setting2 = QLabel("Audio Settings")
        setting2.setStyleSheet("font-size: 18px; color: #34495e;")
        layout.addWidget(setting2, alignment=Qt.AlignCenter)
        
        layout.addStretch()
        
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set up timer for camera updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(20)  # Update every 20ms
        
    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            # Process frame based on selected mode
            if self.face_recognition_enabled:
                try:
                    if hasattr(self, 'face_recognizer'):
                        frame = self.face_recognizer.process_frame(frame)
                    else:
                        print("Face recognizer not initialized")
                except Exception as e:
                    print(f"Face recognition error: {str(e)}")
                    # Fall back to regular camera view
                    self.face_mode_btn.setChecked(False)
                    self.face_recognition_enabled = False
                    self.show_notification("Face recognition failed - switched to regular view")
                    
            elif self.beverage_detection_enabled:
                results = self.yolo_model(frame, verbose=False)[0]
                frame_width = frame.shape[1]
                detections = []

                label_mapping = {
                    "mtndew": "Mountain Dew",
                    "nitrobrew": "Nitro Brew",
                    "drpepper": "Dr Pepper"
                }

                for r in results.boxes.data:
                    x1, y1, x2, y2, conf, cls = r
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    label = self.class_names[int(cls)]
                    label = label_mapping.get(label.lower(), label)
                    side = "left" if (x1 + x2) / 2 < frame_width / 2 else "right"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label} ({side}) {conf:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    detections.append((label, side, conf, x1, y1))

                self.frame_queue.append(detections)
            
                # Voting logic
                vote_counter = {}
                for det_list in self.frame_queue:
                    for label, side, *_ in det_list:
                        key = (label, side)
                        vote_counter[key] = vote_counter.get(key, 0) + 1

                if vote_counter:
                    best = max(vote_counter.items(), key=lambda x: x[1])
                    label, side = best[0]
                    current_time = time.time()
                    if current_time - self.last_print_time > 15:
                        print(f"[TRACKING] {label.upper()} on the {side.upper()} ({best[1]}/10 frames)")
                        self.last_print_time = current_time

                    if best[1] >= 7:
                        self.last_detected_drink = label  # Store detection
                        

            # Convert the image to RGB format
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            # Convert to QImage
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
            # Convert to QPixmap and display
            pixmap = QPixmap.fromImage(qt_image)
            self.camera_label.setPixmap(pixmap)
            
    def capture_image(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"captured_image_{timestamp}.png"
                cv2.imwrite(filename, frame)
                self.show_notification(f"Image captured and saved as {filename}")
    
    def show_notification(self, message):
        # Create a temporary notification label
        notification = QLabel(message)
        notification.setStyleSheet("""
            QLabel {
                background-color: rgba(46, 204, 113, 0.9);
                color: white;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        notification.setAlignment(Qt.AlignCenter)
        
        # Add to main layout and show
        self.main_layout.addWidget(notification, alignment=Qt.AlignTop)
        
        # Remove after 3 seconds
        QTimer.singleShot(3000, lambda: notification.deleteLater())
    
    def show_camera(self):
        self.stacked_widget.setCurrentWidget(self.camera_view)
        
    def show_sensor_data(self):
        self.stacked_widget.setCurrentWidget(self.sensor_view)
        
    def show_music(self):
        self.stacked_widget.setCurrentWidget(self.music_view)
        
    def show_settings(self):
        self.stacked_widget.setCurrentWidget(self.settings_view)
        
    def fetch_and_print_nutrition(self, food_input):
        APP_ID = "d1b56ee3"
        API_KEY = "1952289073a5f8e6f532f465abf99951"
        HEADERS = {
            "x-app-id": APP_ID,
            "x-app-key": API_KEY
        }
        SEARCH_URL = "https://trackapi.nutritionix.com/v2/search/instant"
        ITEM_URL = "https://trackapi.nutritionix.com/v2/search/item"
        
        try:
            search_params = {"query": food_input, "branded": True}
            search_response = requests.get(SEARCH_URL, headers=HEADERS, params=search_params)
            search_response.raise_for_status()
            search_result = search_response.json()

            if not search_result['branded']:
                print(f"No branded item found under the name '{food_input}'. Try being more specific.")
                self.nutrition_text.setText(f"No branded item found under the name '{food_input}'.")
                self.show_nutrition()
                return

            branded_item = search_result['branded'][0]
            item_id = branded_item['nix_item_id']
            item_params = {"nix_item_id": item_id}
            item_response = requests.get(ITEM_URL, headers=HEADERS, params=item_params)
            item_response.raise_for_status()
            item = item_response.json()['foods'][0]

            # Build a formatted text string
            nutrition_info = (
                f"<b>Nutrition Info for:</b> {item['food_name']}<br>"
                f"<b>Brand:</b> {item.get('brand_name', 'N/A')}<br>"
                f"<b>Serving:</b> {item['serving_qty']} {item['serving_unit']} "
                f"({item.get('serving_weight_grams', 'N/A')} g)<br><br>"
                f"<b>Calories:</b> {item.get('nf_calories', 'N/A')} kcal<br>"
                f"<b>Total Fat:</b> {item.get('nf_total_fat', 'N/A')} g<br>"
                f"<b>Sugar:</b> {item.get('nf_sugars', 'N/A')} g<br>"
                f"<b>Cholesterol:</b> {item.get('nf_cholesterol', 'N/A')} mg<br>"
                f"<b>Sodium:</b> {item.get('nf_sodium', 'N/A')} mg<br>"
                f"<b>Total Carbs:</b> {item.get('nf_total_carbohydrate', 'N/A')} g<br>"
                f"<b>Dietary Fiber:</b> {item.get('nf_dietary_fiber', 'N/A')} g<br>"
                f"<b>Protein:</b> {item.get('nf_protein', 'N/A')} g<br>"
                f"<b>Potassium:</b> {item.get('nf_potassium', 'N/A')} mg<br>"
                f"<b>Phosphorus:</b> {item.get('nf_p', 'N/A')} mg<br>"
            )
            
            # Add ingredients if available
            if item.get('nf_ingredient_statement'):
                nutrition_info += f"<br><b>Ingredients:</b><br>{item['nf_ingredient_statement']}"
            
            # Set the nutrition text and show the nutrition view
            self.nutrition_text.setText(nutrition_info)
            self.show_nutrition()
            
            # Also speak the nutrition info
            self.speak_nutrition_info(item)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching nutrition data: {str(e)}"
            print(error_msg)
            self.nutrition_text.setText(error_msg)
            self.show_nutrition()
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(error_msg)
            self.nutrition_text.setText(error_msg)
            self.show_nutrition()

    def speak_nutrition_info(self, item):
        """Use text-to-speech to announce nutrition information"""
        try:
            engine = pyttsx3.init()
            
            # Create a spoken version of the nutrition info
            speech_text = f"Nutrition information for {item['food_name']}. "
            speech_text += f"Calories: {item.get('nf_calories', 'unknown')}. "
            speech_text += f"Total fat: {item.get('nf_total_fat', 'unknown')} grams. "
            speech_text += f"Sugar: {item.get('nf_sugars', 'unknown')} grams. "
            speech_text += f"Protein: {item.get('nf_protein', 'unknown')} grams."
            
            engine.say(speech_text)
            engine.runAndWait()
            
        except Exception as e:
            print(f"Text-to-speech error: {str(e)}")

    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if hasattr(self, 'timer'):
            self.timer.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Show start screen first
    start_screen = StartScreen()
    start_screen.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()