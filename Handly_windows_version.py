import os
import sys
import random
import tempfile
import shutil
from datetime import datetime
import cv2
import numpy as np
import json
import time
import io
import base64
import threading
import pygame  # For playing audio

from io import BytesIO

from PyQt5.QtMultimedia import QSound
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QFrame, QTextEdit, QProgressBar, QSplitter,
                             QMessageBox, QComboBox, QSlider, QGroupBox,
                             QSizePolicy, QStackedWidget, QScrollArea, QLineEdit, QGridLayout, QTabWidget,
                             QTableWidget, QTableWidgetItem, QDialog, QBoxLayout, QListWidgetItem, QListWidget)
from PyQt5.QtCore import Qt, QUrl, QTimer, QSize, pyqtSignal, QPointF, QRectF, QAnimationGroup, QPropertyAnimation, \
    QParallelAnimationGroup, QEasingCurve
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon, QPalette, QColor, QPolygonF, QPainter, QPen

from googleapiclient.http import MediaIoBaseDownload
from pose_format import Pose
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from gtts import gTTS  # Google Text-to-Speech


# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = 'client_secret.json'
TOKEN_FILE = 'token.json'


class LoginWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Handly - Sign In")
        self.setFixedSize(500, 400)
        self.setup_ui()
        self.user_data_file = "users.json"
        self.current_user = None

    # Add this at the beginning of your main window initialization
    def setup_theme(self):
        # Set a modern color palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))  # Light gray background
        palette.setColor(QPalette.WindowText, QColor(53, 53, 53))  # Dark gray text
        palette.setColor(QPalette.Base, QColor(255, 255, 255))  # White for input backgrounds
        palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(53, 53, 53))
        palette.setColor(QPalette.Text, QColor(53, 53, 53))
        palette.setColor(QPalette.Button, QColor(74, 110, 251))  # Main blue color
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))  # White button text
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Highlight, QColor(74, 110, 251))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)

        # Set application-wide stylesheet
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }

            QPushButton {
                background-color: #4a6baf;
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                min-width: 80px;
            }

            QPushButton:hover {
                background-color: #3a5a9f;
            }

            QPushButton:pressed {
                background-color: #2a4a8f;
            }

            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 8px;
                margin-top: 20px;
                padding-top: 15px;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }

            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
                height: 20px;
            }

            QProgressBar::chunk {
                background-color: #4a6baf;
                border-radius: 4px;
            }

            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
            }

            QTabBar::tab {
                background: #f0f0f0;
                border: 1px solid #ddd;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 16px;
                margin-right: 2px;
            }

            QTabBar::tab:selected {
                background: white;
                border-bottom: 2px solid #4a6baf;
            }

            QLabel#titleLabel {
                font-size: 24px;
                font-weight: bold;
                color: #4a6baf;
            }
        """)
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # App logo
        logo_label = QLabel()
        logo_pixmap = QPixmap(":/images/logo.png").scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        # Title
        title = QLabel("Welcome to Handly")
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #4a6baf;
            qproperty-alignment: AlignCenter;
        """)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Your sign language learning companion")
        subtitle.setStyleSheet("""
            font-size: 16px;
            color: #666;
            qproperty-alignment: AlignCenter;
        """)
        layout.addWidget(subtitle)

        # Form container
        form_container = QWidget()
        form_layout = QVBoxLayout(form_container)
        form_layout.setContentsMargins(15, 15, 15, 15)
        form_layout.setSpacing(10)
        form_layout.setAlignment(Qt.AlignCenter)

        # Username input
        username_label = QLabel("Username:")
        username_label.setStyleSheet("font-size: 14px;")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter your username")
        self.username_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }
        """)
        form_layout.addWidget(username_label)
        form_layout.addWidget(self.username_input)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)

        self.sign_in_btn = QPushButton("Sign In")
        self.sign_in_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a6baf;
                color: white;
                padding: 10px 20px;
                font-size: 16px;
                min-width: 120px;
            }
        """)
        self.sign_in_btn.clicked.connect(self.sign_in)
        btn_layout.addWidget(self.sign_in_btn)

        self.new_user_btn = QPushButton("New User")
        self.new_user_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                padding: 10px 20px;
                font-size: 16px;
                min-width: 120px;
            }
        """)
        self.new_user_btn.clicked.connect(self.create_new_user)
        btn_layout.addWidget(self.new_user_btn)

        form_layout.addLayout(btn_layout)

        # Status label
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #dc3545; font-size: 14px; qproperty-alignment: AlignCenter;")
        form_layout.addWidget(self.status_label)

        layout.addWidget(form_container)
        layout.addStretch()

    def load_user_data(self):
        try:
            if os.path.exists(self.user_data_file):
                with open(self.user_data_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading user data: {e}")
            return {}

    def save_user_data(self, data):
        try:
            with open(self.user_data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving user data: {e}")

    def sign_in(self):
        username = self.username_input.text().strip()
        if not username:
            self.status_label.setText("Please enter a username")
            return

        user_data = self.load_user_data()
        if username in user_data:
            self.current_user = user_data[username]

            # Initialize word progress if not exists
            if 'word_progress' not in self.current_user:
                self.current_user['word_progress'] = {}

            # Initialize stats if not exists
            if 'stats' not in self.current_user:
                self.current_user['stats'] = {
                    'total_practice_sessions': 0,
                    'total_words_attempted': 0,
                    'total_words_correct': 0,
                    'streak_days': 0,
                    'last_practice_date': None
                }

            self.accept()
        else:
            self.status_label.setText("User not found")

    def create_new_user(self):
        username = self.username_input.text().strip()
        if not username:
            self.status_label.setText("Please enter a username")
            return

        user_data = self.load_user_data()
        if username in user_data:
            self.status_label.setText("Username already exists")
            return

        new_user = {
            "username": username,
            "stats": {
                "total_practice_sessions": 0,
                "total_words_attempted": 0,
                "total_words_correct": 0,
                "streak_days": 0,
                "last_practice_date": None,
                "learning_points": 0,  # Points from practice
                "game_points": {       # Points from games
                    "quiz": 0,
                    "clock": 0
                }
            },
            "word_progress": {},  # Track progress per word
            "known_words": [],    # Words signed perfectly
            "achievements": [],
            "preferences": {
                "voice_feedback": True,
                "auto_advance": True,
                "difficulty": "normal"
            }
        }

        user_data[username] = new_user
        self.save_user_data(user_data)
        self.current_user = new_user
        self.accept()
        return new_user


class PracticeWidget(QWidget):
    def __init__(self, correct_video_path, parent=None):
        super().__init__(parent)
        # Store the parent reference
        self.parent_app = parent

        # Verify the video path exists
        if not os.path.exists(correct_video_path):
            QMessageBox.critical(self, "Error",
                                 f"Could not find video: {correct_video_path}")
            self.close()
            return

        self.correct_video_path = correct_video_path
        self.correct_frames = []
        self.current_frame_idx = 0
        self.cap = None
        self.timer = QTimer()
        # Add camera initialization
        self.cap = cv2.VideoCapture(0)  # Initialize camera
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", "Could not access camera")
            self.close()
            return

        # Start update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30fps
        # Initialize MediaPipe with proper error handling
        try:
            self.holistic = mp.solutions.holistic.Holistic(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Could not initialize pose detection: {str(e)}")
            self.close()
            return

        # Initialize other components
        self.auto_advance = True
        self.match_threshold = 80
        self.match_duration = 0.4
        self.good_match_start_time = None
        self.current_match_score = 0

        # Initialize audio system
        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"Could not initialize audio: {str(e)}")
            self.voice_feedback_enabled = False
        else:
            self.voice_feedback_enabled = True

        self.last_feedback_time = 0
        self.feedback_cooldown = 3
        self.last_spoken_feedback = ""

        # Setup UI and load frames
        self.setup_ui()
        self.load_correct_frames()

    def setup_ui(self):
        self.setWindowTitle("Practice Mode - Frame by Frame")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.resize(1200, 800)  # Increased window size

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # Title
        title = QLabel("✋ Sign Language Practice Mode")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #4a6baf;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        # Split view with fixed sizes
        splitter = QSplitter(Qt.Horizontal)

        # Camera view - set fixed size
        self.camera_group = QGroupBox("Your Camera")
        camera_layout = QVBoxLayout()
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFixedSize(640, 480)  # Standard 4:3 aspect ratio
        self.camera_label.setStyleSheet("background-color: black;")
        camera_layout.addWidget(self.camera_label)
        self.camera_group.setLayout(camera_layout)

        # Correct pose view - set fixed size
        self.correct_group = QGroupBox("Correct Pose")
        correct_layout = QVBoxLayout()
        self.correct_label = QLabel()
        self.correct_label.setAlignment(Qt.AlignCenter)
        self.correct_label.setFixedSize(640, 480)  # Same size as camera view
        self.correct_label.setStyleSheet("background-color: black;")
        correct_layout.addWidget(self.correct_label)
        self.correct_group.setLayout(correct_layout)

        splitter.addWidget(self.camera_group)
        splitter.addWidget(self.correct_group)
        layout.addWidget(splitter)

        # Rest of the UI remains the same...
        # Feedback area
        self.feedback_group = QGroupBox("Feedback")
        feedback_layout = QVBoxLayout()

        self.match_meter = QProgressBar()
        self.match_meter.setRange(0, 100)
        feedback_layout.addWidget(self.match_meter)

        self.feedback_text = QLabel()
        feedback_layout.addWidget(self.feedback_text)

        self.detailed_feedback = QLabel()
        feedback_layout.addWidget(self.detailed_feedback)

        # Countdown indicator for auto-advance
        self.countdown_label = QLabel()
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        feedback_layout.addWidget(self.countdown_label)

        self.feedback_group.setLayout(feedback_layout)
        layout.addWidget(self.feedback_group)

        # Controls
        controls_layout = QHBoxLayout()
        self.back_btn = QPushButton("◀ Back")
        self.back_btn.clicked.connect(self.close)
        controls_layout.addWidget(self.back_btn)

        self.prev_frame_btn = QPushButton("⏮ Previous Frame")
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        controls_layout.addWidget(self.prev_frame_btn)

        self.next_frame_btn = QPushButton("Next Frame ⏭")
        self.next_frame_btn.clicked.connect(self.next_frame)
        controls_layout.addWidget(self.next_frame_btn)

        # Auto-advance toggle button
        self.auto_advance_btn = QPushButton("Auto Advance: ON")
        self.auto_advance_btn.setCheckable(True)
        self.auto_advance_btn.setChecked(True)
        self.auto_advance_btn.clicked.connect(self.toggle_auto_advance)
        controls_layout.addWidget(self.auto_advance_btn)

        # Voice feedback toggle button
        self.voice_feedback_btn = QPushButton("Voice Feedback: ON")
        self.voice_feedback_btn.setCheckable(True)
        self.voice_feedback_btn.setChecked(True)
        self.voice_feedback_btn.clicked.connect(self.toggle_voice_feedback)
        controls_layout.addWidget(self.voice_feedback_btn)

        self.help_btn = QPushButton("ℹ Help")
        self.help_btn.clicked.connect(self.show_help)
        controls_layout.addWidget(self.help_btn)

        layout.addLayout(controls_layout)

        # Progress
        self.progress_label = QLabel()
        layout.addWidget(self.progress_label)

    def speak_feedback(self, text):
        """Convert text to speech and play it."""
        if not self.voice_feedback_enabled:
            return

        def _speak():
            try:
                tts = gTTS(text=text, lang='en')
                fp = BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)

                pygame.mixer.music.load(fp)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            except Exception as e:
                print(f"Error in TTS: {e}")

        # Run in a thread to avoid blocking the UI
        threading.Thread(target=_speak, daemon=True).start()

    def toggle_voice_feedback(self):
        """Toggle voice feedback on/off."""
        self.voice_feedback_enabled = self.voice_feedback_btn.isChecked()
        if self.voice_feedback_enabled:
            self.voice_feedback_btn.setText("Voice Feedback: ON")
        else:
            self.voice_feedback_btn.setText("Voice Feedback: OFF")
            pygame.mixer.music.stop()

    def show_help(self):
        """Show help information about the practice mode."""
        help_text = """
        <b>Practice Mode Instructions</b>

        <p>1. Match your pose to the demonstration on the right.</p>
        <p>2. When your pose matches well (≥80%), the frame will automatically advance after 1 second.</p>
        <p>3. Use the navigation buttons to manually move between frames.</p>
        <p>4. Toggle 'Auto Advance' to control automatic progression.</p>
        <p>5. Voice feedback will guide you to correct your pose.</p>

        <p><b>Feedback Colors:</b></p>
        <p>• <span style='color:red'>Red</span>: Needs improvement</p>
        <p>• <span style='color:green'>Green</span>: Good match</p>
        """
        QMessageBox.information(self, "Practice Mode Help", help_text)

    def toggle_auto_advance(self):
        """Toggle automatic frame advancement."""
        self.auto_advance = self.auto_advance_btn.isChecked()
        if self.auto_advance:
            self.auto_advance_btn.setText("Auto Advance: ON")
            self.good_match_start_time = None  # Reset timer
        else:
            self.auto_advance_btn.setText("Auto Advance: OFF")
            self.countdown_label.clear()

    def load_correct_frames(self):
        """Load and process frames from the correct demonstration video"""
        try:
            cap = cv2.VideoCapture(self.correct_video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {self.correct_video_path}")

            # Get video properties to ensure consistent sizing
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.correct_frames = []
            all_frames = []

            # Read all frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame to consistent dimensions if needed
                if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                    frame = cv2.resize(frame, (frame_width, frame_height))

                all_frames.append(frame)

            cap.release()

            if not all_frames:
                raise ValueError("No frames found in the video")

            # Second pass: detect movement start
            movement_start_frame = 0
            with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
                for i, frame in enumerate(all_frames):
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(rgb_frame)

                    if results.pose_landmarks:
                        left_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
                        right_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]

                        # Detect when either hand moves up (signing starts)
                        if left_wrist.y < 0.7 or right_wrist.y < 0.7:
                            movement_start_frame = max(0, i - 5)  # Start 5 frames before movement
                            break

            # Third pass: process only frames from movement_start_frame onwards
            with mp_holistic.Holistic(min_detection_confidence=0.7) as holistic:
                for frame in all_frames[movement_start_frame:]:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.holistic.process(rgb_frame)

                    annotated_frame = frame.copy()
                    if results.pose_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            annotated_frame,
                            results.pose_landmarks,
                            mp.solutions.holistic.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                color=(0, 255, 0),
                                thickness=2
                            ),
                            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                color=(0, 200, 0),
                                thickness=2
                            )
                        )

                    self.correct_frames.append({
                        'frame': frame,
                        'landmarks': results.pose_landmarks,
                        'processed': annotated_frame
                    })

            if not self.correct_frames:
                raise ValueError("No valid frames with pose detected")

            self.update_progress_text()


        except Exception as e:
            QMessageBox.critical(self, "Error",    f"Failed to load demonstration video: {str(e)}")
            self.close()

    def update_progress_text(self):
        if self.correct_frames:
            progress_percent = ((self.current_frame_idx + 1) / len(self.correct_frames)) * 100
            progress_text = (
                f"Frame {self.current_frame_idx + 1} of {len(self.correct_frames)} - "
                f"{progress_percent:.1f}% complete"
            )
            self.progress_label.setText(progress_text)

    def update_frame(self):
        if not self.cap or not self.cap.isOpened() or self.current_frame_idx >= len(self.correct_frames):
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        current_time = time.time()
        correct_data = self.correct_frames[self.current_frame_idx]
        user_results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Process and update both frames
        camera_display = self.process_user_frame(frame, user_results, correct_data['landmarks'])
        self.update_label(self.camera_label, camera_display)

        # Ensure correct frame is properly sized
        correct_frame = correct_data['processed']
        self.update_label(self.correct_label, correct_frame)

        # Rest of the method remains the same...
        if user_results.pose_landmarks and correct_data['landmarks']:
            match_score, feedback = self.compare_poses(user_results, correct_data['landmarks'])
            self.current_match_score = match_score
            self.update_feedback(match_score, feedback)

            # Handle auto-advance logic
            if self.auto_advance and match_score >= self.match_threshold:
                if self.good_match_start_time is None:
                    self.good_match_start_time = current_time

                # Calculate remaining time
                elapsed = current_time - self.good_match_start_time
                remaining = max(0, self.match_duration - elapsed)
                self.countdown_label.setText(f"Auto-advancing in: {remaining:.1f}s")

                # Advance frame if held long enough
                if elapsed >= self.match_duration:
                    self.next_frame()
            else:
                self.good_match_start_time = None
                self.countdown_label.clear()

    def process_user_frame(self, frame, user_results, correct_landmarks):
        display_frame = frame.copy()

        if user_results.pose_landmarks and correct_landmarks:
            # Draw correct pose
            mp.solutions.drawing_utils.draw_landmarks(
                display_frame,
                correct_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255, 100)),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 200, 50)))

            # Draw user pose with color coding
            for connection in mp.solutions.holistic.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]

                user_start = user_results.pose_landmarks.landmark[start_idx]
                user_end = user_results.pose_landmarks.landmark[end_idx]
                correct_start = correct_landmarks.landmark[start_idx]
                correct_end = correct_landmarks.landmark[end_idx]

                start_error = np.sqrt((user_start.x - correct_start.x) ** 2 + (user_start.y - correct_start.y) ** 2)
                end_error = np.sqrt((user_end.x - correct_end.x) ** 2 + (user_end.y - correct_end.y) ** 2)
                avg_error = (start_error + end_error) / 2

                r = int(min(255, 255 * avg_error * 2))
                g = int(min(255, 255 * (1 - avg_error * 1.5)))
                b = 0

                start_point = (int(user_start.x * frame.shape[1]), int(user_start.y * frame.shape[0]))
                end_point = (int(user_end.x * frame.shape[1]), int(user_end.y * frame.shape[0]))
                cv2.line(display_frame, start_point, end_point, (b, g, r), 4)
                cv2.circle(display_frame, start_point, 6, (b, g, r), -1)
                cv2.circle(display_frame, end_point, 6, (b, g, r), -1)

        return display_frame

    def compare_poses(self, user_results, correct_landmarks):
        distances = []
        feedback = []

        for i in range(min(len(user_results.pose_landmarks.landmark), len(correct_landmarks.landmark))):
            user_lm = user_results.pose_landmarks.landmark[i]
            correct_lm = correct_landmarks.landmark[i]

            if user_lm.visibility > 0.5 and correct_lm.visibility > 0.5:
                dx = user_lm.x - correct_lm.x
                dy = user_lm.y - correct_lm.y
                distance = np.sqrt(dx ** 2 + dy ** 2)
                distances.append(distance)

                if distance > 0.1:
                    part_name = self.get_landmark_name(i)
                    direction = []
                    if abs(dx) > 0.05:
                        direction.append("left" if dx > 0 else "right")
                    if abs(dy) > 0.05:
                        direction.append("down" if dy > 0 else "up")

                    if direction:
                        feedback.append(f"{part_name} needs to move {' and '.join(direction)}")

        if not distances:
            return 0, ["Could not compare poses"]

        mean_distance = np.mean(distances)
        match_score = max(0, min(100, 100 * (1 - mean_distance * 2)))
        return match_score, feedback

    def get_landmark_name(self, idx):
        names = {
            0: "nose", 1: "left eye (inner)", 2: "left eye", 3: "left eye (outer)",
            4: "right eye (inner)", 5: "right eye", 6: "right eye (outer)",
            7: "left ear", 8: "right ear", 9: "mouth (left)", 10: "mouth (right)",
            11: "left shoulder", 12: "right shoulder", 13: "left elbow", 14: "right elbow",
            15: "left wrist", 16: "right wrist", 17: "left pinky", 18: "right pinky",
            19: "left index", 20: "right index", 21: "left thumb", 22: "right thumb",
            23: "left hip", 24: "right hip", 25: "left knee", 26: "right knee",
            27: "left ankle", 28: "right ankle", 29: "left heel", 30: "right heel",
            31: "left foot index", 32: "right foot index"
        }
        return names.get(idx, f"point {idx}")

    def update_feedback(self, score, feedback):
        self.match_meter.setValue(int(score))

        current_time = time.time()
        if feedback and current_time - self.last_feedback_time > self.feedback_cooldown:
            # Select the most important feedback item
            main_feedback = feedback[0] if feedback else ""

            # Generate voice feedback based on score
            if score >= 80:
                voice_text = "Excellent! You're matching the pose well."
            elif score >= 60:
                voice_text = "Good! " + main_feedback.replace("needs to move", "try moving")
            else:
                voice_text = "Let's improve. " + main_feedback.replace("needs to move", "please move")

            # Only speak if the feedback is different from last time
            if voice_text != self.last_spoken_feedback:
                self.speak_feedback(voice_text)
                self.last_spoken_feedback = voice_text
                self.last_feedback_time = current_time

        if score >= 80:
            self.feedback_text.setText("✅ Excellent match!")
        elif score >= 60:
            self.feedback_text.setText("🟡 Getting close!")
        else:
            self.feedback_text.setText("🔴 Keep practicing!")

        if feedback:
            self.detailed_feedback.setText("Suggestions:\n• " + "\n• ".join(feedback))

    def update_label(self, label, frame):
        """Update a QLabel with a frame while maintaining aspect ratio within fixed size"""
        try:
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w

            # Create QImage
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Create pixmap and scale to fit while maintaining aspect ratio
            pixmap = QPixmap.fromImage(q_img)

            # Calculate scaling to fit the fixed size while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Create a new pixmap with black background
            final_pixmap = QPixmap(label.size())
            final_pixmap.fill(Qt.black)

            # Calculate position to center the scaled pixmap
            x = (final_pixmap.width() - scaled_pixmap.width()) // 2
            y = (final_pixmap.height() - scaled_pixmap.height()) // 2

            # Draw the scaled pixmap centered on the black background
            painter = QPainter(final_pixmap)
            painter.drawPixmap(x, y, scaled_pixmap)
            painter.end()

            label.setPixmap(final_pixmap)

        except Exception as e:
            print(f"Error updating label: {str(e)}")
    def next_frame(self):
        """Move to next frame and reset auto-advance tracking"""
        if self.current_frame_idx < len(self.correct_frames) - 1:
            self.current_frame_idx += 1
            self.update_progress_text()
            self.good_match_start_time = None  # Reset auto-advance timer
            self.countdown_label.clear()

    def prev_frame(self):
        """Move to previous frame and reset auto-advance tracking"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.update_progress_text()
            self.good_match_start_time = None  # Reset auto-advance timer
            self.countdown_label.clear()

    def closeEvent(self, event):
        """Clean up resources when closing the widget"""
        try:
            # Release camera if active
            if self.cap and self.cap.isOpened():
                self.cap.release()

            # Stop any active timers
            if self.timer.isActive():
                self.timer.stop()

            # Close MediaPipe holistic
            if hasattr(self, 'holistic'):
                self.holistic.close()

            # Clean up audio
            try:
                pygame.mixer.quit()
            except:
                pass

        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

        event.accept()

import os
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

class GoogleDriveManager:
    def __init__(self):
        self.creds = None
        self.service = None
        self.root_folder_id = None
        self.initialize_drive()
        self.set_root_folder()  # שינוי כאן

    def initialize_drive(self):
        """Initialize Google Drive API with OAuth2."""
        if os.path.exists(TOKEN_FILE):
            self.creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_FILE, SCOPES)
                self.creds = flow.run_local_server(port=0)

            with open(TOKEN_FILE, 'w') as token:
                token.write(self.creds.to_json())

        self.service = build('drive', 'v3', credentials=self.creds)

    def set_root_folder(self):
        """Set known root folder ID directly (instead of searching by name)."""
        # ✅ זה ה-ID שלך:
        self.root_folder_id = '1t6JIQVCiZXgH2IRm6eBgh-kSXE87aF7K'
        print(f"Using hardcoded root folder ID: {self.root_folder_id}")

    def get_folder_id(self, folder_name, parent_id=None):
        """Get folder ID by name, optionally within a parent folder."""
        parent = parent_id or self.root_folder_id
        if not parent:
            return None

        query = f"'{parent}' in parents and name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        results = self.service.files().list(
            q=query,
            pageSize=1,
            fields="files(id, name)"
        ).execute()
        return results.get('files', [None])[0]

    def download_file(self, file_id, destination):
        """Download a file from Google Drive."""
        request = self.service.files().get_media(fileId=file_id)
        fh = io.FileIO(destination, 'wb')

        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%")

    def list_files(self, folder_id=None, mime_type=None):
        """List files in a folder with optional MIME type filter."""
        parent = folder_id or self.root_folder_id
        if not parent:
            return []

        query = f"'{parent}' in parents and trashed=false"
        if mime_type:
            query += f" and mimeType='{mime_type}'"

        results = self.service.files().list(
            q=query,
            pageSize=35,
            fields="nextPageToken, files(id, name, mimeType)"
        ).execute()
        return results.get('files', [])


import vlc

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor

class VideoPlayerWidget(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()
        self.media = None  # Store media reference for reuse

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel(self.title)
        title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title_label)

        # Video area
        self.video_frame = QWidget()
        self.video_frame.setMinimumSize(500, 400)
        self.video_frame.setAutoFillBackground(True)
        palette = self.video_frame.palette()
        palette.setColor(QPalette.Window, QColor(0, 0, 0))
        self.video_frame.setPalette(palette)

        layout.addWidget(self.video_frame)

        # Controls
        controls = QHBoxLayout()

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.play)
        controls.addWidget(self.play_btn)

        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.pause)
        controls.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop)
        controls.addWidget(self.stop_btn)

        layout.addLayout(controls)

    def play_video(self, file_path):
        print(f"🎬 VLC: Preparing video → {file_path}")
        self.media = self.instance.media_new(file_path)
        self.player.set_media(self.media)

        if sys.platform.startswith('linux'):
            self.player.set_xwindow(self.video_frame.winId())
        elif sys.platform == "win32":
            self.player.set_hwnd(int(self.video_frame.winId()))
        elif sys.platform == "darwin":
            self.player.set_nsobject(int(self.video_frame.winId()))

        self.player.play()
        print("🎬 VLC: Playback started")

    def play(self):
        if self.media is None:
            print("⚠ No media loaded, nothing to play.")
            return

        state = self.player.get_state()
        print("▶ Resume or start playback — current state:", state)

        if state in [vlc.State.Ended, vlc.State.Stopped]:
            self.player.stop()  # Fully reset
            self.player.set_media(self.media)  # Reassign the media
            self.player.play()
            print("🔄 Reloaded media and started again")
        else:
            self.player.play()

    def pause(self):
        print("⏸ Pausing playback")
        self.player.pause()

    def stop(self):
        print("⏹ Stopping playback")
        self.player.stop()


class ProgressDashboard(QWidget):
    def __init__(self, user_data, parent=None):
        super().__init__(parent)
        self.user_data = user_data
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Progress Dashboard")
        self.setFixedSize(1000, 800)

        # Main layout with scroll area
        main_layout = QVBoxLayout(self)

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create scroll content widget
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(20, 20, 20, 20)
        scroll_layout.setSpacing(20)

        # Create tab widget inside scroll area
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                padding: 10px 20px;
                font-size: 14px;
            }
            QTabBar::tab:selected {
                background: #6e8efb;
                color: white;
                border-radius: 4px;
            }
        """)
        scroll_layout.addWidget(self.tabs)

        # Add tabs with real data
        self.setup_overview_tab()
        self.setup_words_tab()
        self.setup_games_tab()

        # Set the scroll content
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    def setup_overview_tab(self):
        """Setup the overview tab with real user stats"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(20)

        # Header with improved styling
        header = QLabel(f"📊 {self.user_data['username']}'s Progress Overview")
        header.setStyleSheet("""
            font-size: 24px; 
            font-weight: bold; 
            color: #4a6baf;
            padding-bottom: 10px;
            border-bottom: 2px solid #6e8efb;
        """)
        layout.addWidget(header)

        # Stats cards with improved layout
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)

        # Learning Progress Card with real data
        learning_card = QGroupBox("📚 Learning Progress")
        learning_card.setStyleSheet("""
            QGroupBox {
                border: 2px solid #6e8efb;
                border-radius: 8px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #4a6baf;
                font-weight: bold;
            }
        """)
        learning_layout = QVBoxLayout()
        learning_layout.setSpacing(10)

        stats = self.user_data['stats']

        # Calculate accuracy safely
        accuracy = 0.0
        if stats['total_words_attempted'] > 0:
            accuracy = (stats['total_words_correct'] / stats['total_words_attempted']) * 100

        learning_points = stats['learning_points']
        total_sessions = stats['total_practice_sessions']
        streak_days = stats['streak_days']
        last_practice = stats.get('last_practice_date', 'Never')

        sessions_label = QLabel(f"🔄 Practice Sessions: {total_sessions}")
        sessions_label.setStyleSheet("font-size: 14px;")

        accuracy_label = QLabel(f"🎯 Accuracy: {accuracy:.1f}%")
        accuracy_label.setStyleSheet("font-size: 14px;")

        points_label = QLabel(f"⭐ Learning Points: {learning_points}")
        points_label.setStyleSheet("font-size: 14px;")

        learning_layout.addWidget(sessions_label)
        learning_layout.addWidget(accuracy_label)
        learning_layout.addWidget(points_label)
        learning_card.setLayout(learning_layout)
        stats_layout.addWidget(learning_card)

        # Game Progress Card with real data
        game_card = QGroupBox("🎮 Game Progress")
        game_card.setStyleSheet(learning_card.styleSheet())
        game_layout = QVBoxLayout()
        game_layout.setSpacing(10)

        game_points = stats['game_points']
        quiz_points = game_points.get('quiz', 0)
        clock_points = game_points.get('clock', 0)
        total_points = quiz_points + clock_points

        total_label = QLabel(f"🏆 Total Game Points: {total_points}")
        total_label.setStyleSheet("font-size: 14px;")

        quiz_label = QLabel(f"❓ Quiz Points: {quiz_points}")
        quiz_label.setStyleSheet("font-size: 14px;")

        clock_label = QLabel(f"⏱ Clock Game Points: {clock_points}")
        clock_label.setStyleSheet("font-size: 14px;")

        game_layout.addWidget(total_label)
        game_layout.addWidget(quiz_label)
        game_layout.addWidget(clock_label)
        game_card.setLayout(game_layout)
        stats_layout.addWidget(game_card)

        # Streak Card with real data
        streak_card = QGroupBox("🔥 Streaks")
        streak_card.setStyleSheet(learning_card.styleSheet())
        streak_layout = QVBoxLayout()
        streak_layout.setSpacing(10)

        streak_label = QLabel(f"📅 Current Streak: {streak_days} days")
        streak_label.setStyleSheet("font-size: 14px;")

        last_label = QLabel(f"🕒 Last Practiced: {last_practice}")
        last_label.setStyleSheet("font-size: 14px;")

        streak_layout.addWidget(streak_label)
        streak_layout.addWidget(last_label)
        streak_card.setLayout(streak_layout)
        stats_layout.addWidget(streak_card)

        layout.addLayout(stats_layout)

        # Progress Charts with real data
        chart_frame = QGroupBox("📈 Progress Charts")
        chart_frame.setStyleSheet(learning_card.styleSheet())
        chart_layout = QVBoxLayout(chart_frame)

        # Weekly progress chart with real data
        weekly_data = self.generate_weekly_data()
        weekly_chart = self.create_real_chart("Weekly Progress", weekly_data)
        chart_layout.addWidget(weekly_chart)

        # Accuracy chart with real data
        accuracy_data = self.generate_accuracy_data()
        accuracy_chart = self.create_real_chart("Accuracy Trend", accuracy_data)
        chart_layout.addWidget(accuracy_chart)

        layout.addWidget(chart_frame)

        self.tabs.addTab(tab, "Overview")

    def setup_words_tab(self):
        """Setup the words tab with real word progress data"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(20)

        # Known words section with real data
        known_frame = QGroupBox("✅ Known Words")
        known_frame.setStyleSheet("""
            QGroupBox {
                border: 2px solid #4CAF50;
                border-radius: 8px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #388E3C;
                font-weight: bold;
            }
        """)
        known_layout = QVBoxLayout(known_frame)

        if not self.user_data.get('known_words', []):
            empty_label = QLabel("You haven't mastered any words yet. Keep practicing!")
            empty_label.setStyleSheet("font-size: 14px; color: #666;")
            known_layout.addWidget(empty_label)
        else:
            words_layout = QGridLayout()
            words_layout.setSpacing(10)
            for i, word in enumerate(sorted(self.user_data['known_words'])):
                word_label = QLabel(word)
                word_label.setStyleSheet("""
                    QLabel {
                        font-size: 16px; 
                        padding: 8px 12px;
                        background-color: #E8F5E9;
                        border-radius: 4px;
                    }
                """)
                words_layout.addWidget(word_label, i // 4, i % 4)

            known_layout.addLayout(words_layout)

        layout.addWidget(known_frame)

        # Words progress section with real data
        progress_frame = QGroupBox("📊 Word Progress")
        progress_frame.setStyleSheet(known_frame.styleSheet().replace("4CAF50", "6e8efb").replace("388E3C", "4a6baf"))
        progress_layout = QVBoxLayout(progress_frame)

        if not self.user_data.get('word_progress', {}):
            empty_label = QLabel("No word progress data available yet.")
            empty_label.setStyleSheet("font-size: 14px; color: #666;")
            progress_layout.addWidget(empty_label)
        else:
            table = QTableWidget()
            table.setColumnCount(3)
            table.setHorizontalHeaderLabels(["Word", "Attempts", "Accuracy"])
            table.setStyleSheet("""
                QTableWidget {
                    font-size: 14px;
                    alternate-background-color: #f0f0f0;
                }
                QHeaderView::section {
                    background-color: #6e8efb;
                    color: white;
                    padding: 5px;
                    font-weight: bold;
                }
            """)
            table.verticalHeader().setVisible(False)
            table.setAlternatingRowColors(True)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.setSelectionBehavior(QTableWidget.SelectRows)
            table.setSelectionMode(QTableWidget.SingleSelection)

            # Sort words by accuracy (highest first)
            words = sorted(self.user_data['word_progress'].items(),
                           key=lambda x: x[1].get('correct', 0) / x[1].get('attempts', 1) if x[1].get('attempts',
                                                                                                      0) > 0 else 0,
                           reverse=True)

            table.setRowCount(len(words))

            for row, (word, data) in enumerate(words):
                attempts = data.get('attempts', 0)
                correct = data.get('correct', 0)
                accuracy = (correct / attempts * 100) if attempts > 0 else 0

                table.setItem(row, 0, QTableWidgetItem(word))
                table.setItem(row, 1, QTableWidgetItem(str(attempts)))

                accuracy_item = QTableWidgetItem(f"{accuracy:.1f}%")
                # Color code accuracy
                if accuracy >= 80:
                    accuracy_item.setForeground(QColor(46, 125, 50))  # Green
                elif accuracy >= 50:
                    accuracy_item.setForeground(QColor(251, 192, 45))  # Yellow
                else:
                    accuracy_item.setForeground(QColor(198, 40, 40))  # Red

                table.setItem(row, 2, accuracy_item)

            table.resizeColumnsToContents()
            progress_layout.addWidget(table)

        layout.addWidget(progress_frame)
        layout.addStretch()

        self.tabs.addTab(tab, "Words")

    def setup_games_tab(self):
        """Setup the games tab with real game stats"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(20)

        # Quiz game stats with real data
        quiz_frame = QGroupBox("❓ Quiz Game Stats")
        quiz_frame.setStyleSheet("""
            QGroupBox {
                border: 2px solid #FF9800;
                border-radius: 8px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #E65100;
                font-weight: bold;
            }
        """)
        quiz_layout = QVBoxLayout(quiz_frame)
        quiz_layout.setSpacing(10)

        stats = self.user_data['stats']
        quiz_points = stats['game_points'].get('quiz', 0)
        quiz_layout.addWidget(QLabel(f"🏆 Total Points: {quiz_points}"))

        # Calculate accuracy for quiz
        quiz_accuracy = 0
        if 'total_quiz_attempts' in stats and stats['total_quiz_attempts'] > 0:
            quiz_accuracy = (stats.get('total_quiz_correct', 0) /
                             stats['total_quiz_attempts']) * 100
        quiz_layout.addWidget(QLabel(f"🎯 Accuracy: {quiz_accuracy:.1f}%"))

        # Add chart for quiz performance with real data
        quiz_data = self.generate_quiz_history()
        quiz_chart = self.create_real_chart("Quiz Performance", quiz_data)
        quiz_layout.addWidget(quiz_chart)

        layout.addWidget(quiz_frame)

        # Clock game stats with real data
        clock_frame = QGroupBox("⏱ Clock Game Stats")
        clock_frame.setStyleSheet(quiz_frame.styleSheet().replace("FF9800", "2196F3").replace("E65100", "1565C0"))
        clock_layout = QVBoxLayout(clock_frame)
        clock_layout.setSpacing(10)

        clock_points = stats['game_points'].get('clock', 0)
        clock_layout.addWidget(QLabel(f"🏆 Total Points: {clock_points}"))

        # Calculate accuracy for clock game
        clock_accuracy = 0
        if 'total_clock_attempts' in stats and stats['total_clock_attempts'] > 0:
            clock_accuracy = (stats.get('total_clock_correct', 0) /
                              stats['total_clock_attempts']) * 100
        clock_layout.addWidget(QLabel(f"🎯 Accuracy: {clock_accuracy:.1f}%"))

        # Add chart for clock game performance with real data
        clock_data = self.generate_clock_history()
        clock_chart = self.create_real_chart("Clock Game Performance", clock_data)
        clock_layout.addWidget(clock_chart)

        layout.addWidget(clock_frame)
        layout.addStretch()

        self.tabs.addTab(tab, "Games")

    def generate_weekly_data(self):
        """Generate weekly progress data based on user history"""
        # In a real app, this would come from user's history
        # For now, we'll simulate based on total sessions
        total_sessions = self.user_data['stats']['total_practice_sessions']
        return [min(5 + i * 2, total_sessions) for i in range(7)]

    def generate_accuracy_data(self):
        """Generate accuracy trend data based on user history"""
        # In a real app, this would come from user's history
        # For now, we'll simulate based on accuracy
        accuracy = 0
        stats = self.user_data['stats']
        if stats['total_words_attempted'] > 0:
            accuracy = (stats['total_words_correct'] / stats['total_words_attempted']) * 100

        # Simulate progression from 50% to current accuracy
        return [max(50, min(100, int(50 + i * (accuracy - 50) / 5))) for i in range(6)]

    def generate_quiz_history(self):
        """Generate quiz game history data"""
        stats = self.user_data['stats']
        if 'quiz_history' in stats:
            return [score for score in stats['quiz_history']]

        # Simulate based on total points if no history exists
        quiz_points = stats['game_points'].get('quiz', 0)
        return [min(100, quiz_points // 2), min(200, quiz_points // 2 + 50),
                min(300, quiz_points // 2 + 100), quiz_points]

    def generate_clock_history(self):
        """Generate clock game history data"""
        stats = self.user_data['stats']
        if 'clock_history' in stats:
            return [score for score in stats['clock_history']]

        # Simulate based on total points if no history exists
        clock_points = stats['game_points'].get('clock', 0)
        return [min(100, clock_points // 2), min(200, clock_points // 2 + 50),
                min(300, clock_points // 2 + 100), clock_points]

    def create_real_chart(self, title, data, color="#4a6baf"):
        """Create a chart widget with the given real data"""
        chart_widget = QWidget()
        chart_widget.setMinimumHeight(200)
        layout = QVBoxLayout(chart_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Chart title
        title_label = QLabel(title)
        title_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")
        layout.addWidget(title_label)

        # Chart drawing area
        chart_area = QLabel()
        chart_area.setAlignment(Qt.AlignCenter)
        chart_area.setStyleSheet("background-color: white; border: 1px solid #ddd;")

        # Generate chart pixmap
        pixmap = self.draw_bar_chart(data, color, chart_area.size())
        chart_area.setPixmap(pixmap)

        layout.addWidget(chart_area)
        return chart_widget

    def draw_bar_chart(self, data, color, size):
        """Draw a bar chart with the given data"""
        if not data:
            return QPixmap(size)

        # Create pixmap
        pixmap = QPixmap(size)
        pixmap.fill(Qt.white)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Chart dimensions
        margin = 20
        chart_width = size.width() - 2 * margin
        chart_height = size.height() - 2 * margin
        bar_width = chart_width / (len(data) * 1.5)  # Leave space between bars

        # Find max value for scaling
        max_value = max(data) if max(data) > 0 else 1

        # Draw grid lines
        pen = QPen(QColor(200, 200, 200))
        pen.setWidth(1)
        painter.setPen(pen)

        # Draw bars
        for i, value in enumerate(data):
            # Calculate bar dimensions
            bar_height = (value / max_value) * chart_height
            x = margin + i * (bar_width * 1.5)
            y = size.height() - margin - bar_height

            # Draw bar
            bar_color = QColor(color)
            painter.setBrush(bar_color)
            painter.setPen(Qt.NoPen)
            painter.drawRect(QRectF(x, y, bar_width, bar_height))

            # Draw value label
            painter.setPen(Qt.black)
            painter.drawText(QRectF(x, y - 20, bar_width, 20),
                             Qt.AlignCenter, str(value))

        # Draw axes
        pen = QPen(Qt.black)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(margin, size.height() - margin,
                         size.width() - margin, size.height() - margin)  # X-axis
        painter.drawLine(margin, size.height() - margin,
                         margin, margin)  # Y-axis

        painter.end()
        return pixmap

class SignRecognitionGame(QMainWindow):
    finished = pyqtSignal()

    def __init__(self, words, user_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sign Against the Clock")
        self.setWindowModality(Qt.WindowModal)
        self.setFixedSize(1000, 700)

        self.words = words
        self.user_data = user_data
        self.score = 0
        self.rounds = 0
        self.max_rounds = 5
        self.current_word = None
        self.is_recording = False
        self.max_recording_time = 3  # seconds

        # Setup UI
        self.setup_ui()

        # Start with instructions
        self.show_instructions()

    def setup_ui(self):
        """Setup the game UI with stacked widgets."""
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Setup all pages
        self.setup_instructions_page()
        self.setup_game_page()
        self.setup_results_page()

    def setup_instructions_page(self):
        """Setup the game instructions page."""
        self.instructions_page = QWidget()
        layout = QVBoxLayout(self.instructions_page)
        layout.setAlignment(Qt.AlignCenter)

        # Title
        title = QLabel("Sign Against the Clock")
        title.setStyleSheet("font-size: 28px; font-weight: bold;")
        layout.addWidget(title, 0, Qt.AlignCenter)

        # Game instructions
        instructions = QLabel(
            "How to Play:\n\n"
            "1. You'll be shown a sign to perform\n"
            "2. Record yourself signing the word in 3 seconds\n"
            "3. Get immediate feedback on your accuracy\n"
            "4. Complete 5 signs to finish the game\n\n"
            "Recording Tips:\n"
            "- Stand about 2 meters from your camera\n"
            "- Make sure your whole upper body is visible\n"
            "- Keep your hands at waist level when starting\n"
            "- The recording will automatically stop after 3 seconds"
        )
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("font-size: 16px;")
        layout.addWidget(instructions)

        # Start button
        start_btn = QPushButton("Let's Begin!")
        start_btn.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                padding: 12px 30px;
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        start_btn.clicked.connect(self.start_game)
        layout.addWidget(start_btn, 0, Qt.AlignCenter)

        # Home button
        home_btn = QPushButton("Return to Home")
        home_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 8px 20px;
                background-color: #6e8efb;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #5a7df4;
            }
        """)
        home_btn.clicked.connect(self.close)
        layout.addWidget(home_btn, 0, Qt.AlignCenter)

        layout.addStretch()
        self.stacked_widget.addWidget(self.instructions_page)

    def setup_game_page(self):
        """Setup the main game play page."""
        self.game_page = QWidget()
        layout = QVBoxLayout(self.game_page)

        # Score and round display
        info_layout = QHBoxLayout()
        self.score_label = QLabel("Score: 0")
        self.score_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.round_label = QLabel("Round: 0/5")
        self.round_label.setStyleSheet("font-size: 18px;")
        info_layout.addWidget(self.score_label)
        info_layout.addStretch()
        info_layout.addWidget(self.round_label)
        layout.addLayout(info_layout)

        # Current word display
        self.word_label = QLabel()
        self.word_label.setStyleSheet("font-size: 36px; font-weight: bold;")
        self.word_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.word_label)

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)

        # Recording time display
        self.recording_time_label = QLabel("Recording: 0.0s")
        self.recording_time_label.setAlignment(Qt.AlignCenter)
        self.recording_time_label.hide()
        layout.addWidget(self.recording_time_label)

        # Record button
        self.record_btn = QPushButton("⏺ Start Recording")
        self.record_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 12px;
                background-color: #f44336;
                color: white;
                border-radius: 8px;
            }
        """)
        self.record_btn.clicked.connect(self.start_recording)
        layout.addWidget(self.record_btn, 0, Qt.AlignCenter)

        # Progress bar for analysis
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Feedback label
        self.feedback_label = QLabel()
        self.feedback_label.setAlignment(Qt.AlignCenter)
        self.feedback_label.setStyleSheet("font-size: 20px;")
        layout.addWidget(self.feedback_label)

        self.stacked_widget.addWidget(self.game_page)

    def setup_results_page(self):
        """Setup the results page."""
        self.results_page = QWidget()
        layout = QVBoxLayout(self.results_page)
        layout.setAlignment(Qt.AlignCenter)

        # Results title
        title = QLabel("Game Results")
        title.setStyleSheet("font-size: 28px; font-weight: bold;")
        layout.addWidget(title, 0, Qt.AlignCenter)

        # Results text
        self.results_text = QLabel()
        self.results_text.setAlignment(Qt.AlignCenter)
        self.results_text.setStyleSheet("font-size: 18px;")
        layout.addWidget(self.results_text)

        # Home button
        home_btn = QPushButton("Return to Home")
        home_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 12px 24px;
                background-color: #6e8efb;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #5a7df4;
            }
        """)
        home_btn.clicked.connect(self.close)
        layout.addWidget(home_btn, 0, Qt.AlignCenter)

        layout.addStretch()
        self.stacked_widget.addWidget(self.results_page)

    def show_instructions(self):
        """Show the instructions page."""
        self.stacked_widget.setCurrentWidget(self.instructions_page)

    def start_game(self):
        """Start the game after instructions."""
        self.score = 0
        self.rounds = 0
        self.update_score_display()
        self.stacked_widget.setCurrentWidget(self.game_page)
        self.start_new_round()

    def start_new_round(self):
        """Start a new round with a new word."""
        if self.rounds >= self.max_rounds:
            self.show_results()
            return

        self.rounds += 1
        self.current_word = random.choice(self.words)
        self.word_label.setText(self.current_word)
        self.feedback_label.clear()
        self.record_btn.setEnabled(True)
        self.record_btn.setText("⏺ Start Recording")
        self.recording_time_label.hide()
        self.progress_bar.hide()

        self.update_score_display()

    def start_recording(self):
        # Initialize camera earlier
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            QMessageBox.critical(self, "Camera Error", "Could not access camera")
            return False
        """Start the recording process with countdown."""
        self.is_recording = False  # Not actually recording during countdown
        self.record_btn.setEnabled(False)
        self.record_btn.setText("Get Ready...")
        self.recording_time_label.hide()
        self.feedback_label.clear()

        # Initialize countdown
        self.countdown_value = 3
        self.countdown_label = QLabel("3", self.video_label)
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet("""
            QLabel {
                font-size: 120px; 
                color: white;
                background-color: rgba(0, 0, 0, 150);
                border-radius: 60px;
                padding: 40px;
            }
        """)
        self.countdown_label.setFixedSize(200, 200)
        self.countdown_label.move(
            (self.video_label.width() - 200) // 2,
            (self.video_label.height() - 200) // 2
        )
        self.countdown_label.show()

        # Setup camera preview without recording
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
        if not self.capture.isOpened():
            QMessageBox.critical(self, "Camera Error", "Could not access camera")
            self.stop_recording()
            return

        # Start countdown timer
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)  # 1 second intervals

        # Start camera preview
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_timer.start(30)  # ~30fps

    def update_countdown(self):
        """Update the countdown display."""
        self.countdown_value -= 1

        if self.countdown_value > 0:
            self.countdown_label.setText(str(self.countdown_value))
            # Play countdown sound if available
            try:
                QSound.play(":/sounds/beep.wav")
            except:
                pass
        elif self.countdown_value == 0:
            self.countdown_label.setText("GO!")
            try:
                QSound.play(":/sounds/go.wav")
            except:
                pass
        else:
            # Countdown finished - start actual recording
            self.countdown_timer.stop()
            self.countdown_label.hide()
            self.countdown_label.deleteLater()
            self.start_actual_recording()

    def update_preview(self):
        """Update the camera preview and handle countdown/recording."""
        if not hasattr(self, 'capture') or not self.capture or not self.capture.isOpened():
            print("Camera not available")
            return

        """Update camera preview during countdown."""
        if not hasattr(self, 'capture') or not self.capture.isOpened():
            return

        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.KeepAspectRatio
            ))

    def start_actual_recording(self):
        """Begin actual video recording after countdown."""
        self.is_recording = True
        if hasattr(self, 'stopping'):
            del self.stopping
        self.record_btn.setText("⏹ Recording...")
        self.recording_time_label.show()

        # Initialize recording
        self.recorded_frames = []
        self.recording_start_time = time.time()

        # Setup recording timer
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording)
        self.recording_timer.start(30)  # ~30fps

        # Stop preview timer
        if hasattr(self, 'preview_timer') and self.preview_timer.isActive():
            self.preview_timer.stop()

    def update_recording(self):
        """Update recording display and check duration."""
        elapsed = time.time() - self.recording_start_time
        self.recording_time_label.setText(f"Recording: {elapsed:.1f}s")

        # Capture frame
        ret, frame = self.capture.read()
        if ret:
            self.recorded_frames.append(frame)

            # Show preview
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.KeepAspectRatio
            ))

        # Stop after 3 seconds using single-shot to avoid reentrancy
        if elapsed >= self.max_recording_time and not hasattr(self, 'stopping'):
            self.stopping = True
            QTimer.singleShot(0, self.stop_recording)

    def stop_recording(self):
        """Stop recording and clean up."""
        # Stop all timers
        if hasattr(self, 'recording_timer') and self.recording_timer.isActive():
            self.recording_timer.stop()

        if hasattr(self, 'preview_timer') and self.preview_timer.isActive():
            self.preview_timer.stop()

        if hasattr(self, 'countdown_timer') and self.countdown_timer.isActive():
            self.countdown_timer.stop()

        # Release camera
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()

        # Reset state
        self.is_recording = False
        self.record_btn.setEnabled(True)
        self.record_btn.setText("⏺ Start Recording")

        # Close preview window if it exists
        if hasattr(self, 'preview_window'):
            self.preview_window.close()
            del self.preview_window

        # If we have recorded frames, show review dialog
        if hasattr(self, 'recording_started') and len(self.recorded_frames) > 0:
            self.review_recording()

        # Clean up
        if hasattr(self, 'recording_started'):
            del self.recording_started
        if hasattr(self, 'recording_start_time'):
            del self.recording_start_time


    def analyze_recording(self):
        """Analyze the recorded video with progress updates."""
        self.feedback_label.setText("Analyzing your sign...")
        self.progress_bar.show()

        # Simulate analysis progress (replace with real analysis)
        self.progress_timer = QTimer()
        self.progress_value = 0

        def update_progress():
            self.progress_value += 5
            self.progress_bar.setValue(self.progress_value)

            if self.progress_value >= 100:
                self.progress_timer.stop()
                self.finish_analysis()

        self.progress_timer.timeout.connect(update_progress)
        self.progress_timer.start(100)

    def finish_analysis(self):
        """Finish the analysis and show results."""
        self.progress_bar.hide()

        # In a real implementation, you would:
        # 1. Save the recorded frames to a temp file
        # 2. Use your pose comparison logic
        # 3. Compare against the current word's threshold

        # For now, we'll simulate a random result
        is_correct = random.random() > 0.5

        if is_correct:
            self.score += 1
            self.feedback_label.setText("✅ Correct! Well done!")
        else:
            self.feedback_label.setText("❌ Incorrect. Try again next round!")

        self.update_score_display()

        # Move to next round after delay
        QTimer.singleShot(3000, self.start_new_round)

    def update_score_display(self):
        """Update the score and round displays."""
        self.score_label.setText(f"Score: {self.score}")
        self.round_label.setText(f"Round: {self.rounds}/{self.max_rounds}")

    def show_results(self):
        """Show the final results."""
        accuracy = (self.score / self.max_rounds) * 100
        self.results_text.setText(
            f"Final Score: {self.score}/{self.max_rounds}\n"
            f"Accuracy: {accuracy:.1f}%\n\n"
            f"{'🌟 Excellent!' if accuracy >= 80 else '👍 Good job!' if accuracy >= 50 else '💪 Keep practicing!'}"
        )
        # Update game points
        self.user_data['stats']['game_points']['clock'] = self.score



        # Update user stats
        self.user_data['stats']['total_practice_sessions'] += 1
        self.user_data['stats']['total_words_attempted'] += self.max_rounds
        self.user_data['stats']['total_words_correct'] += self.score
        # Save user data
        self.parent().save_user_data()

        self.stacked_widget.setCurrentWidget(self.results_page)

    def closeEvent(self, event):
        """Clean up resources when closing."""
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()

        if hasattr(self, 'recording_timer') and self.recording_timer.isActive():
            self.recording_timer.stop()

        if hasattr(self, 'progress_timer') and self.progress_timer.isActive():
            self.progress_timer.stop()

        event.accept()


class SignQuizGame(QMainWindow):
    finished = pyqtSignal()

    def __init__(self, words, user_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Handly - Sign Quiz")
        self.setWindowModality(Qt.WindowModal)
        self.setFixedSize(1000, 700)

        self.words = words
        self.user_data = user_data
        self.score = 0
        self.current_round = 0
        self.total_rounds = 5
        self.time_left = 0
        self.current_word = None
        self.options = []
        self.streak = 0
        self.max_streak = 0
        self.temp_video_files = []

        # Setup central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)

        # Create stacked widget for different screens
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        # Setup all pages
        self.setup_instructions_page()
        self.setup_quiz_page()
        self.setup_results_page()

        # Start with instructions
        self.show_instructions()

        # Initialize MediaPipe
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def setup_instructions_page(self):
        """Setup the instructions page."""
        self.instructions_page = QWidget()
        layout = QVBoxLayout(self.instructions_page)
        layout.setAlignment(Qt.AlignCenter)

        # Title
        title = QLabel("Sign Language Quiz")
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #4a6baf;
            margin-bottom: 20px;
        """)
        layout.addWidget(title, 0, Qt.AlignCenter)

        # Instructions
        instructions = QLabel(
            "How to Play:\n\n"
            "1. Watch the sign language demonstration\n"
            "2. Select the correct word from 4 options\n"
            "3. The faster you answer, the more points you get\n"
            "4. Streaks multiply your points\n\n"
            "Ready to begin?"
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("""
            font-size: 18px;
            margin-bottom: 30px;
        """)
        layout.addWidget(instructions)

        # Start button - FIXED CONNECTION
        start_btn = QPushButton("Start Quiz")
        start_btn.setStyleSheet("""
                QPushButton {
                    font-size: 20px;
                    padding: 12px 30px;
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 8px;
                    min-width: 200px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
        start_btn.clicked.connect(self.start_quiz)  # This connection was correct

        # Home button - FIXED CONNECTION
        home_btn = QPushButton("Return to Main Menu")
        home_btn.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    padding: 8px 20px;
                    background-color: #6e8efb;
                    color: white;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #5a7df4;
                }
            """)
        home_btn.clicked.connect(self.close)  # This connection was correct

        layout.addWidget(start_btn, 0, Qt.AlignCenter)
        layout.addWidget(home_btn, 0, Qt.AlignCenter)

        layout.addStretch()
        self.stacked_widget.addWidget(self.instructions_page)

    def setup_quiz_page(self):
        """Setup the quiz gameplay page."""
        self.quiz_page = QWidget()
        main_layout = QHBoxLayout(self.quiz_page)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left sidebar (20%)
        sidebar = QFrame()
        sidebar.setStyleSheet("background-color: #f0f0f0;")
        sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)

        # Score display
        self.score_label = QLabel("Score: 0")
        self.score_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        sidebar_layout.addWidget(self.score_label)

        # Streak display
        self.streak_label = QLabel("Streak: 0x")
        self.streak_label.setStyleSheet("font-size: 16px; color: #FFD700;")
        sidebar_layout.addWidget(self.streak_label)

        # Timer display
        self.timer_label = QLabel("Time: 0s")
        self.timer_label.setStyleSheet("font-size: 16px;")
        sidebar_layout.addWidget(self.timer_label)

        # Round display
        self.round_label = QLabel(f"Round: 0/{self.total_rounds}")
        self.round_label.setStyleSheet("font-size: 16px;")
        sidebar_layout.addWidget(self.round_label)

        # Home button
        home_btn = QPushButton("🏠 Home")
        home_btn.setStyleSheet("""
            QPushButton {
                padding: 8px;
                margin-top: 20px;
                background-color: #6e8efb;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #5a7df4;
            }
        """)
        home_btn.clicked.connect(self.close)
        sidebar_layout.addWidget(home_btn)

        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar)

        # Right content (80%)
        content = QFrame()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 20, 20, 20)

        # Video display
        self.video_player = VideoPlayerWidget("Sign Demonstration")
        self.video_player.setMinimumHeight(400)
        content_layout.addWidget(self.video_player)

        # Question label
        self.question_label = QLabel("What word is being signed?")
        self.question_label.setStyleSheet("font-size: 20px;")
        self.question_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.question_label)

        # Options grid
        options_frame = QFrame()
        options_layout = QGridLayout(options_frame)
        options_layout.setContentsMargins(50, 20, 50, 20)
        options_layout.setSpacing(15)

        self.option_buttons = []
        for i in range(4):
            btn = QPushButton()
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 18px;
                    padding: 15px;
                    background-color: #6e8efb;
                    color: white;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #5a7df4;
                }
            """)
            btn.clicked.connect(lambda _, idx=i: self.check_answer(idx))
            self.option_buttons.append(btn)
            options_layout.addWidget(btn, i // 2, i % 2)

        content_layout.addWidget(options_frame)
        main_layout.addWidget(content)

        self.stacked_widget.addWidget(self.quiz_page)

    def setup_results_page(self):
        """Setup the results page."""
        self.results_page = QWidget()
        layout = QVBoxLayout(self.results_page)
        layout.setAlignment(Qt.AlignCenter)

        # Results title
        title = QLabel("Quiz Results")
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #4a6baf;
            margin-bottom: 20px;
        """)
        layout.addWidget(title, 0, Qt.AlignCenter)

        # Results text
        self.results_text = QLabel()
        self.results_text.setAlignment(Qt.AlignCenter)
        self.results_text.setStyleSheet("font-size: 18px; margin-bottom: 30px;")
        layout.addWidget(self.results_text)

        # Home button
        home_btn = QPushButton("Return to Main Menu")
        home_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 12px 24px;
                background-color: #6e8efb;
                color: white;
                border-radius: 8px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #5a7df4;
            }
        """)
        home_btn.clicked.connect(self.close)
        layout.addWidget(home_btn, 0, Qt.AlignCenter)

        layout.addStretch()
        self.stacked_widget.addWidget(self.results_page)

    def show_instructions(self):
        """Show the instructions page."""
        self.stacked_widget.setCurrentWidget(self.instructions_page)

    def start_quiz(self):
        """Start the quiz from instructions."""
        self.score = 0
        self.current_round = 0
        self.streak = 0
        self.max_streak = 0
        self.update_score_display()
        self.stacked_widget.setCurrentWidget(self.quiz_page)
        self.next_question()

    def next_question(self):
        """Load the next question."""
        if self.current_round >= self.total_rounds:
            self.show_results()
            return

        self.current_round += 1
        self.current_word = random.choice(self.words)

        # Generate options
        self.options = [self.current_word]
        while len(self.options) < 4:
            wrong = random.choice(self.words)
            if wrong != self.current_word and wrong not in self.options:
                self.options.append(wrong)
        random.shuffle(self.options)

        # Update UI
        for i, btn in enumerate(self.option_buttons):
            btn.setText(self.options[i])
            btn.setEnabled(True)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 18px;
                    padding: 15px;
                    background-color: #6e8efb;
                    color: white;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #5a7df4;
                }
            """)

        # Load demo video
        self.load_demo_video()

        # Start timer (15 seconds per question)
        self.time_left = 15
        self.update_timer_display()

        if not hasattr(self, 'timer'):
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)

        # Update round display
        self.round_label.setText(f"Round: {self.current_round}/{self.total_rounds}")

    def load_demo_video(self):
        """Load the demo video for the current word from Google Drive."""
        try:
            # Find the Avatars folder
            avatars_folder = self.drive_manager.get_folder_id("Avatars")
            if not avatars_folder:
                QMessageBox.warning(self, "Not Found", "Avatars folder not found in Google Drive")
                return False

            # Look for the specific word video (case-insensitive)
            video_name = f"{self.current_word.lower()}.mp4"
            video_files = self.drive_manager.list_files(avatars_folder['id'], mime_type='video/mp4')

            # Find matching video (case-insensitive)
            video_file = next((f for f in video_files if f['name'].lower() == video_name), None)

            # If not found, use hello.mp4 as fallback
            if not video_file:
                video_file = next((f for f in video_files if f['name'].lower() == "hello.mp4"), None)
                if not video_file:
                    QMessageBox.warning(self, "Not Found", f"No demo video found for {self.current_word}")
                    return False

            # Download and play the video
            temp_path = os.path.join(tempfile.gettempdir(), video_file['name'])
            self.drive_manager.download_file(video_file['id'], temp_path)
            self.demo_player.play_video(temp_path)

            # Store the temp path for cleanup
            if not hasattr(self, 'temp_video_files'):
                self.temp_video_files = []
            self.temp_video_files.append(temp_path)

            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load demo video: {str(e)}")
            return False

    def _try_load_specific_video(self):
        """Try to load the specific word video."""
        try:
            avatars_folder = self.drive_manager.get_folder_id("Avatars")
            if not avatars_folder:
                return False

            video_name = f"{self.current_word}.mp4"
            video_files = self.drive_manager.list_files(avatars_folder['id'], mime_type='video/mp4')
            video_file = next((f for f in video_files if f['name'].lower() == video_name.lower()), None)

            if video_file:
                temp_path = os.path.join(tempfile.gettempdir(), video_file['name'])
                self.drive_manager.download_file(video_file['id'], temp_path)
                self.demo_player.play_video(temp_path)

                if not hasattr(self, 'temp_video_files'):
                    self.temp_video_files = []
                self.temp_video_files.append(temp_path)
                return True

        except:
            return False

    def _try_load_fallback_video(self):
        """Try to load the hello.mp4 fallback video."""
        try:
            avatars_folder = self.drive_manager.get_folder_id("Avatars")
            if not avatars_folder:
                return False

            video_files = self.drive_manager.list_files(avatars_folder['id'], mime_type='video/mp4')
            video_file = next((f for f in video_files if f['name'].lower() == "hello.mp4"), None)

            if video_file:
                temp_path = os.path.join(tempfile.gettempdir(), "hello.mp4")
                self.drive_manager.download_file(video_file['id'], temp_path)
                self.demo_player.play_video(temp_path)

                if not hasattr(self, 'temp_video_files'):
                    self.temp_video_files = []
                self.temp_video_files.append(temp_path)
                return True

        except:
            return False

    def cleanup_temp_videos(self):
        """Delete all downloaded demo videos stored during this session."""
        for path in getattr(self, 'temp_video_files', []):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"Failed to delete temp video: {path} → {e}")
        self.temp_video_files.clear()

    def update_timer(self):
        """Update the countdown timer."""
        self.time_left -= 1
        self.update_timer_display()

        if self.time_left <= 0:
            self.timer.stop()
            self.handle_timeout()

    def update_timer_display(self):
        """Update the timer display."""
        self.timer_label.setText(f"Time: {self.time_left}s")

        # Change color when time is running low
        if self.time_left <= 5:
            self.timer_label.setStyleSheet("font-size: 16px; color: red;")
        else:
            self.timer_label.setStyleSheet("font-size: 16px;")

    def update_score_display(self):
        """Update the score and streak displays."""
        self.score_label.setText(f"Score: {self.score}")
        self.streak_label.setText(f"Streak: {self.streak}x")

    def handle_timeout(self):
        """Handle when time runs out for a question."""
        # Disable all buttons
        for btn in self.option_buttons:
            btn.setEnabled(False)

        # Highlight correct answer
        correct_idx = self.options.index(self.current_word)
        self.option_buttons[correct_idx].setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
            }
        """)

        # Show timeout message
        self.question_label.setText("Time's up! The correct answer was highlighted.")

        # Reset streak
        self.streak = 0
        self.update_score_display()

        # Move to next question after delay
        QTimer.singleShot(3000, self.next_question)

    def check_answer(self, selected_idx):
        """Check the selected answer."""
        self.timer.stop()

        # Disable all buttons
        for btn in self.option_buttons:
            btn.setEnabled(False)

        selected_word = self.options[selected_idx]
        is_correct = (selected_word == self.current_word)

        # Calculate points based on time left and streak
        time_bonus = int((self.time_left / 15) * 100)  # Up to 100 points for quick answers
        points = 100 + time_bonus + (self.streak * 20)  # Base 100 + time bonus + streak bonus

        if is_correct:
            self.streak += 1
            self.max_streak = max(self.max_streak, self.streak)
            self.score += points

            # Highlight correct answer in green
            self.option_buttons[selected_idx].setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                }
            """)

            self.question_label.setText(f"✅ Correct! +{points} points (Time bonus: +{time_bonus})")
        else:
            self.streak = 0
            penalty = min(50, self.score)  # Don't go below 0
            self.score = max(0, self.score - penalty)

            # Highlight wrong answer in red and correct in green
            self.option_buttons[selected_idx].setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                }
            """)

            correct_idx = self.options.index(self.current_word)
            self.option_buttons[correct_idx].setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                }
            """)

            self.question_label.setText(f"❌ Incorrect! -{penalty} points")

        self.update_score_display()

        # Move to next question after delay
        QTimer.singleShot(3000, self.next_question)

    def show_results(self):
        """Show final results."""
        self.stacked_widget.setCurrentWidget(self.results_page)

        # Calculate accuracy
        accuracy = (self.score / (100 * self.total_rounds)) * 100 if self.total_rounds > 0 else 0

        # Update results display
        results_text = (
            f"Final Score: {self.score}\n\n"
            f"Accuracy: {accuracy:.1f}%\n"
            f"Highest Streak: {self.max_streak}x\n\n"
        )

        if accuracy >= 80:
            results_text += "🌟 Excellent job! You're a sign language star!"
        elif accuracy >= 60:
            results_text += "👍 Good work! Keep practicing to improve!"
        else:
            results_text += "💪 Keep practicing! You'll get better with time!"

        self.results_text.setText(results_text)

        # Update user stats
        self.user_data['stats']['total_practice_sessions'] += 1
        self.user_data['stats']['total_words_attempted'] += self.total_rounds

        # Count correct answers based on score (approximate)
        estimated_correct = round((accuracy / 100) * self.total_rounds)
        self.user_data['stats']['total_words_correct'] += estimated_correct

        # Update streak if this was done on consecutive days
        today = datetime.now().date()
        last_practice = self.user_data['stats'].get('last_practice_date')

        if last_practice:
            last_date = datetime.fromisoformat(last_practice).date()
            if (today - last_date).days == 1:
                self.user_data['stats']['streak_days'] += 1
            elif (today - last_date).days > 1:
                self.user_data['stats']['streak_days'] = 1
        else:
            self.user_data['stats']['streak_days'] = 1

        self.user_data['stats']['last_practice_date'] = today.isoformat()
        # Update game points
        self.user_data['stats']['game_points']['quiz'] = self.score

        # Save user data through the parent (main window)
        if hasattr(self, 'parent') and self.parent():
            self.parent().save_user_data()
        else:
            # Fallback if parent is not available
            try:
                user_data = {}
                if os.path.exists("users.json"):
                    with open("users.json", 'r') as f:
                        user_data = json.load(f)

                user_data[self.user_data['username']] = self.user_data

                with open("users.json", 'w') as f:
                    json.dump(user_data, f, indent=2)
            except Exception as e:
                print(f"Error saving user data: {e}")

        # Emit signal to update main UI
        self.finished.emit()

    def closeEvent(self, event):
        """Clean up when closing the game."""
        # Clean up temp files
        for temp_file in self.temp_video_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

        # Stop any timers
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()

        # Close MediaPipe holistic
        if hasattr(self, 'holistic'):
            self.holistic.close()

        event.accept()

class AchievementsSystem:
    def __init__(self, user_data):
        self.user_data = user_data
        self.achievements = [
            {
                'id': 'first_sign',
                'name': 'First Sign',
                'description': 'Perform your first sign correctly',
                'icon': '🌟',
                'condition': lambda stats: stats['total_words_correct'] >= 1
            },
            {
                'id': 'perfect_5',
                'name': 'Perfect 5',
                'description': 'Get 5 signs correct in a row',
                'icon': '🏆',
                'condition': lambda stats: stats.get('current_streak', 0) >= 5
            },
            {
                'id': 'daily_user',
                'name': 'Daily User',
                'description': 'Practice for 3 days in a row',
                'icon': '🔥',
                'condition': lambda stats: stats['streak_days'] >= 3
            },
            {
                'id': 'quick_learner',
                'name': 'Quick Learner',
                'description': 'Achieve 80% accuracy in a session',
                'icon': '⚡',
                'condition': lambda stats: (
                        stats['total_words_attempted'] > 0 and
                        (stats['total_words_correct'] / stats['total_words_attempted']) >= 0.8
                )
            }
        ]



class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.login_window = LoginWindow()
        if not self.login_window.exec_():
            sys.exit(0)

        self.current_user = self.login_window.current_user
        self.user_data_file = "users.json"
        # Initialize core attributes
        self.words = ['hello', 'coach', 'clumsy', 'clueless', 'closet', 'close', 'clock', 'climb', 'click', 'clever',
                      'classroom', 'church', 'christ', 'choose', 'choke', 'choice', 'china', 'bible', 'best', 'bell',
                      'behavior',
                      'beginning', 'beautiful', 'battle', 'awkward', 'aware', 'award', 'average', 'available',
                      'authority', 'assume',
                      'assistant', 'assist', 'appropriate', 'alcohol', 'able']
        self.current_word = random.choice(self.words)
        self.current_speed = 1.0
        self.thresholds = {}

        # Initialize word labels
        self.demo_word_label = None
        self.results_word_label = None

        # Video recording variables
        self.is_recording = False
        self.recorded_frames = []
        self.capture = None
        self.recording_timer = QTimer()

        # Initialize Google Drive manager
        self.drive_manager = GoogleDriveManager()

        # Initialize UI
        self.setWindowTitle("Handly - Sign Language Learning")
        self.setGeometry(100, 100, 1200, 800)


        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)

        # Now setup the UI components
        self.setup_ui()
        self.load_thresholds()
        self.update_word_display()
        self.setup_navigation()  # Now main_layout exists
        self.update_score_display()
        self.setup_dictionary_page()

        self.setCentralWidget(self.main_widget)
        self.game_windows = []  # Track open game windows
        # Initialize user progress data
        self.user_progress = {
            'completed_lessons': 0,
            'total_lessons': 10,
            'correct_signs': 0,
            'attempted_signs': 0,
            'streak_days': 0,
            'last_activity': None
        }
        self.recording_attempts = 0
        self.max_attempts = 3

    def reset_camera(self):
        """Completely reset camera state"""
        try:
            # Release existing camera if open
            if hasattr(self, 'capture') and self.capture:
                if self.capture.isOpened():
                    self.capture.release()
                del self.capture

            # Stop any recording timers
            if hasattr(self, 'recording_timer') and self.recording_timer.isActive():
                self.recording_timer.stop()
                del self.recording_timer

            if hasattr(self, 'preview_timer') and self.preview_timer.isActive():
                self.preview_timer.stop()
                del self.preview_timer

            # Clear recorded frames
            if hasattr(self, 'recorded_frames'):
                del self.recorded_frames

            # Reset recording state
            self.is_recording = False
            self.is_paused = False

        except Exception as e:
            print(f"Error resetting camera: {str(e)}")

    def record_user_attempt(self):
        """Handles the recording process with retry logic"""
        try:
            if self.recording_attempts >= self.max_attempts:
                print("Maximum attempts reached. Please try again later.")
                return False

            print("\nStarting recording...")
            # Add your actual recording logic here
            success = self._perform_recording()  # Assume this returns True/False

            if success:
                print("Recording successful!")
                self.recording_attempts = 0  # Reset on success
                return True
            else:
                self.recording_attempts += 1
                remaining = self.max_attempts - self.recording_attempts
                print(f"Recording failed. {remaining} attempts remaining.")
                return False

        except Exception as e:
            print(f"Recording error: {str(e)}")
            return False

    def _perform_recording(self):
        """Actual recording implementation (mock version)"""
        # Replace with your real recording logic
        # This mock version fails randomly for demonstration
        import random
        return random.random() > 0.5  # 50% success rate for testing

    def try_again(self):
        """Reset everything for another attempt with the same word"""
        try:
            # Stop all media playback
            self.stop_all_media_players()

            # Release camera if still open
            if hasattr(self, 'capture') and self.capture and self.capture.isOpened():
                self.capture.release()
                del self.capture

            # Clean up any recording state
            if hasattr(self, 'recorded_frames'):
                del self.recorded_frames
            if hasattr(self, 'is_recording'):
                del self.is_recording
            if hasattr(self, 'is_paused'):
                del self.is_paused

            # Reset UI elements
            self.feedback_text.clear()
            self.feedback_title.clear()
            self.progress_bar.setValue(0)
            self.analysis_progress.hide()
            self.analysis_status.hide()

            # Show recording options again
            self.input_options.show()
            self.record_btn.setEnabled(True)
            self.record_btn.setText("🎥 Record Video")

            # Switch back to demo page
            self.stacked_widget.setCurrentIndex(0)

            # Force UI update
            QApplication.processEvents()

        except Exception as e:
            print(f"Error in try_again: {str(e)}")
            # Fallback - just go back to demo page
            self.stacked_widget.setCurrentIndex(0)
            QMessageBox.warning(self, "Error", "Please try recording again")
    def refresh_progress_dashboard(self):
        """Updates and displays the user's progress dashboard"""
        try:
            # Calculate progress percentage
            progress_percent = (self.user_progress['completed_lessons'] /
                                self.user_progress['total_lessons']) * 100

            # Calculate accuracy rate
            accuracy = (self.user_progress['correct_signs'] /
                        self.user_progress['attempted_signs'] * 100) if self.user_progress['attempted_signs'] > 0 else 0

            # Generate dashboard display
            dashboard = f"""
                    ╔════════════════ SIGN LANGUAGE PROGRESS ═══════════════╗
                    ║                                                        ║
                    ║  Lessons Completed: {self.user_progress['completed_lessons']}/{self.user_progress['total_lessons']} 
                    ║  Progress: [{'■' * int(progress_percent // 10)}{' ' * (10 - int(progress_percent // 10))}] {progress_percent:.1f}% 
                    ║                                                        ║
                    ║  Sign Accuracy: {accuracy:.1f}% ({self.user_progress['correct_signs']}/{self.user_progress['attempted_signs']}) 
                    ║  Current Streak: {self.user_progress['streak_days']} days 
                    ║                                                        ║
                    ║  Last Activity: {self.user_progress['last_activity'] or 'Never'} 
                    ║                                                        ║
                    ╚════════════════════════════════════════════════════════╝
                    """

            print(dashboard)
            return True

        except Exception as e:
            print(f"Error refreshing dashboard: {str(e)}")
            return False

    def create_chart(self, title, data, color="#4a6baf"):
        """Create a simple bar chart widget"""
        chart_widget = QWidget()
        chart_widget.setMinimumHeight(200)
        layout = QVBoxLayout(chart_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Chart title
        title_label = QLabel(title)
        title_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")
        layout.addWidget(title_label)

        # Chart drawing area
        chart_area = QLabel()
        chart_area.setAlignment(Qt.AlignCenter)
        chart_area.setStyleSheet("background-color: white; border: 1px solid #ddd;")

        # Generate chart pixmap
        pixmap = self.draw_bar_chart(data, color, chart_area.size())
        chart_area.setPixmap(pixmap)

        layout.addWidget(chart_area)
        return chart_widget

    def safe_start_practice(self, video_path):
        """Wrapper method to start practice mode with error handling"""
        try:
            # Verify the video exists
            if not os.path.exists(video_path):
                QMessageBox.warning(self, "File Missing",
                                    "The demonstration video could not be found.")
                return

            # Close any existing practice window
            if hasattr(self, 'practice_widget') and self.practice_widget:
                try:
                    self.practice_widget.close()
                except:
                    pass

            # Start new practice session
            self.start_practice_mode(video_path)

        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Could not start practice mode: {str(e)}")
            ##print(f"Error in safe_start_practice: {traceback.format_exc()}")
    def draw_bar_chart(self, data, color, size):
        """Draw a simple bar chart on a pixmap"""
        if not data:
            return QPixmap(size)

        # Create pixmap
        pixmap = QPixmap(size)
        pixmap.fill(Qt.white)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Chart dimensions
        margin = 20
        chart_width = size.width() - 2 * margin
        chart_height = size.height() - 2 * margin
        bar_width = chart_width / (len(data) * 1.5)  # Leave space between bars

        # Find max value for scaling
        max_value = max(data) if max(data) > 0 else 1

        # Draw grid lines
        pen = QPen(QColor(200, 200, 200))
        pen.setWidth(1)
        painter.setPen(pen)

        # Draw bars
        for i, value in enumerate(data):
            # Calculate bar dimensions
            bar_height = (value / max_value) * chart_height
            x = margin + i * (bar_width * 1.5)
            y = size.height() - margin - bar_height

            # Draw bar
            bar_color = QColor(color)
            painter.setBrush(bar_color)
            painter.setPen(Qt.NoPen)
            painter.drawRect(QRectF(x, y, bar_width, bar_height))

            # Draw value label
            painter.setPen(Qt.black)
            painter.drawText(QRectF(x, y - 20, bar_width, 20),
                             Qt.AlignCenter, str(value))

        # Draw axes
        pen = QPen(Qt.black)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(margin, size.height() - margin,
                         size.width() - margin, size.height() - margin)  # X-axis
        painter.drawLine(margin, size.height() - margin,
                         margin, margin)  # Y-axis

        painter.end()
        return pixmap

    def stop_all_media_players(self):
        """Safely stop all media players with error handling"""
        players = [
            getattr(self, 'demo_player', None),
            getattr(self, 'user_player', None),
            getattr(self, 'correct_player', None),
            getattr(self, 'overlay_player', None)
        ]

        for player in players:
            if player is not None and hasattr(player, 'player'):
                try:
                    if player.player.is_playing():
                        player.player.stop()
                    # Release media if exists
                    if hasattr(player.player, 'set_media'):
                        player.player.set_media(None)
                except Exception as e:
                    print(f"Error stopping player: {str(e)}")
    def reset_analysis_state(self):
        """Reset all analysis-related UI elements."""
        if hasattr(self, 'analysis_progress'):
            self.analysis_progress.hide()
            self.analysis_progress.setValue(0)

        if hasattr(self, 'analysis_status'):
            self.analysis_status.hide()
            self.analysis_status.clear()

        if hasattr(self, 'input_options'):
            self.input_options.show()

        if hasattr(self, 'feedback_text'):
            self.feedback_text.clear()

        if hasattr(self, 'feedback_title'):
            self.feedback_title.clear()
    def setup_navigation(self):
        """Setup the navigation bar now that main_layout exists"""
        nav_bar = QHBoxLayout()

        # Home button
        home_btn = QPushButton("🏠 Home")
        home_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        nav_bar.addWidget(home_btn)

        # Games button
        games_btn = QPushButton("🎮 Games")
        games_btn.clicked.connect(self.show_games_menu)
        nav_bar.addWidget(games_btn)

        # Progress button
        progress_btn = QPushButton("📊 Progress")
        progress_btn.clicked.connect(self.show_progress_dashboard)
        nav_bar.addWidget(progress_btn)

        # Insert the navigation bar at the top
        self.main_layout.insertLayout(1, nav_bar)  # After title, before content

    def show_games_menu(self):
        """Show game selection menu."""
        menu = QDialog(self)
        menu.setWindowTitle("Select Game")
        layout = QVBoxLayout(menu)

        # Clock game button
        clock_btn = QPushButton("⏱ Sign Against the Clock")
        clock_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 12px;
                margin: 5px;
            }
        """)
        clock_btn.clicked.connect(lambda: self.start_game_and_close(menu, 'clock'))
        layout.addWidget(clock_btn)

        # Quiz game button
        quiz_btn = QPushButton("❓ Sign Quiz")
        quiz_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 12px;
                margin: 5px;
            }
        """)
        quiz_btn.clicked.connect(lambda: self.start_game_and_close(menu, 'quiz'))
        layout.addWidget(quiz_btn)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(menu.close)
        layout.addWidget(close_btn)

        menu.exec_()

    def start_game_and_close(self, menu, game_type):
        """Start game and close menu."""
        self.start_game(game_type)
        menu.accept()

    def start_game(self, game_type):
        if game_type == 'clock':
            game = SignRecognitionGame(self.words, self.current_user, self)  # Pass self as parent
            game.thresholds = self.thresholds
        elif game_type == 'quiz':
            game = SignQuizGame(self.words, self.current_user, self)  # Pass self as parent
            game.drive_manager = self.drive_manager

        game.finished.connect(self.update_score_display)
        game.finished.connect(lambda: self.game_windows.remove(game))

        # Center and show the window
        game_geometry = game.frameGeometry()
        center_point = self.frameGeometry().center()
        game_geometry.moveCenter(center_point)
        game.move(game_geometry.topLeft())

        self.game_windows.append(game)
        game.show()
        return game

    def show_progress_dashboard(self):
        """Show the progress dashboard page."""
        # Update the dashboard with current data before showing
        self.progress_page = ProgressDashboard(self.current_user)

        # Remove old progress page if it exists
        if hasattr(self, 'progress_page'):
            self.stacked_widget.removeWidget(self.progress_page)

        # Add new progress page
        self.progress_page = ProgressDashboard(self.current_user)
        self.stacked_widget.addWidget(self.progress_page)
        self.stacked_widget.setCurrentWidget(self.progress_page)

    def calculate_accuracy(self):
        """Calculate user's accuracy percentage"""
        stats = self.current_user['stats']
        if stats['total_words_attempted'] > 0:
            return (stats['total_words_correct'] / stats['total_words_attempted']) * 100
        return 0

    def calculate_total_game_points(self):
        """Calculate total game points"""
        return (self.current_user['stats']['game_points'].get('quiz', 0) +
                self.current_user['stats']['game_points'].get('clock', 0))

    def create_stats_card(self, title, content_html):
        """Create a stats card with title and HTML content"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4a6baf;")
        layout.addWidget(title_label)

        # Content
        content = QLabel(content_html)
        content.setStyleSheet("font-size: 14px;")
        content.setWordWrap(True)
        layout.addWidget(content)

        return self.create_card("", widget)

    def create_achievements_card(self):
        """Create achievements card"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Title
        title = QLabel("🏆 Achievements")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #4a6baf;")
        layout.addWidget(title)

        # Achievements list
        achievements_list = QListWidget()
        achievements_list.setStyleSheet("""
            QListWidget {
                border: none;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 8px 0;
                border-bottom: 1px solid #eee;
            }
        """)

        # Sample achievements - replace with real user achievements
        items = [
            ("🌟 First Sign", "Performed your first sign correctly", True),
            ("🔥 3-Day Streak", "Practiced for 3 days in a row",
             self.current_user['stats']['streak_days'] >= 3),
            ("⚡ Fast Learner", "Achieved 80% accuracy",
             self.calculate_accuracy() >= 80),
            ("🏆 Perfect 5", "Got 5 signs correct in a row", False)
        ]

        for icon, text, unlocked in items:
            item = QListWidgetItem(f"{icon} {text}")
            if not unlocked:
                item.setForeground(QColor("#999"))
            achievements_list.addItem(item)

        layout.addWidget(achievements_list)
        return self.create_card("", widget)

    # These methods would be implemented to get real user data:
    def get_accuracy_history(self):
        """Get user's accuracy history (to be implemented)"""
        return None

    def get_words_learned_history(self):
        """Get words learned history (to be implemented)"""
        return None

    def get_activity_history(self):
        """Get activity history (to be implemented)"""
        return None
    def update_score_display(self):
        stats = self.current_user['stats']
        game_points = stats['game_points']
        total_game_points = game_points.get('quiz', 0) + game_points.get('clock', 0)

        score_text = (f"🏆 Points - Learning: {stats['learning_points']} | "
                      f"Games: {total_game_points} (Quiz: {game_points.get('quiz', 0)}, "
                      f"Clock: {game_points.get('clock', 0)})")

        if hasattr(self, 'score_display'):
            self.score_display.setText(score_text)
        else:
            self.score_display = QLabel(score_text)
            self.score_display.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    color: white;
                    background-color: rgba(0, 0, 0, 0.5);
                    padding: 5px 10px;
                    border-radius: 10px;
                }
            """)
            self.title_layout.addWidget(self.score_display, 0, Qt.AlignRight)

    def toggle_fullscreen(self):
        """Toggle between fullscreen and normal mode"""
        if self.fullscreen_btn.isChecked():
            self.showFullScreen()
            self.fullscreen_btn.setText("❎ Exit Fullscreen")
        else:
            self.showNormal()
            self.fullscreen_btn.setText("⛶ Fullscreen")
    def setup_ui(self):
        # Clear any existing layout
        for i in reversed(range(self.main_layout.count())):
            self.main_layout.itemAt(i).widget().setParent(None)

        # Title section
        title_frame = QFrame()
        title_layout = QVBoxLayout(title_frame)
        self.title_layout = title_layout  # Save reference for score display

        title_label = QLabel("👋 Handly")
        title_label.setStyleSheet("font-size: 36px; font-weight: bold;")
        title_layout.addWidget(title_label, 0, Qt.AlignCenter)

        subtitle_label = QLabel("Your friendly sign language learning buddy!")
        subtitle_label.setStyleSheet("font-size: 16px;")
        title_layout.addWidget(subtitle_label, 0, Qt.AlignCenter)

        self.main_layout.addWidget(title_frame)

        # Add the stacked widget for pages
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        # Setup pages
        self.setup_demo_page()
        self.setup_results_page()
        self.setup_progress_page()

    def setup_progress_page(self):
        """Setup the progress dashboard page"""
        self.progress_page = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header
        header = QLabel(f"{self.current_user['username']}'s Progress")
        header.setStyleSheet("""
                font-size: 28px;
                font-weight: bold;
                color: #4a6baf;
                padding-bottom: 10px;
                border-bottom: 2px solid #eee;
            """)
        layout.addWidget(header)

        # Stats cards in a grid
        stats_grid = QGridLayout()
        stats_grid.setSpacing(15)

        # Learning stats card
        learning_stats = self.create_stats_card(
            "📚 Learning",
            f"""
                <p>• <b>Sessions:</b> {self.current_user['stats']['total_practice_sessions']}</p>
                <p>• <b>Accuracy:</b> {self.calculate_accuracy():.1f}%</p>
                <p>• <b>Points:</b> {self.current_user['stats']['learning_points']}</p>
                <p>• <b>Streak:</b> {self.current_user['stats']['streak_days']} days</p>
                """
        )
        stats_grid.addWidget(learning_stats, 0, 0)

        # Game stats card
        game_stats = self.create_stats_card(
            "🎮 Games",
            f"""
                <p>• <b>Total Points:</b> {self.calculate_total_game_points()}</p>
                <p>• <b>Quiz Points:</b> {self.current_user['stats']['game_points'].get('quiz', 0)}</p>
                <p>• <b>Clock Game:</b> {self.current_user['stats']['game_points'].get('clock', 0)}</p>
                """
        )
        stats_grid.addWidget(game_stats, 0, 1)

        # Achievements card
        achievements = self.create_achievements_card()
        stats_grid.addWidget(achievements, 1, 0, 1, 2)

        layout.addLayout(stats_grid)

        # Charts section
        charts_title = QLabel("📈 Progress Over Time")
        charts_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #4a6baf;")
        layout.addWidget(charts_title)

        # Sample charts (using real user data if available)
        charts_grid = QGridLayout()
        charts_grid.setSpacing(15)

        # Accuracy chart - use real data if available
        accuracy_data = self.get_accuracy_history() or [65, 70, 75, 80, 85, 90]
        accuracy_chart = self.create_chart("Accuracy Trend", accuracy_data, "#4a6baf")
        charts_grid.addWidget(self.create_card("Accuracy", accuracy_chart), 0, 0)

        # Words learned chart
        words_data = self.get_words_learned_history() or [5, 10, 15, 20, 25]
        words_chart = self.create_chart("Words Learned", words_data, "#6e8efb")
        charts_grid.addWidget(self.create_card("Vocabulary", words_chart), 0, 1)

        # Activity chart
        activity_data = self.get_activity_history() or [3, 5, 7, 4, 6, 8, 5]
        activity_chart = self.create_chart("Weekly Activity", activity_data, "#a777e3")
        charts_grid.addWidget(self.create_card("Activity", activity_chart), 1, 0, 1, 2)

        layout.addLayout(charts_grid)
        layout.addStretch()

        scroll.setWidget(content)
        self.stacked_widget.addWidget(scroll)

    def setup_demo_page(self):
        """Setup the demonstration and upload page."""
        demo_page = QWidget()
        main_layout = QVBoxLayout(demo_page)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create scroll area with proper resizing
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create the scroll content widget
        scroll_content = QWidget()
        scroll_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(15)

        # Title section
        self.setup_title_section(scroll_layout)

        # Current word section
        self.setup_word_section(scroll_layout)

        # Demo video section
        demo_video_frame = QFrame()
        demo_video_frame.setStyleSheet("""
            background-color: white;
            border-radius: 12px;
            padding: 15px;
        """)
        demo_video_layout = QVBoxLayout(demo_video_frame)

        demo_title = QLabel("🎥 Demonstration Video")
        demo_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        demo_video_layout.addWidget(demo_title)

        self.demo_player = VideoPlayerWidget("")
        demo_video_layout.addWidget(self.demo_player)

        scroll_layout.addWidget(demo_video_frame)

        # Upload section
        upload_frame = QFrame()
        upload_frame.setStyleSheet("""
            background-color: white;
            border-radius: 12px;
            padding: 15px;
        """)
        upload_layout = QVBoxLayout(upload_frame)

        upload_title = QLabel("📤 Your Turn to Sign!")
        upload_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        upload_layout.addWidget(upload_title)

        upload_instructions = QLabel("Record yourself signing the word using the options below")
        upload_instructions.setStyleSheet("font-size: 14px; color: #666;")
        upload_layout.addWidget(upload_instructions)

        # Video input options
        self.input_options = QWidget()
        input_layout = QHBoxLayout(self.input_options)

        # Upload existing video button
        self.upload_btn = QPushButton("📁 Upload Video")
        self.upload_btn.setIcon(QIcon.fromTheme("document-open"))
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.upload_btn.clicked.connect(self.upload_video)
        input_layout.addWidget(self.upload_btn)

        # Record new video button
        self.record_btn = QPushButton("🎥 Record Video")
        self.record_btn.setIcon(QIcon.fromTheme("camera-web"))
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.record_btn.clicked.connect(self.toggle_recording)
        input_layout.addWidget(self.record_btn)

        upload_layout.addWidget(self.input_options, 0, Qt.AlignCenter)

        # Add progress bar (initially hidden)
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setTextVisible(True)
        self.analysis_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #6e8efb;
                border-radius: 5px;
                height: 20px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #6e8efb;
                width: 10px;
            }
        """)
        self.analysis_progress.hide()
        upload_layout.addWidget(self.analysis_progress)

        # Add status label (initially hidden)
        self.analysis_status = QLabel()
        self.analysis_status.setStyleSheet("font-size: 14px; color: #666;")
        self.analysis_status.setAlignment(Qt.AlignCenter)
        self.analysis_status.hide()
        upload_layout.addWidget(self.analysis_status)

        # Recording controls (initially hidden)
        self.recording_controls = QWidget()
        recording_control_layout = QHBoxLayout(self.recording_controls)

        self.recording_preview = QLabel()
        self.recording_preview.setAlignment(Qt.AlignCenter)
        self.recording_preview.setMinimumSize(320, 240)
        self.recording_preview.setStyleSheet("background-color: black;")

        self.start_stop_btn = QPushButton("⏺ Start Recording")
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.start_stop_btn.clicked.connect(self.toggle_recording)

        self.pause_resume_btn = QPushButton("⏸ Pause")
        self.pause_resume_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.pause_resume_btn.clicked.connect(self.toggle_pause)
        self.pause_resume_btn.setEnabled(False)

        self.discard_btn = QPushButton("🗑 Discard")
        self.discard_btn.setStyleSheet("""
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.discard_btn.clicked.connect(self.discard_recording)
        self.discard_btn.setEnabled(False)

        self.save_btn = QPushButton("💾 Save Recording")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.save_btn.clicked.connect(self.save_recording)
        self.save_btn.setEnabled(False)

        control_buttons = QWidget()
        control_buttons_layout = QVBoxLayout(control_buttons)
        control_buttons_layout.addWidget(self.start_stop_btn)
        control_buttons_layout.addWidget(self.pause_resume_btn)
        control_buttons_layout.addWidget(self.discard_btn)
        control_buttons_layout.addWidget(self.save_btn)

        recording_control_layout.addWidget(self.recording_preview)
        recording_control_layout.addWidget(control_buttons)

        self.recording_controls.hide()
        upload_layout.addWidget(self.recording_controls)

        scroll_layout.addWidget(upload_frame)

        # Add stretch to push content up
        scroll_layout.addStretch(1)

        # Set the scroll content
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        self.stacked_widget.addWidget(demo_page)

    def create_card(self, title, content_widget, color="#4a6baf"):
        """Create a modern card component"""
        card = QWidget()
        card.setStyleSheet(f"""
            QWidget {{
                background-color: white;
                border-radius: 12px;
                border-left: 4px solid {color};
            }}
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Card header
        header = QLabel(title)
        header.setStyleSheet(f"""
            font-size: 18px;
            font-weight: bold;
            color: {color};
            padding-bottom: 5px;
            border-bottom: 1px solid #eee;
        """)
        layout.addWidget(header)

        # Card content
        layout.addWidget(content_widget)

        return card
    def toggle_recording(self):

        """Start the recording process with countdown and body detection."""
        # Clean up any previous recording state
        if hasattr(self, 'capture'):
            if self.capture is not None and self.capture.isOpened():
                self.capture.release()
            self.capture = None  # Explicitly set to None after release

        if hasattr(self, 'preview_timer') and self.preview_timer.isActive():
            self.preview_timer.stop()
            del self.preview_timer

        if hasattr(self, 'countdown_timer') and self.countdown_timer.isActive():
            self.countdown_timer.stop()
            del self.countdown_timer

        # Reset all recording-related attributes
        self.is_recording = True
        self.is_paused = False
        self.recorded_frames = []
        self.recording_start_time = None
        self.countdown_finished = False
        self.body_position_ok = False

        # Reinitialize pygame mixer for audio
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        except Exception as e:
            print(f"Error reinitializing audio mixer: {e}")

        # Show instructions
        QMessageBox.information(
            self,
            "Recording Instructions",
            "1. Stand about 2 meters from your camera\n"
            "2. Make sure your whole upper body is visible\n"
            "3. Keep your hands at waist level to start\n"
            "4. The recording will begin automatically after countdown\n"
            "5. Your recording should be up to 3 seconds long"
        )

        # Initialize recording state
        self.is_recording = True
        self.is_paused = False
        self.recorded_frames = []
        self.recording_start_time = None
        self.countdown_finished = False
        self.body_position_ok = False

        # Create full-screen preview window
        self.preview_window = QMainWindow()
        self.preview_window.setWindowTitle("Recording Preview")
        self.preview_window.setWindowState(Qt.WindowFullScreen)
        self.preview_window.setAttribute(Qt.WA_DeleteOnClose)

        # Central widget with black background
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: black;")
        self.preview_window.setCentralWidget(central_widget)

        # Layout for preview and controls
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Video preview label (centered)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label, 1)

        # Instruction label - larger and more prominent
        self.instruction_label = QLabel("Position yourself in frame with hands at waist level")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setStyleSheet("""
            QLabel {
                font-size: 36px;
                color: white;
                background-color: rgba(0, 0, 0, 150);
                padding: 40px;
                border-radius: 20px;
            }
        """)
        layout.addWidget(self.instruction_label, 0, Qt.AlignCenter)

        # Countdown label (centered) - larger and more visible
        self.countdown_label = QLabel("", self.preview_label)
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet("""
            QLabel {
                font-size: 200px;
                color: white;
                background-color: rgba(0, 0, 0, 150);
                border-radius: 50%;
                padding: 50px;
            }
        """)
        self.countdown_label.setFixedSize(300, 300)
        self.countdown_label.hide()  # Hide until countdown starts

        # Setup camera - initialize self.capture here
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            QMessageBox.critical(self, "Camera Error", "Could not access camera")
            self.preview_window.close()
            self.is_recording = False
            self.record_btn.setEnabled(True)
            return

        # Initialize pygame mixer for audio
        pygame.mixer.init()

        # Play initial instruction audio
        self.play_instruction_audio()

        # Start preview timer
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_timer.start(30)

        self.preview_window.show()
    def play_instruction_audio(self):
        """Play audio instructions for positioning."""
        try:
            # Create temporary audio file with instructions
            instruction_text = (
                "Please stand about 2 meters from the camera. "
                "Make sure your whole upper body is visible. "
                "Keep your hands at waist level. "
                "Recording will start automatically when you're positioned correctly."
            )

            tts = gTTS(text=instruction_text, lang='en')
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)

            pygame.mixer.music.load(fp)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Error playing instruction audio: {e}")

    def update_preview(self):
        """Update the camera preview and handle countdown/recording."""
        if not self.capture or not self.capture.isOpened():
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        # Mirror the frame for more natural feel
        frame = cv2.flip(frame, 1)

        # Convert to QImage for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale to fit while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(pixmap)

        # Center the countdown label
        self.countdown_label.move(
            self.preview_label.width() // 2 - 150,
            self.preview_label.height() // 2 - 150
        )

        # Body detection (only if countdown hasn't started)
        if not hasattr(self, 'countdown_started') or not self.countdown_started:
            with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
                results = holistic.process(rgb_frame)

                # Check if body is properly framed
                body_visible = False
                hands_at_waist = False

                if results.pose_landmarks:
                    # Draw landmarks for user feedback
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                    # Check if shoulders and hips are visible
                    required_landmarks = [
                        mp_holistic.PoseLandmark.LEFT_SHOULDER,
                        mp_holistic.PoseLandmark.RIGHT_SHOULDER,
                        mp_holistic.PoseLandmark.LEFT_HIP,
                        mp_holistic.PoseLandmark.RIGHT_HIP
                    ]

                    body_visible = all(
                        results.pose_landmarks.landmark[lm].visibility > 0.5
                        for lm in required_landmarks
                    )

                    # Check if hands are at waist level (between hips and shoulders)
                    if body_visible:
                        left_shoulder = results.pose_landmarks.landmark[
                            mp_holistic.PoseLandmark.LEFT_SHOULDER]
                        right_hip = results.pose_landmarks.landmark[
                            mp_holistic.PoseLandmark.RIGHT_HIP]

                        waist_level = (left_shoulder.y + right_hip.y) / 2

                        if results.left_hand_landmarks:
                            left_hand = results.left_hand_landmarks.landmark[0]  # Wrist
                            hands_at_waist = abs(left_hand.y - waist_level) < 0.1

                        if results.right_hand_landmarks and not hands_at_waist:
                            right_hand = results.right_hand_landmarks.landmark[0]  # Wrist
                            hands_at_waist = abs(right_hand.y - waist_level) < 0.1

                self.body_position_ok = body_visible and hands_at_waist

                # Update instruction text based on positioning
                if not body_visible:
                    self.instruction_label.setText("⚠️ Move back - make sure your whole upper body is visible")
                    self.instruction_label.setStyleSheet("color: #FFA500; font-size: 36px;")  # Orange
                elif not hands_at_waist:
                    self.instruction_label.setText("✋ Please place your hands at waist level to start")
                    self.instruction_label.setStyleSheet("color: #FFFF00; font-size: 36px;")  # Yellow
                else:
                    self.instruction_label.setText("✓ Ready! Recording will start automatically...")
                    self.instruction_label.setStyleSheet("color: #00FF00; font-size: 36px;")  # Green

                    # Start countdown if body is properly positioned
                    if not hasattr(self, 'countdown_started'):
                        self.countdown_started = True
                        self.start_countdown()

        # If recording (after countdown), save frames
        if (self.is_recording and not self.is_paused and
                hasattr(self, 'recording_started') and self.recording_started):
            self.recorded_frames.append(frame)

            # Update timer
            elapsed = time.time() - self.recording_start_time
            self.timer_label.setText(f"{min(3, elapsed):.1f}s")

            # Auto-stop after 3 seconds
            if elapsed >= 3.0:
                self.stop_recording()

    def start_countdown(self):
        """Start the 3-2-1 countdown before recording begins with audio."""
        if not hasattr(self, 'countdown'):
            self.countdown = 3
        else:
            self.countdown = 3  # Reset countdown

        self.countdown_label.setText(str(self.countdown))
        self.countdown_label.show()

        # Play initial sound
        self.play_countdown_sound(self.countdown)

        # Start countdown timer
        if hasattr(self, 'countdown_timer'):
            self.countdown_timer.stop()
            del self.countdown_timer

        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)  # 1 second intervals

    def play_countdown_sound(self, number):
        """Play countdown sound effect."""
        try:
            if number > 0:
                text = str(number)
            else:
                text = "Go!"

            tts = gTTS(text=text, lang='en')
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)

            # Use a sound effect for better feedback
            sound = pygame.mixer.Sound(fp)
            sound.play()
        except Exception as e:
            print(f"Error playing countdown sound: {e}")

    def update_countdown(self):
        """Update the countdown and start recording when done."""
        self.countdown -= 1

        if self.countdown > 0:
            # Update countdown display
            self.countdown_label.setText(str(self.countdown))
            self.play_countdown_sound(self.countdown)
        else:
            # Countdown finished - start recording
            self.countdown_timer.stop()
            self.countdown_label.setText("GO!")
            self.play_countdown_sound(0)  # Play "Go!" sound

            # Hide instruction label
            self.instruction_label.hide()

            # Add recording timer label
            self.timer_label = QLabel("00:00", self.preview_label)
            self.timer_label.setAlignment(Qt.AlignCenter)
            self.timer_label.setStyleSheet("""
                QLabel {
                    font-size: 24px;
                    color: white;
                    background-color: rgba(0, 0, 0, 150);
                    padding: 10px 20px;
                    border-radius: 8px;
                }
            """)
            self.timer_label.setFixedSize(150, 50)
            self.timer_label.move(20, 20)
            self.timer_label.show()

            # Add recording indicator
            self.recording_indicator = QLabel("● REC", self.preview_label)
            self.recording_indicator.setStyleSheet("""
                QLabel {
                    font-size: 24px;
                    color: red;
                    background-color: rgba(0, 0, 0, 150);
                    padding: 10px;
                    border-radius: 8px;
                }
            """)
            self.recording_indicator.move(self.preview_label.width() - 120, 20)
            self.recording_indicator.show()

            # Mark recording as started and record start time
            self.recording_started = True
            self.recording_start_time = time.time()

            # Hide countdown after brief delay
            QTimer.singleShot(500, lambda: self.countdown_label.hide())



    def capture_frame(self):
        """Capture a frame from the camera during recording."""
        ret, frame = self.capture.read()
        if ret:
            # Store frame
            self.recorded_frames.append(frame)

            # Show preview (mirrored for more natural feel)
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.preview_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

    def stop_recording(self):
        """Stop recording and clean up."""
        # Stop all timers
        if hasattr(self, 'recording_timer') and self.recording_timer.isActive():
            self.recording_timer.stop()
            del self.recording_timer

        if hasattr(self, 'preview_timer') and self.preview_timer.isActive():
            self.preview_timer.stop()
            del self.preview_timer

        if hasattr(self, 'countdown_timer') and self.countdown_timer.isActive():
            self.countdown_timer.stop()
            del self.countdown_timer

        # Release camera
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()
            del self.capture

        # Reset state
        self.is_recording = False
        self.record_btn.setEnabled(True)
        self.record_btn.setText("🎥 Record Video")

        # Clean up any recording-related attributes
        for attr in ['recording_started', 'recording_start_time',
                     'countdown_started', 'countdown_value']:
            if hasattr(self, attr):
                delattr(self, attr)

        # Close preview window if it exists
        if hasattr(self, 'preview_window'):
            self.preview_window.close()
            del self.preview_window

        # If we have recorded frames, show review dialog
        if len(self.recorded_frames) > 0:
            self.review_recording()
        else:
            self.cleanup_recording()

    def cleanup_recording(self):
        """Clean up recorded frames."""
        self.recorded_frames = []

    def use_recorded_video(self, video_path):
        """Use the recorded video for analysis."""
        try:
            # Close the preview window if it exists
            if hasattr(self, 'preview_window'):
                self.preview_window.close()
                del self.preview_window

            # Save to permanent location
            final_path = os.path.join(tempfile.gettempdir(), f"user_recording_{int(time.time())}.mp4")
            shutil.copy(video_path, final_path)

            # Clean up
            self.cleanup_recording()

            # Analyze the video (same as uploaded video)
            self.analyze_video(final_path)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process recording: {str(e)}")
    def review_recording(self):
        """Show dialog to review and save/discard recording."""
        # Create review dialog
        review_dialog = QDialog(self)
        review_dialog.setWindowTitle("Review Your Recording")
        review_dialog.setMinimumSize(800, 600)

        layout = QVBoxLayout(review_dialog)

        # Video player
        video_player = VideoPlayerWidget("Your Recording")

        # Save recording to temp file for preview
        temp_file = os.path.join(tempfile.gettempdir(), f"review_{int(time.time())}.mp4")
        height, width, _ = self.recorded_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_file, fourcc, 30.0, (width, height))

        for frame in self.recorded_frames:
            out.write(frame)
        out.release()

        video_player.play_video(temp_file)
        layout.addWidget(video_player)

        # Button controls
        button_box = QHBoxLayout()

        retry_btn = QPushButton("🔄 Retry")
        retry_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 10px 20px;
            }
        """)
        retry_btn.clicked.connect(lambda: (review_dialog.close(), self.toggle_recording()))

        discard_btn = QPushButton("🗑 Discard")
        discard_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 10px 20px;
                background-color: #f44336;
                color: white;
            }
        """)
        discard_btn.clicked.connect(lambda: (review_dialog.close(), self.cleanup_recording()))

        use_btn = QPushButton("✅ Use This Video")
        use_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
            }
        """)
        use_btn.clicked.connect(lambda: (review_dialog.close(), self.use_recorded_video(temp_file)))

        button_box.addWidget(retry_btn)
        button_box.addWidget(discard_btn)
        button_box.addWidget(use_btn)
        layout.addLayout(button_box)

        review_dialog.exec_()

    def save_recording(self):
        """Save the recorded video and analyze it."""
        if len(self.recorded_frames) == 0:
            return

        # Save to permanent temp file
        temp_file = os.path.join(tempfile.gettempdir(), f"user_recording_{int(time.time())}.mp4")
        height, width, _ = self.recorded_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_file, fourcc, 30.0, (width, height))

        for frame in self.recorded_frames:
            out.write(frame)
        out.release()

        # Clean up review temp file
        if hasattr(self, 'temp_review_file') and os.path.exists(self.temp_review_file):
            os.remove(self.temp_review_file)

        # Analyze the video
        self.analyze_video(temp_file)

        # Clean up recorded frames
        self.recorded_frames = []


    def toggle_pause(self):
        """Pause or resume recording."""
        self.is_paused = not self.is_paused

        if self.is_paused:
            self.pause_btn.setText("▶ Resume")
            self.pause_btn.setStyleSheet("""
                QPushButton {
                    font-size: 24px;
                    color: white;
                    background-color: #4CAF50;
                    padding: 15px 30px;
                    border-radius: 8px;
                }
            """)
        else:
            self.pause_btn.setText("⏸ Pause")
            self.pause_btn.setStyleSheet("""
                QPushButton {
                    font-size: 24px;
                    color: white;
                    background-color: #FF9800;
                    padding: 15px 30px;
                    border-radius: 8px;
                }
            """)

    def discard_recording(self):
        """Discard the current recording."""
        reply = QMessageBox.question(
            self,
            "Discard Recording",
            "Are you sure you want to discard this recording?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Close the preview window if it exists
            if hasattr(self, 'preview_window'):
                self.preview_window.close()
                del self.preview_window

            self.stop_recording()
            self.recorded_frames = []
            self.recording_controls.hide()

    def setup_results_page(self):
        """Setup the results page."""
        results_page = QWidget()
        main_layout = QVBoxLayout(results_page)

        # Create a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(10, 10, 10, 10)  # Add margins
        scroll_layout.setSpacing(15)

        # Title section
        self.setup_title_section(scroll_layout)

        # Current word section
        self.setup_word_section(scroll_layout)

        # Feedback section
        feedback_frame = QFrame()
        feedback_frame.setStyleSheet("""
            background-color: white;
            border-radius: 12px;
            padding: 15px;
        """)
        feedback_layout = QVBoxLayout(feedback_frame)

        self.feedback_title = QLabel()
        self.feedback_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        feedback_layout.addWidget(self.feedback_title)

        self.feedback_text = QTextEdit()
        self.feedback_text.setReadOnly(True)
        feedback_layout.addWidget(self.feedback_text)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #6e8efb;
                border-radius: 5px;
                height: 20px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #6e8efb;
                width: 10px;
            }
        """)
        feedback_layout.addWidget(self.progress_bar)

        # Add practice button container
        self.practice_btn_container = QWidget()
        practice_btn_layout = QHBoxLayout(self.practice_btn_container)
        practice_btn_layout.setContentsMargins(0, 10, 0, 0)
        feedback_layout.addWidget(self.practice_btn_container)

        scroll_layout.addWidget(feedback_frame)

        # Video comparison section
        comparison_frame = QFrame()
        comparison_frame.setStyleSheet("""
            background-color: white;
            border-radius: 12px;
            padding: 15px;
        """)
        comparison_layout = QVBoxLayout(comparison_frame)

        comparison_title = QLabel("🎬 Video Comparison")
        comparison_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        comparison_layout.addWidget(comparison_title)

        # Splitter for side-by-side videos
        video_splitter = QSplitter(Qt.Horizontal)

        self.user_player = VideoPlayerWidget("Your Signing")
        video_splitter.addWidget(self.user_player)

        self.correct_player = VideoPlayerWidget("Correct Example")
        video_splitter.addWidget(self.correct_player)

        video_splitter.setSizes([600, 600])
        comparison_layout.addWidget(video_splitter)

        scroll_layout.addWidget(comparison_frame)

        # Overlay video section
        overlay_frame = QFrame()
        overlay_frame.setStyleSheet("""
            background-color: white;
            border-radius: 12px;
            padding: 15px;
        """)
        overlay_layout = QVBoxLayout(overlay_frame)

        overlay_title = QLabel("👀 Detailed Comparison")
        overlay_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        overlay_layout.addWidget(overlay_title)

        overlay_instructions = QLabel("Red highlights show differences from correct sign")
        overlay_instructions.setStyleSheet("font-size: 14px; color: #666;")
        overlay_layout.addWidget(overlay_instructions)

        self.overlay_player = VideoPlayerWidget("Overlay Comparison")
        overlay_layout.addWidget(self.overlay_player)

        scroll_layout.addWidget(overlay_frame)

        # Action buttons
        action_frame = QFrame()
        action_layout = QHBoxLayout(action_frame)

        self.try_again_btn = QPushButton("🔄 Try Again")
        self.try_again_btn.clicked.connect(self.try_again)

        self.new_word_btn = QPushButton("✨ New Word")
        self.new_word_btn.clicked.connect(self.new_word)

        action_layout.addWidget(self.try_again_btn)
        action_layout.addWidget(self.new_word_btn)

        scroll_layout.addWidget(action_frame)

        # Add stretch to push content up
        scroll_layout.addStretch(1)

        # Set the scroll content
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        self.stacked_widget.addWidget(results_page)

    def setup_title_section(self, parent_layout):
        header = QWidget()
        header.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                        stop:0 #4a6baf, stop:1 #6e8efb);
            border-radius: 12px;
        """)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(20, 20, 20, 20)

        # Main title row
        title_row = QHBoxLayout()

        # App icon and title
        title_container = QHBoxLayout()
        icon_label = QLabel()
        icon_pixmap = QPixmap(":/icons/app-icon.png").scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        icon_label.setPixmap(icon_pixmap)
        title_container.addWidget(icon_label)

        title_text = QLabel("Handly")
        title_text.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: white;
            margin-left: 10px;
        """)
        title_container.addWidget(title_text)
        title_row.addLayout(title_container)

        # User info and score
        user_container = QHBoxLayout()

        # User avatar
        avatar_label = QLabel()
        avatar_pixmap = QPixmap(":/icons/user-avatar.png").scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        avatar_label.setPixmap(avatar_pixmap)
        user_container.addWidget(avatar_label)

        # User name and score
        user_info = QVBoxLayout()
        self.username_label = QLabel(self.current_user['username'])
        self.username_label.setStyleSheet("font-size: 14px; color: white; font-weight: bold;")

        self.score_label = QLabel("⭐ 1,250 points")
        self.score_label.setStyleSheet("font-size: 12px; color: white;")

        user_info.addWidget(self.username_label)
        user_info.addWidget(self.score_label)
        user_container.addLayout(user_info)

        title_row.addLayout(user_container)
        header_layout.addLayout(title_row)

        # Navigation bar
        nav_bar = QHBoxLayout()
        nav_bar.setSpacing(10)

        # In your setup_title_section method, change the buttons list to:
        buttons = [
            ("🏠 Home", lambda: self.stacked_widget.setCurrentIndex(0)),
            ("🎮 Games", self.show_games_menu),
            ("📊 Progress", self.show_progress_dashboard)
        ]

        for text, callback in buttons:
            btn = QPushButton(text)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255,255,255,0.2);
                    color: white;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: rgba(255,255,255,0.3);
                }
            """)
            btn.clicked.connect(callback)
            nav_bar.addWidget(btn)

        header_layout.addLayout(nav_bar)
        parent_layout.addWidget(header)

    def show_dictionary(self):
        """Show the sign language dictionary"""
        if not hasattr(self, 'dictionary_page'):
            self.setup_dictionary_page()
        self.stacked_widget.setCurrentWidget(self.dictionary_page)

    def setup_dictionary_page(self):
        """Setup the dictionary page"""
        self.dictionary_page = QWidget()
        layout = QVBoxLayout(self.dictionary_page)

        # Title
        title = QLabel("Sign Language Dictionary")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #4a6baf;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        # Search bar
        search_bar = QLineEdit()
        search_bar.setPlaceholderText("Search for a sign...")
        search_bar.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 20px;
                font-size: 16px;
            }
        """)
        layout.addWidget(search_bar)

        # Word list
        word_list = QListWidget()
        word_list.addItems(sorted(self.words))
        word_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 8px;
                font-size: 16px;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #4a6baf;
                color: white;
            }
        """)
        word_list.itemClicked.connect(self.show_word_in_dictionary)
        layout.addWidget(word_list)

        self.stacked_widget.addWidget(self.dictionary_page)

    def show_word_in_dictionary(self, item):
        """Show details for a selected word"""
        word = item.text()
        # Implement word details display here
        print(f"Selected word: {word}")  # Replace with actual implementation
    def safe_new_word(self):
        """Wrapper method for new_word that handles exceptions."""
        try:
            # Disable button during operation to prevent rapid clicks
            self.new_word_btn.setEnabled(False)
            QApplication.processEvents()

            self.new_word()

        except Exception as e:
            print(f"Error in safe_new_word: {str(e)}")
            # Fallback to known good state
            self.current_word = "hello"
            self.update_word_display()
            self.stacked_widget.setCurrentIndex(0)

        finally:
            # Re-enable button after operation
            self.new_word_btn.setEnabled(True)
    def setup_word_section(self, parent_layout):
        word_frame = QFrame()
        word_frame.setStyleSheet("""
            background-color: white;
            border-radius: 12px;
            border-left: 5px solid #6e8efb;
            padding: 15px;
        """)
        word_layout = QHBoxLayout(word_frame)
        word_layout.setContentsMargins(10, 10, 10, 10)

        # Create the word label
        word_label = QLabel(f"📝 Current Sign: {self.current_word}")
        word_label.setStyleSheet("font-size: 20px;")
        word_layout.addWidget(word_label, 1)

        # Store reference based on which page we're setting up
        if self.stacked_widget.currentIndex() == 0:  # Demo page
            self.demo_word_label = word_label
        else:  # Results page
            self.results_word_label = word_label

        new_word_btn = QPushButton("🔀 Random Word")
        new_word_btn.setIcon(QIcon.fromTheme("media-skip-forward"))
        new_word_btn.clicked.connect(self.safe_new_word)
        word_layout.addWidget(new_word_btn)
        parent_layout.addWidget(word_frame)

    def start_practice_mode(self, correct_video_path):
        """Start the frame-by-frame practice mode with proper initialization"""
        try:
            # Ensure the video path exists
            if not os.path.exists(correct_video_path):
                QMessageBox.warning(self, "File Not Found",
                                    "The demonstration video could not be found.")
                return

            # Create and show the practice widget
            self.practice_widget = PracticeWidget(correct_video_path, self)

            # Center the window on the main app
            practice_geometry = self.practice_widget.frameGeometry()
            center_point = self.frameGeometry().center()
            practice_geometry.moveCenter(center_point)
            self.practice_widget.move(practice_geometry.topLeft())

            self.practice_widget.show()

        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Could not start practice mode: {str(e)}")
            #print(f"Error starting practice mode: {traceback.format_exc()}")

    def new_word(self):
        """Select a new random word safely and update the UI."""
        try:
            # Stop all media players and clean up first
            self.stop_all_media_players()
            self.cleanup_temp_files()

            # Get current words list excluding the current word
            available_words = [w for w in self.words if w != self.current_word]
            if not available_words:
                available_words = self.words  # Fallback if empty

            # Select new word (with retry logic)
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    new_word = random.choice(available_words)
                    self.current_word = new_word

                    # Update UI
                    self.update_word_display()

                    # Reset analysis state
                    self.reset_analysis_state()

                    # Ensure we're on the demo page
                    self.stacked_widget.setCurrentIndex(0)

                    # Force UI update
                    QApplication.processEvents()

                    # Success - exit the retry loop
                    break

                except Exception as e:
                    if attempt == max_attempts - 1:  # Last attempt failed
                        raise
                    print(f"Attempt {attempt + 1} failed, retrying... Error: {str(e)}")
                    continue

        except Exception as e:
            print(f"Failed to switch to new word: {str(e)}")
            # Fallback to a known good word
            self.current_word = "hello"
            self.update_word_display()
            self.stacked_widget.setCurrentIndex(0)
            QMessageBox.warning(self, "Error",
                                "Could not load a new word. Defaulting to 'hello'.")

    def try_again(self):
        """Let the user try the same word again."""
        self.cleanup_temp_files()
        self.feedback_text.clear()
        self.progress_bar.setValue(0)
        self.user_player.media_player.stop()
        self.correct_player.media_player.stop()
        self.overlay_player.media_player.stop()
        self.stacked_widget.setCurrentIndex(0)  # Switch back to demo page

    def download_and_save_thresholds_to_json(self, output_path="thresholds_export.json"):
        """Download all *_threshold.txt files from Drive and save them as a single JSON file."""

        thresholds = {}

        try:
            # מציאת תיקיית thresholds
            thresholds_folder = self.drive_manager.get_folder_id("thresholds")
            if not thresholds_folder:
                print("❌ 'thresholds' folder not found in Google Drive.")
                return

            # קבלת כל הקבצים בתיקייה
            threshold_files = self.drive_manager.list_files(thresholds_folder['id'])
            print(f"📂 Found {len(threshold_files)} files in 'thresholds' folder")

            for file in threshold_files:
                name = file['name']
                if not name.endswith("_threshold.txt"):
                    print(f"⏭ Skipping unrelated file: {name}")
                    continue

                word = name.replace("_threshold.txt", "")
                temp_path = os.path.join(tempfile.gettempdir(), name)

                try:
                    self.drive_manager.download_file(file['id'], temp_path)
                    with open(temp_path, "r") as f:
                        value = float(f.read().strip())
                        thresholds[word] = value
                        print(f"✅ Loaded threshold for '{word}': {value:.4f}")
                except Exception as e:
                    print(f"⚠️ Could not process '{name}': {str(e)}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            # שמירה לקובץ JSON
            with open(output_path, "w") as out_file:
                json.dump(thresholds, out_file, indent=2)
                print(f"📦 Saved all thresholds to JSON file: {output_path}")

        except Exception as e:
            print(f"💥 Error downloading and saving thresholds: {str(e)}")
    def load_thresholds(self, use_cache=True):
        """Load thresholds from a local JSON cache or from Google Drive."""
        # self.download_and_save_thresholds_to_json("/Users/yaelbatat/Desktop/pythonProject/final_project/my_thresholds.json")
        self.thresholds = {}
        cache_path = r"C:\Users\shirb\PycharmProjects\FInal_Project_new\my_thresholds.json"

        # 🚀 Try to load from cache first
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    self.thresholds = json.load(f)
                    print(f"✅ Loaded {len(self.thresholds)} thresholds from local cache")
                    return
            except Exception as e:
                print(f"⚠️ Failed to load local cache: {str(e)} — fallback to Drive")

        try:
            # ⏬ Fallback: Load from Google Drive
            thresholds_folder = self.drive_manager.get_folder_id("thresholds")
            if not thresholds_folder:
                print("❌ 'thresholds' folder not found in Google Drive.")
                return

            threshold_files = self.drive_manager.list_files(thresholds_folder['id'])

            for file in threshold_files:
                name = file['name']
                if not name.endswith('_threshold.txt'):
                    continue  # Skip unrelated files

                word = name.replace('_threshold.txt', '')
                temp_path = os.path.join(tempfile.gettempdir(), name)

                try:
                    self.drive_manager.download_file(file['id'], temp_path)
                    with open(temp_path, 'r') as f:
                        value = float(f.read().strip())
                        self.thresholds[word] = value
                except Exception as e:
                    print(f"⚠️ Could not load threshold for '{word}': {str(e)}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            print(f"✅ Loaded {len(self.thresholds)} thresholds from Drive.")

            # 💾 Save to local JSON cache
            try:
                with open(cache_path, "w") as f:
                    json.dump(self.thresholds, f)
                    print("📦 Saved thresholds to local cache.")
            except Exception as e:
                print(f"⚠️ Could not save cache: {str(e)}")

        except Exception as e:
            print(f"💥 Error loading thresholds: {str(e)}")

    def update_word_display(self):
        """Update the UI with the current word and load its demo video safely."""
        try:
            # Update word labels
            if self.demo_word_label is not None:
                self.demo_word_label.setText(f"📝 Current Sign: {self.current_word}")
            if self.results_word_label is not None:
                self.results_word_label.setText(f"📝 Current Sign: {self.current_word}")

            # Load demo video with retry
            if not self.load_demo_video():
                # If loading failed, try the fallback
                if not self._try_load_fallback_video():
                    QMessageBox.warning(self, "Video Unavailable",
                                        f"Could not load demo for '{self.current_word}'")

        except Exception as e:
            print(f"Error updating word display: {str(e)}")
    def load_demo_video(self):
        """Load the demo video for the current word from Google Drive."""
        try:
            # Find the Avatars folder
            avatars_folder = self.drive_manager.get_folder_id("Avatars")
            if not avatars_folder:
                QMessageBox.warning(self, "Not Found", "Avatars folder not found in Google Drive")
                return False

            # Look for the specific word video
            video_name = f"{self.current_word}.mp4"
            video_files = self.drive_manager.list_files(avatars_folder['id'], mime_type='video/mp4')

            # Find the video matching our current word
            video_file = next((f for f in video_files if f['name'].lower() == video_name.lower()), None)

            if not video_file:
                # Fallback to Hello.mp4 if specific word video not found
                video_file = next((f for f in video_files if f['name'].lower() == "hello.mp4"), None)
                if not video_file:
                    QMessageBox.warning(self, "Not Found", f"No demo video found for {self.current_word}")
                    return False

            temp_path = os.path.join(tempfile.gettempdir(), video_file['name'])

            # Download and play the video
            self.drive_manager.download_file(video_file['id'], temp_path)
            self.demo_player.play_video(temp_path)

            # Store the temp path for cleanup
            if not hasattr(self, 'temp_video_files'):
                self.temp_video_files = []
            self.temp_video_files.append(temp_path)

            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load demo video: {str(e)}")
            return False

    def upload_video(self):
        """Handle video upload from the user."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select your sign video", "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )

        if file_path:
            self.analyze_video(file_path)
    def show_results(self, results):
        """Display the analysis results to the user."""
        # Hide progress bar and status
        self.analysis_progress.hide()
        self.analysis_status.hide()
        self.input_options.show()

        # Calculate accuracy percentage
        accuracy = results['confidence_score'] * 100

        # Switch to results page
        self.stacked_widget.setCurrentIndex(1)

        # Clear any existing practice button
        for i in reversed(range(self.practice_btn_container.layout().count())):
            self.practice_btn_container.layout().itemAt(i).widget().setParent(None)

        if results['prediction'] == 'Correct':
            self.feedback_title.setText("✅ Great job! Your sign looks correct!")
            feedback_text = f"\nAccuracy: {accuracy:.1f}%\n\nKeep up the good work!"
            self.feedback_text.setPlainText(feedback_text)

            # Only show user video for correct results
            self.user_player.play_video(results['user_video_path'])
            self.correct_player.hide()
            self.overlay_player.hide()
        else:
            self.feedback_title.setText("❌ Your sign needs improvement")
            feedback_text = f"\nAccuracy: {accuracy:.1f}%\n\nSuggestions:\n"
            feedback_text += "- Pay attention to hand shape\n"
            feedback_text += "- Watch the timing of movements\n"
            feedback_text += "- Compare with the demo video"
            self.feedback_text.setPlainText(feedback_text)

            # Add practice button
            practice_btn = QPushButton("🔄 Practice This Sign")
            practice_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6e8efb;
                    color: white;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #5a7df4;
                }
            """)
            practice_btn.clicked.connect(lambda: self.start_practice_mode(results['closest_correct_example']))
            self.practice_btn_container.layout().addWidget(practice_btn, 0, Qt.AlignCenter)

            # Show both videos and overlay for incorrect results
            self.user_player.play_video(results['user_video_path'])
            self.correct_player.play_video(results['closest_correct_example'])
            self.overlay_player.play_video(results['comparison_path'])
            self.correct_player.show()
            self.overlay_player.show()

    def setup_word_directories(self):
        """Create local directories for the current word."""
        base_path = os.path.join(tempfile.gettempdir(), "FinalProject", "videos", self.current_word)
        directories = {
            "correct": os.path.join(base_path, "Correct"),
            "user_input": os.path.join(base_path, "User_input"),
            "aligned": os.path.join(base_path, "Aligned")
        }

        for dir_path in directories.values():
            os.makedirs(dir_path, exist_ok=True)

        return directories

    def align_and_scale_video(self, input_path, output_path, target_size=(800, 600)):
        """Align and scale video with proper error handling while maintaining larger size"""
        try:
            # Verify input path exists
            if not isinstance(input_path, str) or not os.path.exists(input_path):
                raise ValueError(f"Invalid video path: {input_path}")

            # Initialize video capture
            cap = cv2.VideoCapture()
            if not cap.open(input_path):
                raise ValueError(f"Could not open video: {input_path}")

            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Initialize video writer with larger target size
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
            if not out.isOpened():
                raise ValueError(f"Could not create output video: {output_path}")

            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                # First frame processing for alignment parameters
                ret, frame = cap.read()
                if not ret:
                    raise ValueError("Could not read first frame")

                # Process frame and get landmarks
                results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not results.pose_landmarks:
                    raise ValueError("No pose detected in first frame")

                # Calculate alignment parameters based on shoulders and hips
                landmarks = results.pose_landmarks.landmark
                left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[mp_holistic.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_holistic.PoseLandmark.RIGHT_HIP]

                # Calculate scale based on torso size (shoulders to hips)
                shoulder_width = abs(right_shoulder.x - left_shoulder.x) * frame_width
                torso_height = abs((left_shoulder.y + right_shoulder.y) / 2 -
                                   (left_hip.y + right_hip.y) / 2) * frame_height

                # Target proportions - we'll make the torso take up more of the frame
                TARGET_TORSO_HEIGHT = 0.5 * target_size[1]  # Torso will be 50% of frame height
                scale = TARGET_TORSO_HEIGHT / torso_height

                # Calculate center point between shoulders and hips
                center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
                center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4

                # Calculate translation to center the torso
                tx = target_size[0] / 2 - center_x * frame_width * scale
                ty = target_size[1] / 2 - center_y * frame_height * scale + 140

                # Create transformation matrix
                M = np.float32([
                    [scale, 0, tx],
                    [0, scale, ty]
                ])

                # Process all frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Apply transformation
                    aligned_frame = cv2.warpAffine(frame, M, target_size, flags=cv2.INTER_LINEAR)

                    # If the aligned frame is smaller than target, pad it
                    if aligned_frame.shape[0] < target_size[1] or aligned_frame.shape[1] < target_size[0]:
                        padded_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                        y_offset = (target_size[1] - aligned_frame.shape[0]) // 2
                        x_offset = (target_size[0] - aligned_frame.shape[1]) // 2
                        padded_frame[y_offset:y_offset + aligned_frame.shape[0],
                        x_offset:x_offset + aligned_frame.shape[1]] = aligned_frame
                        aligned_frame = padded_frame

                    out.write(aligned_frame)

        except Exception as e:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if 'out' in locals() and out.isOpened():
                out.release()
            raise

        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if 'out' in locals() and out.isOpened():
                out.release()

    def video_to_pose(self, video_path, output_pose_path):
        """Convert a video to pose format using command line."""
        cmd = f"video_to_pose -i {video_path} --format mediapipe -o {output_pose_path}"
        ret = os.system(cmd)
        if ret != 0:
            raise ValueError(f"Pose conversion failed (return code {ret})")

    def read_pose(self, pose_path):
        """Read a pose file."""
        with open(pose_path, "rb") as f:
            return Pose.read(f.read())

    def embed_pose(self, pose, model_name='default'):
        """Generate embeddings for a pose."""
        poses = [pose] if not isinstance(pose, list) else pose
        poses = [p if isinstance(p, Pose) else self.read_pose(p) for p in poses]

        pose_data = []
        for pose in poses:
            memory_file = io.BytesIO()
            pose.write(memory_file)
            encoded = base64.b64encode(memory_file.getvalue())
            pose_data.append(encoded.decode('ascii'))

        embeddings = self.query_embedding({
            "pose": pose_data,
            "model_name": model_name,
        })

        # Ensure we return a single embedding array
        if len(embeddings) == 1:
            return [embeddings[0]]  # Return as a list with one element
        return embeddings

    def analyze_video(self, file_path):
        """Analyze the user's video with fallback for alignment errors"""
        self.input_options.hide()
        self.analysis_progress.show()
        self.analysis_status.show()
        self.analysis_progress.setValue(0)
        self.analysis_status.setText("🔍 Analyzing your sign... Please wait")
        QApplication.processEvents()

        temp_dir = tempfile.mkdtemp(prefix="handly_")
        aligned_user_path = os.path.join(temp_dir, "user_aligned.mp4")

        try:
            threshold = self.thresholds.get(self.current_word, 0.5)
            self.analysis_progress.setValue(10)
            self.analysis_status.setText("⏳ Loaded threshold")
            QApplication.processEvents()

            dirs = self.setup_word_directories()
            user_input_path = os.path.join(dirs["user_input"], "user_input.mp4")

            self.align_and_scale_video(file_path, aligned_user_path)
            shutil.copy(aligned_user_path, user_input_path)
            self.analysis_progress.setValue(20)
            self.analysis_status.setText("⏳ Aligned user video")
            QApplication.processEvents()

            correct_embeddings, correct_videos = self.get_correct_embeddings()
            if len(correct_embeddings) == 0:
                raise ValueError("No correct examples found for comparison")

            pose_file = os.path.join(temp_dir, "user_pose.pose")
            self.video_to_pose(aligned_user_path, pose_file)
            self.analysis_progress.setValue(30)
            self.analysis_status.setText("⏳ Generated pose landmarks")
            QApplication.processEvents()

            pose = self.read_pose(pose_file)
            user_embedding_result = self.embed_pose(pose)
            user_embedding = user_embedding_result[0]
            os.remove(pose_file)
            self.analysis_progress.setValue(40)
            self.analysis_status.setText("⏳ Generated pose embedding")
            QApplication.processEvents()

            user_embedding = np.asarray(user_embedding)
            if user_embedding.ndim == 3:
                user_embedding = user_embedding.squeeze()
            if user_embedding.ndim == 1:
                user_embedding = user_embedding.reshape(1, -1)

            all_data = np.vstack([correct_embeddings, user_embedding])
            scaler = StandardScaler()
            all_data_normalized = scaler.fit_transform(all_data)

            correct_embeddings_normalized = all_data_normalized[:-1]
            user_embedding_normalized = all_data_normalized[-1]

            knn_model = NearestNeighbors(n_neighbors=min(6, len(correct_embeddings_normalized)))
            knn_model.fit(correct_embeddings_normalized)

            distances, indices = knn_model.kneighbors(user_embedding_normalized.reshape(1, -1))
            mean_distance = float(distances.mean())

            aligned_correct_path = None
            comparison_path = None
            used_video_index = 0
            good_percentage = 0

            for i, video_index in enumerate(indices[0]):
                try:
                    current_video = correct_videos[video_index]
                    aligned_correct_path = os.path.join(temp_dir, f"correct_aligned_{i}.mp4")

                    self.analysis_progress.setValue(50 + i * 10)
                    self.analysis_status.setText(f"⏳ Aligning correct example {i + 1}/{len(indices[0])}")
                    QApplication.processEvents()

                    self.align_and_scale_video(current_video, aligned_correct_path)
                    used_video_index = i
                    comparison_path = os.path.join(temp_dir, f"comparison_{i}.mp4")

                    frame_matches = self.compare_and_overlay_videos(aligned_user_path, aligned_correct_path,
                                                                    comparison_path)

                    total_frames = len(frame_matches)
                    good_frames = sum(1 for match in frame_matches if match == 'good')
                    good_percentage = (good_frames / total_frames) * 100 if total_frames > 0 else 0

                    if good_percentage >= 90:
                        mean_distance = threshold * 0.8
                        print(f"Overriding classification - 90%+ frame match ({good_percentage:.1f}%)")

                    break

                except Exception as e:
                    print(f"Failed to align video {current_video}, trying next: {str(e)}")
                    continue

            if aligned_correct_path is None:
                raise ValueError("Could not align any of the correct examples")

            self.analysis_progress.setValue(90)
            self.analysis_status.setText(f"⏳ Used example {used_video_index + 1}, Mean distance: {mean_distance:.4f}")
            QApplication.processEvents()

            results = {
                'prediction': 'Correct' if mean_distance < threshold else 'Incorrect',
                'confidence_score': max(0, min(1, 1 - (mean_distance / threshold))),
                'distance': mean_distance,
                'threshold': threshold,
                'closest_correct_example': aligned_correct_path,
                'user_video_path': aligned_user_path,
                'comparison_path': comparison_path,
                'temp_dir': temp_dir,
                'used_video_index': used_video_index,
                'frame_match_percentage': good_percentage
            }

            # Update user stats
            from datetime import datetime
            today = datetime.now().date()
            last_practice = datetime.fromisoformat(self.current_user['stats'].get('last_practice_date', '')).date() if \
            self.current_user['stats'].get('last_practice_date') else None

            # Update practice sessions and word attempts
            self.current_user['stats']['total_practice_sessions'] += 1
            self.current_user['stats']['total_words_attempted'] += 1
            self.current_user['stats']['last_practice_date'] = datetime.now().isoformat()

            # Update streak
            if last_practice:
                if (today - last_practice).days == 1:  # Consecutive day
                    self.current_user['stats']['streak_days'] += 1
                elif (today - last_practice).days > 1:  # Broken streak
                    self.current_user['stats']['streak_days'] = 1
            else:  # First time practicing
                self.current_user['stats']['streak_days'] = 1

            if results['prediction'] == 'Correct':
                if self.current_word in self.current_user.get('struggled_words', []):
                    self.current_user['struggled_words'].remove(self.current_word)

                points = int(results['confidence_score'] * 100)
                self.current_user['stats']['learning_points'] += points
                self.current_user['stats']['total_words_correct'] += 1

                if self.current_word not in self.current_user['word_progress']:
                    self.current_user['word_progress'][self.current_word] = {'attempts': 0, 'correct': 0}

                self.current_user['word_progress'][self.current_word]['attempts'] += 1
                self.current_user['word_progress'][self.current_word]['correct'] += 1

                if (results['confidence_score'] > 0.9 and self.current_word not in self.current_user['known_words']):
                    self.current_user['known_words'].append(self.current_word)
                    self.current_user['known_words'].sort()
            else:
                if self.current_word not in self.current_user['word_progress']:
                    self.current_user['word_progress'][self.current_word] = {'attempts': 0, 'correct': 0}
                self.current_user['word_progress'][self.current_word]['attempts'] += 1

                if 'struggled_words' not in self.current_user:
                    self.current_user['struggled_words'] = []

                if self.current_word not in self.current_user['struggled_words']:
                    self.current_user['struggled_words'].append(self.current_word)
                    self.current_user['struggled_words'] = self.current_user['struggled_words'][-20:]

            self.save_user_data()
            self.refresh_progress_dashboard()  # Refresh the progress view

            self.analysis_progress.setValue(100)
            self.analysis_status.setText("✅ Analysis complete!")
            QApplication.processEvents()

            QTimer.singleShot(1000, lambda: self.show_results(results))

        except Exception as e:
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)

            self.analysis_status.setText(f"❌ Error: {str(e)}")
            self.input_options.show()
            self.analysis_progress.hide()

            QMessageBox.critical(self, "Analysis Error", f"An error occurred: {str(e)}")
    def save_user_data(self):
        """Save the current user's data to file"""
        try:
            # Load all users
            user_data = {}
            if os.path.exists(self.user_data_file):
                with open(self.user_data_file, 'r') as f:
                    user_data = json.load(f)

            # Update current user's data
            user_data[self.current_user['username']] = self.current_user

            # Save back to file
            with open(self.user_data_file, 'w') as f:
                json.dump(user_data, f, indent=2)

        except Exception as e:
            print(f"Error saving user data: {e}")

    def get_correct_embeddings(self):
        """Get embeddings for correct examples from Google Drive's Aligned/Correct folder."""
        embeddings = []
        video_paths = []

        try:
            # Find the videos folder
            videos_folder = self.drive_manager.get_folder_id("videos")
            if not videos_folder:
                return np.array([]), []  # Return empty array with proper format

            # Find the current word folder
            word_folder = self.drive_manager.get_folder_id(self.current_word, videos_folder['id'])
            if not word_folder:
                return np.array([]), []  # Return empty array with proper format

            # Find the Aligned folder
            aligned_folder = self.drive_manager.get_folder_id("Aligned", word_folder['id'])
            if not aligned_folder:
                return np.array([]), []  # Return empty array with proper format

            # Find the Correct folder inside Aligned
            correct_folder = self.drive_manager.get_folder_id("Correct", aligned_folder['id'])
            if not correct_folder:
                return np.array([]), []  # Return empty array with proper format

            # Get up to 3 correct videos from Aligned/Correct
            correct_videos = self.drive_manager.list_files(correct_folder['id'], mime_type='video/mp4')[:3]
            temp_dir = tempfile.mkdtemp()

            for i, video_file in enumerate(correct_videos):
                try:
                    video_path = os.path.join(temp_dir, f"correct_{i}.mp4")
                    self.drive_manager.download_file(video_file['id'], video_path)

                    pose_path = os.path.join(temp_dir, f"correct_{i}.pose")
                    self.video_to_pose(video_path, pose_path)

                    pose = self.read_pose(pose_path)
                    # Get the first element of the embedding result to ensure we have a 1D array
                    embedding_result = self.embed_pose(pose)
                    if isinstance(embedding_result, list) and len(embedding_result) > 0:
                        embedding = embedding_result[0]
                        embeddings.append(embedding)
                        video_paths.append(video_path)

                except Exception as e:
                    print(f"Error processing correct example {i}: {str(e)}")
                    continue

        except Exception as e:
            print(f"Error getting correct embeddings: {str(e)}")

        # Make sure to return a proper numpy array even when empty
        if not embeddings:
            # Return empty array with proper shape (0, n) where n is embedding dimension
            return np.zeros((0, 512)), []  # Assuming embedding size is 512

        return np.array(embeddings), video_paths

    def query_embedding(self, data):
        """Query the embedding API."""
        modality = 'text' if 'text' in data.keys() else 'pose'
        url = f"https://pub.cl.uzh.ch/demo/sign_clip/{modality}"
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps(data)

        try:
            response = requests.request("GET", url, headers=headers, data=payload,verify = False)
            response.raise_for_status()  # Raise exception for bad status codes
            result = json.loads(response.text)

            # Convert to numpy array and ensure proper shape
            embeddings = np.asarray(result['embeddings'])
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)  # Ensure 2D array

            return embeddings

        except requests.exceptions.RequestException as e:
            print(f"Error querying embedding API: {str(e)}")
            return np.array([])  # Return empty array on error

    def compare_and_overlay_videos(self, user_video_path, correct_video_path, output_path):
        """Create a comparison video with aligned user video as background and both poses overlaid.
        Returns a list of frame match results ('good', 'medium', 'bad')"""
        # Open both videos (user video should already be aligned)
        user_cap = cv2.VideoCapture(user_video_path)
        correct_cap = cv2.VideoCapture(correct_video_path)

        if not user_cap.isOpened() or not correct_cap.isOpened():
            raise ValueError("Could not open videos")

        # Get video properties from user video
        frame_width = int(user_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(user_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(user_cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(user_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Visualization parameters
        connection_thickness = 6
        landmark_radius = 6
        threshold = 0.08

        # Colors for visualization
        CORRECT_COLOR = (255, 200, 50)  # Blue for correct pose
        USER_GOOD_COLOR = (0, 255, 0)  # Green for good match
        USER_MEDIUM_COLOR = (0, 255, 255)  # Yellow for medium match
        USER_BAD_COLOR = (0, 0, 255)  # Red for bad match

        # List to store frame match results
        frame_matches = []

        with mp_holistic.Holistic(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
        ) as holistic:
            frame_count = 0

            # First, detect when movement starts in both videos
            def detect_movement_start(cap):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                with mp_holistic.Holistic(min_detection_confidence=0.5) as detector:
                    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                        ret, frame = cap.read()
                        if not ret:
                            return 0

                        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        if results.pose_landmarks:
                            left_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
                            right_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]

                            # Detect when either hand moves up (signing starts)
                            if left_wrist.y < 0.7 or right_wrist.y < 0.7:
                                return max(0, i - 5)  # Start 5 frames before movement
                return 0

            # Find movement start frames
            user_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            user_start_frame = detect_movement_start(user_cap)

            correct_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            correct_start_frame = detect_movement_start(correct_cap)

            # Calculate frame offsets to synchronize
            user_offset = max(0, user_start_frame)
            correct_offset = max(0, correct_start_frame)

            # Set both videos to their respective starting points
            user_cap.set(cv2.CAP_PROP_POS_FRAMES, user_offset)
            correct_cap.set(cv2.CAP_PROP_POS_FRAMES, correct_offset)

            # Process frames in sync
            last_user_frame, last_correct_frame = None, None

            while True:
                ret_user, user_frame = user_cap.read()
                ret_correct, correct_frame = correct_cap.read()

                if not ret_user and not ret_correct:
                    break

                if not ret_user:
                    user_frame = last_user_frame
                else:
                    last_user_frame = user_frame

                if not ret_correct:
                    correct_frame = last_correct_frame
                else:
                    last_correct_frame = correct_frame

                if user_frame is None or correct_frame is None:
                    break

                # Process both frames
                user_image = cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB)
                correct_image = cv2.cvtColor(correct_frame, cv2.COLOR_BGR2RGB)

                user_results = holistic.process(user_image)
                correct_results = holistic.process(correct_image)

                # Start with aligned user video as background
                combined_frame = user_frame.copy()

                # Track match quality for this frame
                frame_match = 'bad'  # Default to bad match
                good_connections = 0
                total_connections = 0

                if user_results.pose_landmarks and correct_results.pose_landmarks:
                    # Calculate alignment transformation between the two poses
                    # We'll use the shoulders and hips for alignment
                    user_landmarks = user_results.pose_landmarks.landmark
                    correct_landmarks = correct_results.pose_landmarks.landmark

                    # Get key points (shoulders and hips)
                    points_user = []
                    points_correct = []

                    for idx in [mp_holistic.PoseLandmark.LEFT_SHOULDER,
                                mp_holistic.PoseLandmark.RIGHT_SHOULDER,
                                mp_holistic.PoseLandmark.LEFT_HIP,
                                mp_holistic.PoseLandmark.RIGHT_HIP]:
                        if (user_landmarks[idx].visibility > 0.5 and
                                correct_landmarks[idx].visibility > 0.5):
                            points_user.append([user_landmarks[idx].x * frame_width,
                                                user_landmarks[idx].y * frame_height])
                            points_correct.append([correct_landmarks[idx].x * frame_width,
                                                   correct_landmarks[idx].y * frame_height])

                    if len(points_user) >= 3:  # Need at least 3 points for good alignment
                        points_user = np.array(points_user, dtype=np.float32)
                        points_correct = np.array(points_correct, dtype=np.float32)

                        # Calculate affine transformation
                        M, _ = cv2.estimateAffinePartial2D(points_correct, points_user)

                        if M is not None:
                            # Apply transformation to correct pose landmarks
                            for landmark in correct_landmarks:
                                if landmark.visibility > 0.5:
                                    point = np.array([landmark.x * frame_width, landmark.y * frame_height, 1])
                                    transformed = M.dot(point)
                                    landmark.x = transformed[0] / frame_width
                                    landmark.y = transformed[1] / frame_height

                    # Draw CORRECT pose (semi-transparent)
                    for connection in mp_holistic.POSE_CONNECTIONS:
                        start_idx, end_idx = connection

                        correct_start = correct_landmarks[start_idx]
                        correct_end = correct_landmarks[end_idx]

                        if correct_start.visibility > 0.5 and correct_end.visibility > 0.5:
                            start_point = (int(correct_start.x * frame_width),
                                           int(correct_start.y * frame_height))
                            end_point = (int(correct_end.x * frame_width),
                                         int(correct_end.y * frame_height))

                            # Draw semi-transparent lines
                            overlay = combined_frame.copy()
                            cv2.line(overlay, start_point, end_point, CORRECT_COLOR, connection_thickness)
                            cv2.addWeighted(overlay, 0.4, combined_frame, 0.6, 0, combined_frame)

                    # Draw USER pose with color-coded feedback and track match quality
                    for connection in mp_holistic.POSE_CONNECTIONS:
                        start_idx, end_idx = connection

                        user_start = user_landmarks[start_idx]
                        user_end = user_landmarks[end_idx]
                        correct_start = correct_landmarks[start_idx]
                        correct_end = correct_landmarks[end_idx]

                        if (user_start.visibility > 0.5 and user_end.visibility > 0.5 and
                                correct_start.visibility > 0.5 and correct_end.visibility > 0.5):

                            # Calculate normalized error distances (0-1)
                            start_error = np.sqrt(
                                (user_start.x - correct_start.x) ** 2 +
                                (user_start.y - correct_start.y) ** 2
                            )
                            end_error = np.sqrt(
                                (user_end.x - correct_end.x) ** 2 +
                                (user_end.y - correct_end.y) ** 2
                            )
                            avg_error = (start_error + end_error) / 2

                            # Calculate color based on error
                            if avg_error < threshold:
                                color = USER_GOOD_COLOR
                                good_connections += 1
                            elif avg_error < threshold * 2:
                                color = USER_MEDIUM_COLOR
                            else:
                                color = USER_BAD_COLOR

                            total_connections += 1

                            # Draw connection
                            start_point = (int(user_start.x * frame_width),
                                           int(user_start.y * frame_height))
                            end_point = (int(user_end.x * frame_width),
                                         int(user_end.y * frame_height))

                            cv2.line(combined_frame, start_point, end_point, color, connection_thickness)
                            cv2.circle(combined_frame, start_point, landmark_radius, color, -1)
                            cv2.circle(combined_frame, end_point, landmark_radius, color, -1)

                    # Determine overall frame match quality
                    if total_connections > 0:
                        good_ratio = good_connections / total_connections
                        if good_ratio >= 0.8:  # 80%+ connections are good
                            frame_match = 'good'
                        elif good_ratio >= 0.5:  # 50%+ connections are good
                            frame_match = 'medium'

                frame_matches.append(frame_match)

                # Add progress indicator
                progress = frame_count / total_frames
                cv2.rectangle(combined_frame, (0, 0), (frame_width, 100), (0, 0, 0), -1)
                cv2.putText(combined_frame, f"Sign: {self.current_word}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(combined_frame, "Blue: Correct Pose | Your Pose: Green (Good) -> Red (Bad)",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Add progress bar
                bar_width = int(frame_width * 0.8)
                bar_start = int(frame_width * 0.1)
                cv2.rectangle(combined_frame, (bar_start, 90), (bar_start + bar_width, 95),
                              (100, 100, 100), -1)
                cv2.rectangle(combined_frame, (bar_start, 90),
                              (bar_start + int(bar_width * progress), 95),
                              (255, 255, 255), -1)

                out.write(combined_frame)
                frame_count += 1

        # Clean up
        for cap in [user_cap, correct_cap]:
            if cap and cap.isOpened():
                cap.release()
        if out and out.isOpened():
            out.release()

        return frame_matches

    def cleanup_temp_files(self):
        """Clean all temporary files including aligned videos with better error handling."""
        try:
            # Clean temp video files
            if hasattr(self, 'temp_video_files'):
                for temp_file in self.temp_video_files[:]:  # Iterate over a copy
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        self.temp_video_files.remove(temp_file)
                    except Exception as e:
                        print(f"Error removing temp file {temp_file}: {str(e)}")

            # Clean alignment temp dirs
            temp_dir = os.path.join(tempfile.gettempdir(), "handly_align")
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    print(f"Error removing temp dir {temp_dir}: {str(e)}")

            # Clean any other temp directories
            for dir_name in os.listdir(tempfile.gettempdir()):
                if dir_name.startswith("handly_"):
                    dir_path = os.path.join(tempfile.gettempdir(), dir_name)
                    if os.path.isdir(dir_path):
                        try:
                            shutil.rmtree(dir_path, ignore_errors=True)
                        except Exception as e:
                            print(f"Error removing temp dir {dir_path}: {str(e)}")

        except Exception as e:
            print(f"Error in cleanup_temp_files: {str(e)}")

    def closeEvent(self, event):
        """Handle window close event to ensure proper cleanup."""
        # Close all game windows
        for window in self.game_windows:
            window.close()

        # Clean up temp files
        self.cleanup_temp_files()

        # Stop any media players
        for player in [self.demo_player, self.user_player,
                       self.correct_player, self.overlay_player]:
            if hasattr(player, 'media_player'):
                player.media_player.stop()

        # Release camera if still open
        if hasattr(self, 'capture') and self.capture and self.capture.isOpened():
            self.capture.release()

        # Call parent's close event
        super().closeEvent(event)


class EnhancedVideoPlayer(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Title bar
        title_bar = QWidget()
        title_bar.setStyleSheet("background-color: #4a6baf; border-radius: 4px 4px 0 0;")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(10, 5, 10, 5)

        title_label = QLabel(self.title)
        title_label.setStyleSheet("color: white; font-weight: bold;")
        title_layout.addWidget(title_label)

        self.fullscreen_btn = QPushButton()
        self.fullscreen_btn.setIcon(QIcon.fromTheme("view-fullscreen"))
        self.fullscreen_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                padding: 2px;
            }
            QPushButton:hover {
                background: rgba(255,255,255,0.2);
                border-radius: 2px;
            }
        """)
        title_layout.addWidget(self.fullscreen_btn)

        layout.addWidget(title_bar)

        # Video display
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_widget)

        # Controls
        controls = QWidget()
        controls.setStyleSheet("background-color: #f8f9fa; border-radius: 0 0 4px 4px;")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(10, 5, 10, 5)

        self.play_btn = QPushButton()
        self.play_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_btn.setStyleSheet("padding: 5px;")

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #ddd;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 12px;
                height: 12px;
                background: #4a6baf;
                border-radius: 6px;
                margin: -4px 0;
            }
            QSlider::sub-page:horizontal {
                background: #4a6baf;
                border-radius: 2px;
            }
        """)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("font-size: 12px; color: #666;")

        self.volume_btn = QPushButton()
        self.volume_btn.setIcon(QIcon.fromTheme("audio-volume-medium"))
        self.volume_btn.setStyleSheet("padding: 5px;")

        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.time_slider)
        controls_layout.addWidget(self.time_label)
        controls_layout.addWidget(self.volume_btn)

        layout.addWidget(controls)

    def setup_progress_page(self):
        """Setup the progress dashboard with modern design"""
        self.progress_page = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header
        header = QLabel(f"{self.current_user['username']}'s Progress")
        header.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #4a6baf;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        """)
        layout.addWidget(header)

        # Stats cards in a grid
        stats_grid = QGridLayout()
        stats_grid.setSpacing(15)

        # Learning stats card
        learning_stats = QWidget()
        learning_layout = QVBoxLayout(learning_stats)

        title = QLabel("📚 Learning")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #4a6baf;")

        stats = self.current_user['stats']
        accuracy = (stats['total_words_correct'] / stats['total_words_attempted'] * 100) if stats[
                                                                                                'total_words_attempted'] > 0 else 0

        stats_text = QLabel(f"""
            <p>• <b>Sessions:</b> {stats['total_practice_sessions']}</p>
            <p>• <b>Accuracy:</b> {accuracy:.1f}%</p>
            <p>• <b>Points:</b> {stats['learning_points']}</p>
            <p>• <b>Streak:</b> {stats['streak_days']} days</p>
        """)
        stats_text.setStyleSheet("font-size: 14px;")

        learning_layout.addWidget(title)
        learning_layout.addWidget(stats_text)
        stats_grid.addWidget(self.create_card("Learning Stats", learning_stats), 0, 0)

        # Game stats card
        game_stats = QWidget()
        game_layout = QVBoxLayout(game_stats)

        title = QLabel("🎮 Games")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #4a6baf;")

        game_points = stats['game_points']
        total_points = game_points.get('quiz', 0) + game_points.get('clock', 0)

        stats_text = QLabel(f"""
            <p>• <b>Total Points:</b> {total_points}</p>
            <p>• <b>Quiz Points:</b> {game_points.get('quiz', 0)}</p>
            <p>• <b>Clock Game:</b> {game_points.get('clock', 0)}</p>
        """)
        stats_text.setStyleSheet("font-size: 14px;")

        game_layout.addWidget(title)
        game_layout.addWidget(stats_text)
        stats_grid.addWidget(self.create_card("Game Stats", game_stats), 0, 1)

        # Achievements card
        achievements = QWidget()
        achievements_layout = QVBoxLayout(achievements)

        title = QLabel("🏆 Achievements")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #4a6baf;")

        # Sample achievements - in a real app you'd load these from user data
        achievements_list = QListWidget()
        achievements_list.setStyleSheet("""
            QListWidget {
                border: none;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 8px 0;
                border-bottom: 1px solid #eee;
            }
        """)

        items = [
            ("🌟 First Sign", "Performed your first sign correctly", True),
            ("🔥 3-Day Streak", "Practiced for 3 days in a row", stats['streak_days'] >= 3),
            ("⚡ Fast Learner", "Achieved 80% accuracy", accuracy >= 80),
            ("🏆 Perfect 5", "Got 5 signs correct in a row", False)
        ]

        for icon, text, unlocked in items:
            item = QListWidgetItem(f"{icon} {text}")
            if not unlocked:
                item.setForeground(QColor("#999"))
            achievements_list.addItem(item)

        achievements_layout.addWidget(title)
        achievements_layout.addWidget(achievements_list)
        stats_grid.addWidget(self.create_card("Achievements", achievements), 1, 0, 1, 2)

        layout.addLayout(stats_grid)

        # Charts section
        charts_title = QLabel("📈 Progress Over Time")
        charts_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #4a6baf;")
        layout.addWidget(charts_title)

        # Sample charts (in a real app these would be real charts)
        charts_grid = QGridLayout()
        charts_grid.setSpacing(15)

        # Accuracy chart
        accuracy_chart = self.create_chart("Accuracy Trend", [65, 70, 75, 80, 85, 90], "#4a6baf")
        charts_grid.addWidget(self.create_card("Accuracy", accuracy_chart), 0, 0)

        # Words learned chart
        words_chart = self.create_chart("Words Learned", [5, 10, 15, 20, 25], "#6e8efb")
        charts_grid.addWidget(self.create_card("Vocabulary", words_chart), 0, 1)

        # Activity chart
        activity_chart = self.create_chart("Weekly Activity", [3, 5, 7, 4, 6, 8, 5], "#a777e3")
        charts_grid.addWidget(self.create_card("Activity", activity_chart), 1, 0, 1, 2)

        layout.addLayout(charts_grid)
        layout.addStretch()

        scroll.setWidget(content)
        self.stacked_widget.addWidget(scroll)

    def setup_animations(self):
        """Setup animations for smoother transitions"""
        self.animation_group = QParallelAnimationGroup()

        # Fade animation for page transitions
        self.fade_animation = QPropertyAnimation(self.stacked_widget, b"windowOpacity")
        self.fade_animation.setDuration(300)
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.finished.connect(self.complete_page_transition)
        self.animation_group.addAnimation(self.fade_animation)

        # Slide animation for widgets
        self.slide_animation = QPropertyAnimation(self, b"pos")
        self.slide_animation.setDuration(400)
        self.slide_animation.setEasingCurve(QEasingCurve.OutQuad)
        self.animation_group.addAnimation(self.slide_animation)

    def set_current_page(self, index):
        """Animate page transitions"""
        if not self.animation_group.state() == QAnimationGroup.Running:
            self.next_page_index = index
            self.fade_animation.start()

    def complete_page_transition(self):
        """Complete the page transition after fade out"""
        self.stacked_widget.setCurrentIndex(self.next_page_index)

        # Fade back in
        fade_in = QPropertyAnimation(self.stacked_widget, b"windowOpacity")
        fade_in.setDuration(300)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.start()

    def resizeEvent(self, event):
        """Handle window resize events to update video sizes"""
        super().resizeEvent(event)

        # Update current frames if available
        if self.correct_frames and self.current_frame_idx < len(self.correct_frames):
            correct_data = self.correct_frames[self.current_frame_idx]
            self.update_label(self.correct_label, correct_data['processed'])

        # Update camera view if available
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                correct_data = self.correct_frames[self.current_frame_idx]
                user_results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                camera_display = self.process_user_frame(frame, user_results, correct_data['landmarks'])
                self.update_label(self.camera_label, camera_display)

    class FeedbackWidget(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setup_ui()

        def setup_ui(self):
            layout = QVBoxLayout(self)
            layout.setContentsMargins(15, 15, 15, 15)
            layout.setSpacing(10)

            # Header
            self.header = QLabel("Feedback")
            self.header.setStyleSheet("""
                font-size: 18px;
                font-weight: bold;
                color: #4a6baf;
                padding-bottom: 5px;
                border-bottom: 1px solid #eee;
            """)
            layout.addWidget(self.header)

            # Score meter
            self.score_meter = QProgressBar()
            self.score_meter.setRange(0, 100)
            self.score_meter.setTextVisible(False)
            self.score_meter.setStyleSheet("""
                QProgressBar {
                    height: 20px;
                    border-radius: 10px;
                    background: #f0f0f0;
                }
                QProgressBar::chunk {
                    border-radius: 10px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #ff5e5e, stop:0.5 #ffcc00, stop:1 #4CAF50);
                }
            """)
            layout.addWidget(self.score_meter)

            # Score label
            self.score_label = QLabel()
            self.score_label.setStyleSheet("font-size: 16px; font-weight: bold;")
            layout.addWidget(self.score_label)

            # Feedback text
            self.feedback_text = QTextEdit()
            self.feedback_text.setReadOnly(True)
            self.feedback_text.setStyleSheet("""
                QTextEdit {
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 10px;
                    font-size: 14px;
                }
            """)
            layout.addWidget(self.feedback_text)

            # Suggestions
            self.suggestions_list = QListWidget()
            self.suggestions_list.setStyleSheet("""
                QListWidget {
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    font-size: 14px;
                }
                QListWidget::item {
                    padding: 8px;
                    border-bottom: 1px solid #eee;
                }
                QListWidget::item:last {
                    border-bottom: none;
                }
            """)
            self.suggestions_list.setWordWrap(True)
            layout.addWidget(self.suggestions_list)

        def set_feedback(self, score, feedback, suggestions):
            """Update the feedback display"""
            # Set score
            self.score_meter.setValue(int(score))
            self.score_label.setText(f"Match Score: {score:.1f}%")

            # Set feedback color based on score
            if score >= 80:
                color = "#4CAF50"  # Green
                self.score_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")
            elif score >= 60:
                color = "#FFC107"  # Yellow
                self.score_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")
            else:
                color = "#F44336"  # Red
                self.score_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")

            # Set feedback text
            self.feedback_text.setPlainText(feedback)

            # Add suggestions
            self.suggestions_list.clear()
            for suggestion in suggestions:
                item = QListWidgetItem(f"• {suggestion}")
                self.suggestions_list.addItem(item)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())