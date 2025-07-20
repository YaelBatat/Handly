import math
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
from fairseq.examples.MMPT.demo_sign import embed_pose, embed_text, preprocess_pose, preprocess_text
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QFrame, QTextEdit, QProgressBar, QSplitter,
                             QMessageBox, QComboBox, QStyle, QSlider, QGroupBox,
                             QSizePolicy, QStackedWidget, QScrollArea, QSplitter, QLineEdit, QGridLayout, QTabWidget,
                             QTableWidget, QTableWidgetItem, QDesktopWidget)
from PyQt5.QtCore import Qt, QUrl, QTimer, QSize, pyqtSignal, QPointF, QRectF, QPoint
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon, QPalette, QColor, QPolygonF, QPainter, QPen, QPolygon, \
    QRadialGradient, QLinearGradient
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QSound
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QSoundEffect
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QDialog
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
import pygame  # For playing audio
from io import BytesIO
import threading

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = 'client_secret.json'
TOKEN_FILE = 'token.json'


class PracticeWidget(QWidget):
    def __init__(self, correct_video_path, parent=None):
        super().__init__(parent)
        self.correct_video_path = correct_video_path
        self.correct_frames = []
        self.current_frame_idx = 0
        self.cap = None
        self.timer = QTimer()
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            static_image_mode=False
        )
        self.hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2
        )

        # Auto-advance settings
        self.auto_advance = True  # Enabled by default
        self.match_threshold = 70  # Required match score to auto-advance
        self.match_duration = 0.1  # Seconds of good match required

        # Tracking variables
        self.good_match_start_time = None
        self.current_match_score = 0
        self.session_complete = False  # Track if practice session is complete

        # Voice feedback state

        # Voice feedback state initialization
        # Voice feedback state
        self.voice_queue = []
        self.is_speaking = False
        self.current_voice_feedback = ""
        self.next_voice_feedback = ""
        self.voice_lock = threading.Lock()
        # Voice feedback
        self.voice_feedback_enabled = True
        self.last_feedback_time = 0
        self.feedback_cooldown = 5  # seconds between feedbacks
        self.last_spoken_feedback = ""  # Track the last spoken feedback
        pygame.mixer.init()  # Initialize audio mixer

        self.setup_ui()
        self.show_usage_tips()  # Show tips before loading frames

    def show_usage_tips(self):
        """Show a full-screen usage tips dialog before starting practice"""
        self.tips_dialog = QDialog(self)
        self.tips_dialog.setWindowTitle("Practice Mode Tips")
        self.tips_dialog.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)

        # Make it full screen
        screen = QDesktopWidget().screenGeometry()
        self.tips_dialog.resize(screen.width(), screen.height())

        layout = QVBoxLayout(self.tips_dialog)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(30)

        # Title
        title = QLabel("✋ Practice Mode Tips")
        title.setStyleSheet("font-size: 36px; font-weight: bold; color: #4a6baf;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        # Tips content
        tips_content = QTextEdit()
        tips_content.setReadOnly(True)
        tips_content.setStyleSheet("""
            QTextEdit {
                font-size: 24px;
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 10px;
                padding: 20px;
            }
        """)

        tips_html = """
        <h2 style='color:#4a6baf;'>How to Get the Best Results:</h2>
        <ol>
            <li><b>Position Yourself Properly</b><br>
            • Stand about 2 meters (6 feet) from your camera<br>
            • Make sure your <span style='color:#d63384;'>entire upper body</span> is visible<br>
            • Keep your <span style='color:#d63384;'>hands at waist level</span> when starting</li>
            <br>

            <li><b>Camera Setup</b><br>
            • Ensure good lighting - avoid backlighting<br>
            • Remove distracting backgrounds<br>
            • Wear contrasting colors (avoid skin-tone colors)</li>
            <br>

            <li><b>During Practice</b><br>
            • The <span style='color:#4a6baf;'>blue transparent skeleton</span> shows the correct pose<br>
            • <span style='color:#28a745;'>Green lines</span> mean you're matching well<br>
            • <span style='color:#dc3545;'>Red lines</span> show where adjustments are needed</li>
            <br>

            <li><b>Navigation</b><br>
            • Use <span style='color:#4a6baf;'>Previous/Next Frame</span> buttons to move<br>
            • When you match well (≥80%), it will auto-advance after 0.4 seconds<br>
            • Toggle auto-advance if you prefer manual control</li>
        </ol>

        <h3 style='color:#6f42c1;'>Remember:</h3>
        <p>Focus on <span style='color:#d63384;'>hand shapes</span> first, then <span style='color:#d63384;'>movement timing</span>.</p>
        """

        tips_content.setHtml(tips_html)
        layout.addWidget(tips_content, 1)

        # Start button
        start_btn = QPushButton("Start Practicing")
        start_btn.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                padding: 15px 30px;
                background-color: #28a745;
                color: white;
                border-radius: 8px;
                min-width: 300px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        start_btn.clicked.connect(self.start_practice)
        layout.addWidget(start_btn, 0, Qt.AlignCenter)

        self.tips_dialog.exec_()

    def start_practice(self):
        """Called when user clicks Start button in tips dialog"""
        self.tips_dialog.accept()
        self.load_correct_frames()
        self.setup_camera()

    def setup_ui(self):
        self.setWindowTitle("Practice Mode - Frame by Frame")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.resize(1200, 800)  # Wider window to accommodate side panel

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Left side - Video views
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(15)

        # Title
        title = QLabel("✋ Sign Language Practice Mode")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #4a6baf;")
        video_layout.addWidget(title, alignment=Qt.AlignCenter)

        # Video splitter
        video_splitter = QSplitter(Qt.Horizontal)

        # Camera view
        self.camera_group = QGroupBox("Your Camera")
        camera_layout = QVBoxLayout()
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(480, 360)
        camera_layout.addWidget(self.camera_label)
        self.camera_group.setLayout(camera_layout)

        # Correct pose view
        self.correct_group = QGroupBox("Correct Pose")
        correct_layout = QVBoxLayout()
        self.correct_label = QLabel()
        self.correct_label.setAlignment(Qt.AlignCenter)
        self.correct_label.setMinimumSize(480, 360)
        correct_layout.addWidget(self.correct_label)
        self.correct_group.setLayout(correct_layout)

        video_splitter.addWidget(self.camera_group)
        video_splitter.addWidget(self.correct_group)
        video_layout.addWidget(video_splitter)

        # Progress
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("font-size: 16px;")
        video_layout.addWidget(self.progress_label)

        # Controls
        controls_layout = QHBoxLayout()
        self.back_btn = QPushButton("◀ Back to Learning")
        self.back_btn.clicked.connect(self.close)
        controls_layout.addWidget(self.back_btn)

        self.prev_frame_btn = QPushButton("⏮ Previous Frame")
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        controls_layout.addWidget(self.prev_frame_btn)

        self.next_frame_btn = QPushButton("Next Frame ⏭")
        self.next_frame_btn.clicked.connect(self.next_frame)
        controls_layout.addWidget(self.next_frame_btn)

        video_layout.addLayout(controls_layout)
        main_layout.addLayout(video_layout, stretch=3)

        # Right side - Feedback panel
        feedback_panel = QFrame()
        feedback_panel.setFrameShape(QFrame.StyledPanel)
        feedback_panel.setStyleSheet("background-color: #f8f9fa; border-radius: 10px;")
        feedback_layout = QVBoxLayout(feedback_panel)
        feedback_layout.setContentsMargins(15, 15, 15, 15)
        feedback_layout.setSpacing(15)

        # Match meter with icon
        meter_layout = QHBoxLayout()
        self.match_icon = QLabel()
        self.match_icon.setFixedSize(40, 40)
        meter_layout.addWidget(self.match_icon)

        self.match_meter = QProgressBar()
        self.match_meter.setRange(0, 100)
        self.match_meter.setTextVisible(True)
        self.match_meter.setFormat("%p% Match")
        self.match_meter.setStyleSheet("""
            QProgressBar {
                height: 30px;
                border-radius: 5px;
                text-align: center;
                font-size: 16px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 5px;
            }
        """)
        meter_layout.addWidget(self.match_meter, stretch=1)
        feedback_layout.addLayout(meter_layout)

        # Main feedback
        self.feedback_text = QLabel()
        self.feedback_text.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            min-height: 40px;
        """)
        self.feedback_text.setWordWrap(True)
        feedback_layout.addWidget(self.feedback_text)

        # Detailed feedback
        self.detailed_feedback = QTextEdit()
        self.detailed_feedback.setReadOnly(True)
        self.detailed_feedback.setStyleSheet("""
            QTextEdit {
                font-size: 20px;
                background-color: white;
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 15px;
            }
        """)
        feedback_layout.addWidget(self.detailed_feedback, stretch=1)

        # Completion message label (initially hidden)
        self.completion_label = QLabel(self)
        self.completion_label.setAlignment(Qt.AlignCenter)
        self.completion_label.setStyleSheet("""
            font-size: 48px;
            font-weight: bold;
            color: #FF5722;
            background-color: rgba(255,255,255,0.8);
            border-radius: 20px;
            padding: 30px;
        """)
        self.completion_label.hide()
        self.completion_label.raise_()

        # Confetti effect placeholder
        self.confetti = QLabel(self)
        self.confetti.hide()
        self.confetti.raise_()

        # Session complete panel
        self.session_complete_panel = QFrame()
        self.session_complete_panel.setStyleSheet("""
            background-color: #e8f5e9;
            border-radius: 10px;
            padding: 20px;
        """)
        self.session_complete_panel.hide()

        complete_layout = QVBoxLayout(self.session_complete_panel)
        complete_layout.setContentsMargins(20, 20, 20, 20)
        complete_layout.setSpacing(20)

        complete_label = QLabel("🎉 Practice Session Complete!")
        complete_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #2e7d32;
            qproperty-alignment: AlignCenter;
        """)
        complete_layout.addWidget(complete_label)

        self.final_score_label = QLabel()
        self.final_score_label.setStyleSheet("font-size: 20px;")
        complete_layout.addWidget(self.final_score_label)

        # Buttons for session completion
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)

        self.practice_again_btn = QPushButton("🔄 Practice Again")
        self.practice_again_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 12px 24px;
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
        """)
        self.practice_again_btn.clicked.connect(self.restart_practice)
        btn_layout.addWidget(self.practice_again_btn)

        self.return_btn = QPushButton("🏠 Return to Learning")
        self.return_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 12px 24px;
                background-color: #2196F3;
                color: white;
                border-radius: 8px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.return_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.return_btn)

        complete_layout.addLayout(btn_layout)
        feedback_layout.addWidget(self.session_complete_panel)

        # Toggle buttons at bottom
        toggle_layout = QHBoxLayout()

        self.auto_advance_btn = QPushButton("Auto Advance: ON")
        self.auto_advance_btn.setCheckable(True)
        self.auto_advance_btn.setChecked(True)
        self.auto_advance_btn.setStyleSheet("padding: 8px;")
        self.auto_advance_btn.clicked.connect(self.toggle_auto_advance)
        toggle_layout.addWidget(self.auto_advance_btn)

        self.voice_feedback_btn = QPushButton("Voice: ON")
        self.voice_feedback_btn.setCheckable(True)
        self.voice_feedback_btn.setChecked(True)
        self.voice_feedback_btn.setStyleSheet("padding: 8px;")
        self.voice_feedback_btn.clicked.connect(self.toggle_voice_feedback)
        toggle_layout.addWidget(self.voice_feedback_btn)

        feedback_layout.addLayout(toggle_layout)
        main_layout.addWidget(feedback_panel, stretch=1)

    def restart_practice(self):
        """Restart the practice session from the beginning."""
        self.session_complete = False

        # Hide completion elements
        if hasattr(self, 'completion_label'):
            self.completion_label.hide()
        if hasattr(self, 'confetti'):
            self.confetti.hide()
        if hasattr(self, 'session_complete_panel'):
            self.session_complete_panel.hide()

        # Reset practice state
        self.current_frame_idx = 0
        self.update_progress_text()
        self.good_match_start_time = None

        # Re-enable the navigation buttons
        self.prev_frame_btn.setEnabled(True)
        self.next_frame_btn.setEnabled(True)

        # Reset feedback display
        self.feedback_text.clear()
        self.detailed_feedback.clear()
        self.match_meter.setValue(0)

        # Stop any ongoing audio
        try:
            pygame.mixer.music.stop()
        except:
            pass

    def update_feedback(self, score, feedback):
        """Update feedback with complete voice output"""
        self.match_meter.setValue(int(score))

        # Simple status colors
        if score >= self.match_threshold:
            status_color = "#4CAF50"  # Green
            status_text = "✓ Good match"
        elif score >= self.match_threshold * 0.7:
            status_color = "#FFC107"  # Yellow
            status_text = "⚠ Getting closer"
        else:
            status_color = "#F44336"  # Red
            status_text = "✗ Needs work"

        # Update meter
        self.match_meter.setStyleSheet(f"""
            QProgressBar {{
                height: 30px;
                border-radius: 5px;
                text-align: center;
                font-size: 16px;
            }}
            QProgressBar::chunk {{
                background-color: {status_color};
                border-radius: 5px;
            }}
        """)

        # Status text
        self.feedback_text.setText(
            f'<span style="color:{status_color}; font-size: 20px;">{status_text}</span>'
        )

        # Update visual feedback
        html_feedback = """
        <style>
            li { margin-bottom: 12px; font-size: 18px; color: #333; }
        </style>
        <ul style='margin-left: 5px; padding-left: 15px;'>
        """
        current_feedback = feedback[:3] if feedback else []
        for item in current_feedback:
            html_feedback += f"<li>{item}</li>"

        # Add positive reinforcement when matching well with no specific feedback
        if score >= self.match_threshold and not current_feedback:
            positive_feedback = "You're matching the pose well!"
            html_feedback += f"<li style='color:#4CAF50;'>{positive_feedback}</li>"
            current_feedback.append(positive_feedback)

        html_feedback += "</ul>"
        self.detailed_feedback.setHtml(html_feedback)

        # Update voice feedback if enabled
        if self.voice_feedback_enabled and current_feedback:
            new_feedback = " ".join(current_feedback)  # Combine all feedback items

            # Only speak if the feedback is different from last spoken
            if new_feedback != self.last_spoken_feedback:
                self.last_spoken_feedback = new_feedback
                with self.voice_lock:
                    if not self.is_speaking:
                        # If not speaking, start immediately
                        self.current_voice_feedback = new_feedback
                        self._speak_feedback()
                    else:
                        # If speaking, queue the next feedback (replaces any pending feedback)
                        self.next_voice_feedback = new_feedback

    def _speak_feedback(self):
        """Speak the current feedback in a separate thread"""

        def speak():
            with self.voice_lock:
                if not self.current_voice_feedback or self.is_speaking:
                    return

                self.is_speaking = True

            try:
                # Stop any current speech
                pygame.mixer.music.stop()

                # Create speech
                tts = gTTS(text=self.current_voice_feedback, lang='en')
                fp = BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)

                # Play and wait for completion
                pygame.mixer.music.load(fp)
                pygame.mixer.music.play()

                # Wait for speech to complete
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

            except Exception as e:
                print(f"Voice error: {e}")
            finally:
                with self.voice_lock:
                    self.is_speaking = False
                    # If there's queued feedback, speak it now only if different
                    if self.next_voice_feedback and self.next_voice_feedback != self.last_spoken_feedback:
                        self.current_voice_feedback = self.next_voice_feedback
                        self.last_spoken_feedback = self.next_voice_feedback
                        self.next_voice_feedback = ""
                        self._speak_feedback()

        # Start in a new thread
        threading.Thread(target=speak, daemon=True).start()

    def speak_status(self, text):
        """Speak just the status (good match/getting closer/needs work)"""
        if not self.voice_feedback_enabled or self.is_speaking:
            return

        # Clear any pending feedback
        self.voice_queue = []

        # Speak immediately
        threading.Thread(target=self._speak_text, args=(text, False), daemon=True).start()

    def speak_feedback(self, text):
        """Queue detailed feedback to be spoken"""
        if not self.voice_feedback_enabled:
            return

        # Clear any pending feedback if we're getting new instructions
        if self.voice_queue:
            self.voice_queue = []

        self.voice_queue.append(text)

        # Start speaking if not already
        if not self.is_speaking:
            self._process_voice_queue()

    def _process_voice_queue(self):
        """Process the voice queue one item at a time"""
        if not self.voice_queue or self.is_speaking:
            return

        text = self.voice_queue.pop(0)
        threading.Thread(target=self._speak_text, args=(text, True), daemon=True).start()

    def _speak_text(self, text, is_feedback):
        """Actually speak the text (runs in separate thread)"""
        self.is_speaking = True

        try:
            # Stop any current speech
            pygame.mixer.music.stop()

            # Create speech
            tts = gTTS(text=text, lang='en')
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)

            # Play and wait for completion
            pygame.mixer.music.load(fp)
            pygame.mixer.music.play()

            # Wait for speech to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

        except Exception as e:
            print(f"Voice error: {e}")
        finally:
            self.is_speaking = False
            # Process next item in queue if any
            QTimer.singleShot(100, self._process_voice_queue)
    def setup_camera(self):
        """Setup camera with same parameters as recording view"""
        # Release any existing capture
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()

        # Initialize camera with same settings as recording
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", "Could not open camera")
            self.close()
            return

        # Set camera properties to match recording view
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Wider resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Start timer for frame updates
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        if not self.cap or not self.cap.isOpened() or self.current_frame_idx >= len(self.correct_frames):
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Process frame
        current_time = time.time()
        correct_data = self.correct_frames[self.current_frame_idx]
        user_results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Check if hands are lowered (session complete condition)
        if self.check_hands_lowered(correct_data['landmarks']):
            self.handle_session_complete()
            return

        # Process frames with new scaling
        camera_display = self.process_user_frame(frame, user_results, correct_data['landmarks'])
        self.update_label(self.camera_label, camera_display)

        # Show correct pose
        self.update_label(self.correct_label, correct_data['processed'])

        if user_results.pose_landmarks and correct_data['landmarks']:
            match_score, feedback = self.compare_poses(user_results, correct_data['landmarks'])
            self.current_match_score = round(match_score, 2)
            self.update_feedback(match_score, feedback)

            # Auto-advance logic - only if hands are matching well
            hand_match = self.check_hand_match(user_results, correct_data['landmarks'])
            if self.auto_advance and hand_match:
                if self.good_match_start_time is None:
                    self.good_match_start_time = current_time

                elapsed = current_time - self.good_match_start_time
                if elapsed >= self.match_duration:
                    self.next_frame()
            else:
                self.good_match_start_time = None

    def check_hand_match(self, user_results, correct_landmarks):
        """Check if hands match well enough for auto-advance"""
        try:
            # Get transformation matrix
            user_ref_points = self.get_reference_points(user_results)
            correct_ref_points = self.get_reference_points(correct_landmarks)

            # Check if we have valid reference points
            if user_ref_points is None or correct_ref_points is None:
                return False

            transform = self.get_similarity_transform(correct_ref_points, user_ref_points)

            # Check left hand
            left_ok = False
            if (hasattr(user_results, 'left_hand_landmarks') and
                    user_results.left_hand_landmarks and
                    hasattr(correct_landmarks, 'left_hand_landmarks') and
                    correct_landmarks.left_hand_landmarks):

                score, _ = self.compare_single_hand(
                    user_results.left_hand_landmarks,
                    correct_landmarks.left_hand_landmarks,
                    "left hand",
                    transform
                )
                left_ok = score >= self.match_threshold
            else:
                # Fall back to wrist position if no hand data
                wrist_pos = self.estimate_wrist_position(correct_landmarks, 'left')
                if wrist_pos:
                    score, _ = self.compare_hand_position_only(
                        user_results.left_hand_landmarks,
                        wrist_pos,
                        "left hand",
                        transform
                    )
                    left_ok = score >= (self.match_threshold * 0.8)  # Lower threshold for wrist-only

            # Check right hand
            right_ok = False
            if (hasattr(user_results, 'right_hand_landmarks') and
                    user_results.right_hand_landmarks and
                    hasattr(correct_landmarks, 'right_hand_landmarks') and
                    correct_landmarks.right_hand_landmarks):

                score, _ = self.compare_single_hand(
                    user_results.right_hand_landmarks,
                    correct_landmarks.right_hand_landmarks,
                    "right hand",
                    transform
                )
                right_ok = score >= self.match_threshold
            else:
                # Fall back to wrist position if no hand data
                wrist_pos = self.estimate_wrist_position(correct_landmarks, 'right')
                if wrist_pos:
                    score, _ = self.compare_hand_position_only(
                        user_results.right_hand_landmarks,
                        wrist_pos,
                        "right hand",
                        transform
                    )
                    right_ok = score >= (self.match_threshold * 0.8)  # Lower threshold for wrist-only

            return left_ok and right_ok

        except Exception as e:
            print(f"Error in check_hand_match: {e}")
            return False

    def check_hands_lowered(self, landmarks):
        """Check if hands are lowered (indicating end of sign)."""
        if not landmarks or not landmarks.pose_landmarks:
            return False

        try:
            # Get relevant landmarks
            left_wrist = landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_WRIST]
            left_hip = landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_HIP]

            # Calculate waist level (more robust calculation)
            waist_level = (left_hip.y + right_hip.y) / 2

            # Check if wrists are below waist level (with more lenient threshold)
            wrists_lowered = (left_wrist.y > waist_level + 0.15 and
                              right_wrist.y > waist_level + 0.15)

            # Additionally check if hands are not visible (optional)
            hands_not_visible = (not hasattr(landmarks, 'left_hand_landmarks')) and \
                                (not hasattr(landmarks, 'right_hand_landmarks'))

            return wrists_lowered or hands_not_visible
        except Exception as e:
            print(f"Error in hands lowered check: {e}")
            return False

    def handle_session_complete(self):
        """Handle session completion with visible buttons"""
        if not self.session_complete:
            self.session_complete = True
            self.timer.stop()  # Stop frame updates

            # Set final score
            final_score = max(0, min(100, self.current_match_score))

            # Show completion elements in correct order
            if hasattr(self, 'confetti'):
                self.show_confetti_effect()

            # Show and raise the panel with buttons FIRST
            if hasattr(self, 'session_complete_panel'):
                self.session_complete_panel.move(
                    self.width() // 2 - self.session_complete_panel.width() // 2,
                    self.height() // 2 - self.session_complete_panel.height() // 2
                )
                self.session_complete_panel.show()
                self.session_complete_panel.raise_()  # Bring to front
                self.final_score_label.setText(f"Final Score: {final_score}%")

            # Then show the completion label (will be behind the panel)
            if hasattr(self, 'completion_label'):
                self.completion_label.setText(f"Excellent Job! 🎉\nScore: {final_score}%")
                self.completion_label.resize(self.size())
                self.completion_label.show()
                # Don't call lower() here - let it stay at natural z-order

            # Play completion audio
            self.play_completion_sequence(final_score)

            # Disable navigation buttons
            self.prev_frame_btn.setEnabled(False)
            self.next_frame_btn.setEnabled(False)

    def play_completion_sequence(self, final_score):
        """Play completion sounds and voice without interruption"""
        try:
            # Stop any ongoing audio and clear queue
            pygame.mixer.music.stop()
            with self.voice_lock:
                self.current_voice_feedback = ""
                self.next_voice_feedback = ""
                self.is_speaking = False

            # Play celebration sound
            success_sound = os.path.join(os.path.dirname(__file__), "success.wav")
            if os.path.exists(success_sound):
                pygame.mixer.music.load(success_sound)
                pygame.mixer.music.play()

                # Wait for sound to finish before speaking
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

            # Speak only the completion message
            completion_message = (
                f"Practice session complete! Your final score was {final_score} percent. "
                "Excellent work!"
            )
            with self.voice_lock:
                self.current_voice_feedback = completion_message
                self._speak_feedback()

        except Exception as e:
            print(f"Audio error: {e}")
            # Fallback if audio fails
            with self.voice_lock:
                self.current_voice_feedback = "Practice complete! Excellent work!"
                self._speak_feedback()

    def show_confetti_effect(self):
        """Working confetti animation that stays behind buttons"""
        if not hasattr(self, 'confetti'):
            return

        self.confetti.clear()
        pixmap = QPixmap(self.width(), self.height())
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw colorful confetti pieces
        rng = random.Random()
        colors = [
            QColor(255, 0, 0, 180),  # Red
            QColor(0, 255, 0, 180),  # Green
            QColor(0, 0, 255, 180),  # Blue
            QColor(255, 255, 0, 180),  # Yellow
            QColor(255, 0, 255, 180)  # Purple
        ]

        for _ in range(100):
            color = rng.choice(colors)
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)

            # Draw different shapes
            shape_type = rng.randint(0, 2)
            x = rng.randint(0, self.width())
            y = rng.randint(0, self.height())
            size = rng.randint(8, 20)

            if shape_type == 0:
                painter.drawEllipse(x, y, size, size)
            elif shape_type == 1:
                painter.drawRect(x, y, size, size)
            else:
                points = QPolygon([
                    QPoint(x, y),
                    QPoint(x + size, y),
                    QPoint(x + size // 2, y + size)
                ])
                painter.drawPolygon(points)

        painter.end()
        self.confetti.setPixmap(pixmap)
        self.confetti.resize(self.size())
        self.confetti.show()
        self.confetti.lower()  # Keep behind other elements

        # Animate falling
        self.confetti_pos = 0
        self.confetti_timer = QTimer()
        self.confetti_timer.timeout.connect(self.update_confetti)
        self.confetti_timer.start(30)  # Faster animation

    def update_confetti(self):
        """Animate confetti without blocking UI"""
        self.confetti_pos += 8
        if self.confetti_pos > self.height():
            self.confetti_timer.stop()
            self.confetti.hide()
        else:
            self.confetti.move(0, self.confetti_pos)
            # Ensure buttons stay on top
            if hasattr(self, 'session_complete_panel'):
                self.session_complete_panel.raise_()

    def next_frame(self):
        """Move to next frame and reset auto-advance tracking"""
        if self.current_frame_idx < len(self.correct_frames) - 1:
            self.current_frame_idx += 1
            self.update_progress_text()
            self.good_match_start_time = None  # Reset auto-advance timer

        else:
            # Reached the end of the frames
            self.handle_session_complete()





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

    def load_correct_frames(self):
        cap = cv2.VideoCapture(self.correct_video_path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Video Error", f"Could not open video: {self.correct_video_path}")
            self.close()
            return

        self.correct_frames = []
        all_frames = []

        # First pass: collect all frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()

        # Detect when movement starts (when hands move up)
        movement_start_frame = 0
        buffer_frames = 5  # Number of frames to include before movement detection
        with mp.solutions.holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                static_image_mode=False
        ) as holistic:
            for i, frame in enumerate(all_frames):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)

                if results.pose_landmarks:
                    # Check both hand landmarks and pose wrist landmarks
                    hand_positions = []

                    # Left hand
                    if results.left_hand_landmarks:
                        hand_positions.append(results.left_hand_landmarks.landmark[0].y)  # Wrist
                    elif results.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.LEFT_WRIST]:
                        hand_positions.append(
                            results.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.LEFT_WRIST].y)

                    # Right hand
                    if results.right_hand_landmarks:
                        hand_positions.append(results.right_hand_landmarks.landmark[0].y)  # Wrist
                    elif results.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_WRIST]:
                        hand_positions.append(
                            results.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_WRIST].y)

                    # If we have any hand positions, check if they're in the upper part of the frame
                    if hand_positions and any(y < 0.7 for y in hand_positions):
                        movement_start_frame = max(0, i - buffer_frames)
                        break

        # Second pass: process only frames from movement_start_frame onwards with higher confidence
        with mp.solutions.holistic.Holistic(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
                static_image_mode=False
        ) as holistic:
            for frame in all_frames[movement_start_frame:]:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)

                annotated_frame = self.draw_reference_landmarks(frame.copy(), results)

                self.correct_frames.append({
                    'frame': frame,
                    'landmarks': results,
                    'processed': annotated_frame
                })

        # If no movement was detected but we have frames, use them all
        if not self.correct_frames and all_frames:
            print("No clear movement detected - using all available frames")
            with mp.solutions.holistic.Holistic(
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                    static_image_mode=False
            ) as holistic:
                for frame in all_frames:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(rgb_frame)
                    annotated_frame = self.draw_reference_landmarks(frame.copy(), results)
                    self.correct_frames.append({
                        'frame': frame,
                        'landmarks': results,
                        'processed': annotated_frame
                    })

        print(f"Loaded {len(self.correct_frames)} frames for practice")
        self.update_progress_text()

    def draw_reference_landmarks(self, frame, results):
        """Draw reference landmarks including hands"""
        annotated_frame = frame.copy()

        # Draw pose
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0),  # Green for reference
                    thickness=2
                )
            )

        # Draw hands
        if results.left_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_frame,
                results.left_hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0),  # Green for reference
                    thickness=2
                )
            )

        if results.right_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_frame,
                results.right_hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0),  # Green for reference
                    thickness=2
                )
            )

        return annotated_frame

    def update_progress_text(self):
        if self.correct_frames:
            progress_percent = ((self.current_frame_idx + 1) / len(self.correct_frames)) * 100
            progress_text = (
                f"Frame {self.current_frame_idx + 1} of {len(self.correct_frames)} - "
                f"{progress_percent:.1f}% complete"
            )
            self.progress_label.setText(progress_text)

    def process_user_frame(self, frame, user_results, correct_landmarks):
        display_frame = frame.copy()

        if user_results.pose_landmarks and correct_landmarks.pose_landmarks:
            # Get key points for alignment (shoulders and face)
            user_left_shoulder = user_results.pose_landmarks.landmark[
                mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER]
            user_right_shoulder = user_results.pose_landmarks.landmark[
                mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER]
            user_nose = user_results.pose_landmarks.landmark[
                mp.solutions.holistic.PoseLandmark.NOSE]

            correct_left_shoulder = correct_landmarks.pose_landmarks.landmark[
                mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER]
            correct_right_shoulder = correct_landmarks.pose_landmarks.landmark[
                mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER]
            correct_nose = correct_landmarks.pose_landmarks.landmark[
                mp.solutions.holistic.PoseLandmark.NOSE]

            # Calculate alignment points
            user_reference_points = np.array([
                [user_left_shoulder.x, user_left_shoulder.y],
                [user_right_shoulder.x, user_right_shoulder.y],
                [user_nose.x, user_nose.y]
            ])

            correct_reference_points = np.array([
                [correct_left_shoulder.x, correct_left_shoulder.y],
                [correct_right_shoulder.x, correct_right_shoulder.y],
                [correct_nose.x, correct_nose.y]
            ])

            # Calculate transformation matrix
            transform = self.get_similarity_transform(correct_reference_points, user_reference_points)

            # 1. Draw the correct pose (blue) - unchanged
            self.draw_complete_aligned_pose(display_frame, correct_landmarks, transform)

            # 2. Draw user's arms with color feedback
            self.draw_user_arms_with_feedback(display_frame, user_results, correct_landmarks, transform)

            # 3. Draw user's hands with detailed finger feedback
            if user_results.left_hand_landmarks:
                self.draw_hand_with_feedback(
                    display_frame,
                    user_results.left_hand_landmarks,
                    correct_landmarks.left_hand_landmarks if correct_landmarks.left_hand_landmarks else None,
                    transform,
                    'left'
                )

            if user_results.right_hand_landmarks:
                self.draw_hand_with_feedback(
                    display_frame,
                    user_results.right_hand_landmarks,
                    correct_landmarks.right_hand_landmarks if correct_landmarks.right_hand_landmarks else None,
                    transform,
                    'right'
                )

        return display_frame

    def draw_user_arms_with_feedback(self, frame, user_results, correct_landmarks, transform):
        frame_height, frame_width = frame.shape[:2]

        # Calculate thresholds based on auto-advance settings
        excellent_thresh = self.match_threshold / 1000  # Scale to 0-0.1 range
        good_thresh = excellent_thresh * 1.5
        fair_thresh = excellent_thresh * 2

        # Arm connections to check
        arm_connections = [
            (mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER, mp.solutions.holistic.PoseLandmark.LEFT_ELBOW),
            (mp.solutions.holistic.PoseLandmark.LEFT_ELBOW, mp.solutions.holistic.PoseLandmark.LEFT_WRIST),
            (mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER, mp.solutions.holistic.PoseLandmark.RIGHT_ELBOW),
            (mp.solutions.holistic.PoseLandmark.RIGHT_ELBOW, mp.solutions.holistic.PoseLandmark.RIGHT_WRIST)
        ]

        for start_idx, end_idx in arm_connections:
            # Get landmarks
            user_start = user_results.pose_landmarks.landmark[start_idx]
            user_end = user_results.pose_landmarks.landmark[end_idx]
            correct_start = correct_landmarks.pose_landmarks.landmark[start_idx]
            correct_end = correct_landmarks.pose_landmarks.landmark[end_idx]

            # Transform correct points
            aligned_start = transform @ np.array([correct_start.x, correct_start.y, 1])
            aligned_end = transform @ np.array([correct_end.x, correct_end.y, 1])

            # Calculate errors
            start_error = np.sqrt((user_start.x - aligned_start[0]) ** 2 + (user_start.y - aligned_start[1]) ** 2)
            end_error = np.sqrt((user_end.x - aligned_end[0]) ** 2 + (user_end.y - aligned_end[1]) ** 2)
            avg_error = (start_error + end_error) / 2

            # Determine color based on auto-advance thresholds
            if avg_error < excellent_thresh:  # Excellent match
                color = (0, 255, 0)  # Green
                thickness = 2
            elif avg_error < good_thresh:  # Good match
                color = (0, 255, 255)  # Yellow
                thickness = 3
            elif avg_error < fair_thresh:  # Fair match
                color = (0, 165, 255)  # Orange
                thickness = 4
            else:  # Poor match
                color = (0, 0, 255)  # Red
                thickness = 5

            # Draw connection
            start_point = (int(user_start.x * frame_width), int(user_start.y * frame_height))
            end_point = (int(user_end.x * frame_width), int(user_end.y * frame_height))

            cv2.line(frame, start_point, end_point, color, thickness)

            # Draw landmarks - larger circles for worse matches
            cv2.circle(frame, start_point, thickness + 2, color, -1)
            cv2.circle(frame, end_point, thickness + 2, color, -1)

            # Add error text for poor matches
            if avg_error >= fair_thresh:
                mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
                cv2.putText(frame, f"{avg_error * 100:.0f}%",
                            (mid_point[0] + 5, mid_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def draw_hand_with_feedback(self, frame, user_hand, correct_hand, transform, hand_type):
        """Draw user's hand with detailed finger feedback"""
        frame_height, frame_width = frame.shape[:2]

        if not user_hand:
            return

        # Define finger connections and important points
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
        finger_bases = [2, 5, 9, 13, 17]  # Bases of each finger
        palm_landmarks = [0, 1, 5, 9, 13, 17]  # Wrist and palm points

        # Draw palm connections first
        for i in range(len(palm_landmarks) - 1):
            start_idx = palm_landmarks[i]
            end_idx = palm_landmarks[i + 1]

            start_point = (int(user_hand.landmark[start_idx].x * frame_width),
                           int(user_hand.landmark[start_idx].y * frame_height))
            end_point = (int(user_hand.landmark[end_idx].x * frame_width),
                         int(user_hand.landmark[end_idx].y * frame_height))

            # Get color for palm area
            color = self.get_finger_feedback_color(user_hand, correct_hand, start_idx, end_idx, transform)
            cv2.line(frame, start_point, end_point, color, 2)

        # Draw each finger separately with feedback
        for tip_idx, base_idx in zip(finger_tips, finger_bases):
            # Draw finger from base to tip
            for i in range(base_idx, tip_idx):
                start_idx = i
                end_idx = i + 1

                start_point = (int(user_hand.landmark[start_idx].x * frame_width),
                               int(user_hand.landmark[start_idx].y * frame_height))
                end_point = (int(user_hand.landmark[end_idx].x * frame_width),
                             int(user_hand.landmark[end_idx].y * frame_height))

                # Get color for this finger segment
                color = self.get_finger_feedback_color(user_hand, correct_hand, start_idx, end_idx, transform)
                cv2.line(frame, start_point, end_point, color, 2)

                # Draw landmarks
                cv2.circle(frame, start_point, 3, color, -1)
                cv2.circle(frame, end_point, 3, color, -1)

    def get_finger_feedback_color(self, user_hand, correct_hand, start_idx, end_idx, transform):
        """Determine color for a finger segment based on auto-advance thresholds"""
        if not correct_hand:
            return (0, 255, 0)  # Default to green if no comparison available

        # Get points
        user_start = (user_hand.landmark[start_idx].x, user_hand.landmark[start_idx].y)
        user_end = (user_hand.landmark[end_idx].x, user_hand.landmark[end_idx].y)

        # Transform correct points
        correct_start = (correct_hand.landmark[start_idx].x, correct_hand.landmark[start_idx].y)
        correct_end = (correct_hand.landmark[end_idx].x, correct_hand.landmark[end_idx].y)

        aligned_start = transform @ np.array([correct_start[0], correct_start[1], 1])
        aligned_end = transform @ np.array([correct_end[0], correct_end[1], 1])

        # Calculate errors
        start_error = np.sqrt((user_start[0] - aligned_start[0]) ** 2 + (user_start[1] - aligned_start[1]) ** 2)
        end_error = np.sqrt((user_end[0] - aligned_end[0]) ** 2 + (user_end[1] - aligned_end[1]) ** 2)
        avg_error = (start_error + end_error) / 2

        # Set thresholds based on auto-advance settings
        excellent_thresh = 0.03 * (self.match_threshold / 80)  # Scale threshold
        good_thresh = 0.06 * (self.match_threshold / 80)
        fair_thresh = 0.09 * (self.match_threshold / 80)

        # Determine color
        if avg_error < excellent_thresh:
            return (0, 255, 0)  # Green
        elif avg_error < good_thresh:
            return (0, 255, 255)  # Yellow
        elif avg_error < fair_thresh:
            return (0, 165, 255)  # Orange
        else:
            return (0, 0, 255)  # Red

    def draw_complete_aligned_pose(self, frame, landmarks, transform):
        """Draw the complete aligned pose including hands"""
        frame_height, frame_width = frame.shape[:2]

        # Draw pose
        if landmarks.pose_landmarks:
            for connection in mp.solutions.holistic.POSE_CONNECTIONS:
                start_idx, end_idx = connection

                # Transform correct pose points
                correct_start = landmarks.pose_landmarks.landmark[start_idx]
                correct_end = landmarks.pose_landmarks.landmark[end_idx]

                # Apply transformation matrix
                aligned_start = transform @ np.array([correct_start.x, correct_start.y, 1])
                aligned_end = transform @ np.array([correct_end.x, correct_end.y, 1])

                # Convert to pixel coordinates
                start_point = (int(aligned_start[0] * frame_width),
                               int(aligned_start[1] * frame_height))
                end_point = (int(aligned_end[0] * frame_width),
                             int(aligned_end[1] * frame_height))

                # Draw semi-transparent correct pose (blue)
                overlay = frame.copy()
                cv2.line(overlay, start_point, end_point, (255, 0, 0), 3)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Draw left hand if present
        if landmarks.left_hand_landmarks:
            self.draw_aligned_hand(frame, landmarks.left_hand_landmarks, transform)

        # Draw right hand if present
        if landmarks.right_hand_landmarks:
            self.draw_aligned_hand(frame, landmarks.right_hand_landmarks, transform)

    def draw_aligned_hand(self, frame, hand_landmarks, transform):
        """Draw a single hand aligned with the transformation"""
        frame_height, frame_width = frame.shape[:2]

        # Draw hand connections
        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection

            # Transform hand points
            start = hand_landmarks.landmark[start_idx]
            end = hand_landmarks.landmark[end_idx]

            # Apply transformation
            aligned_start = transform @ np.array([start.x, start.y, 1])
            aligned_end = transform @ np.array([end.x, end.y, 1])

            # Convert to pixel coordinates
            start_point = (int(aligned_start[0] * frame_width),
                           int(aligned_start[1] * frame_height))
            end_point = (int(aligned_end[0] * frame_width),
                         int(aligned_end[1] * frame_height))

            # Draw semi-transparent correct hand (blue)
            overlay = frame.copy()
            cv2.line(overlay, start_point, end_point, (255, 0, 0), 2)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Draw hand landmarks
        for landmark in hand_landmarks.landmark:
            aligned_point = transform @ np.array([landmark.x, landmark.y, 1])
            center = (int(aligned_point[0] * frame_width),
                      int(aligned_point[1] * frame_height))
            cv2.circle(frame, center, 2, (255, 0, 0), -1)

    def get_similarity_transform(self, src_points, dst_points):
        """Improved transformation that preserves body proportions better"""
        # Center the points
        src_center = np.mean(src_points, axis=0)
        dst_center = np.mean(dst_points, axis=0)

        centered_src = src_points - src_center
        centered_dst = dst_points - dst_center

        # Calculate scale using both dimensions
        src_scale = np.mean(np.std(centered_src, axis=0))
        dst_scale = np.mean(np.std(centered_dst, axis=0))
        scale = dst_scale / src_scale if src_scale != 0 else 1.0

        # Calculate rotation using Kabsch algorithm with all points
        H = centered_src.T @ centered_dst
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Build transformation matrix
        transform = np.eye(3)
        transform[:2, :2] = scale * R
        transform[:2, 2] = dst_center - scale * (R @ src_center)

        return transform

    def compare_poses(self, user_results, correct_landmarks):
        """Compare poses with focus on hands and arms for sign language"""
        feedback = []
        detailed_feedback = []

        # Get transformation using upper body reference points
        user_ref_points = self.get_reference_points(user_results)
        correct_ref_points = self.get_reference_points(correct_landmarks)

        if user_ref_points is None or correct_ref_points is None:
            return 0, ["Cannot detect upper body for comparison. Ensure your shoulders are visible"]

        transform = self.get_similarity_transform(correct_ref_points, user_ref_points)

        # Hand comparison (primary focus)
        hand_score, hand_feedback = self.compare_hands(user_results, correct_landmarks, transform)
        feedback.extend(hand_feedback)
        detailed_feedback.extend(hand_feedback)

        # Arm comparison (secondary)
        arm_score, arm_feedback = self.compare_arms(user_results, correct_landmarks, transform)
        feedback.extend(arm_feedback)

        # Body orientation comparison
        body_score, body_feedback = self.compare_body_orientation(user_results, correct_landmarks)
        feedback.extend(body_feedback)

        # Calculate overall score (weighted)
        total_score = (hand_score * 0.7) + (arm_score * 0.2) + (body_score * 0.1)

        # Combine feedback, removing duplicates
        unique_feedback = []
        seen_messages = set()
        for item in feedback:
            if item not in seen_messages:
                seen_messages.add(item)
                unique_feedback.append(item)

        return total_score, unique_feedback[:5]  # Return max 5 feedback items

    def compare_arms(self, user_results, correct_landmarks, transform):
        """Compare arm positions as fallback when hand data isn't good"""
        if not (user_results.pose_landmarks and correct_landmarks.pose_landmarks):
            return 0, []

        arm_connections = [
            (mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER, mp.solutions.holistic.PoseLandmark.LEFT_ELBOW),
            (mp.solutions.holistic.PoseLandmark.LEFT_ELBOW, mp.solutions.holistic.PoseLandmark.LEFT_WRIST),
            (mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER, mp.solutions.holistic.PoseLandmark.RIGHT_ELBOW),
            (mp.solutions.holistic.PoseLandmark.RIGHT_ELBOW, mp.solutions.holistic.PoseLandmark.RIGHT_WRIST)
        ]

        arm_scores = []
        feedback = []

        for start_idx, end_idx in arm_connections:
            # Get landmarks
            user_start = user_results.pose_landmarks.landmark[start_idx]
            user_end = user_results.pose_landmarks.landmark[end_idx]
            correct_start = correct_landmarks.pose_landmarks.landmark[start_idx]
            correct_end = correct_landmarks.pose_landmarks.landmark[end_idx]

            # Transform correct points
            aligned_start = transform @ np.array([correct_start.x, correct_start.y, 1])
            aligned_end = transform @ np.array([correct_end.x, correct_end.y, 1])

            # Calculate errors
            start_error = np.sqrt((user_start.x - aligned_start[0]) ** 2 +
                                  (user_start.y - aligned_start[1]) ** 2)
            end_error = np.sqrt((user_end.x - aligned_end[0]) ** 2 +
                                (user_end.y - aligned_end[1]) ** 2)
            avg_error = (start_error + end_error) / 2

            # Score based on auto-advance threshold
            connection_score = max(0, 100 - (avg_error * 300))
            arm_scores.append(connection_score)

            # Only give feedback for significant errors
            if avg_error > 0.15:
                part_name = self.get_landmark_name(start_idx)
                feedback.append(f"Adjust your {part_name} position")

        if arm_scores:
            return sum(arm_scores) / len(arm_scores), feedback
        return 0, feedback

    def get_position_direction(self, error_vector):
        """Determine the primary direction of position error"""
        x, y = error_vector[0], error_vector[1]

        if abs(x) > abs(y):
            return 'left' if x > 0 else 'right'
        else:
            return 'high' if y > 0 else 'low'

    def compare_body_orientation(self, user_results, correct_landmarks):
        """Compare the overall body orientation (shoulders and torso)"""
        if not (user_results.pose_landmarks and correct_landmarks.pose_landmarks):
            return 0, []

        feedback = []
        score = 100  # Start with perfect score

        # Key points for body orientation
        ref_points = [
            mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER,
            mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER,
            mp.solutions.holistic.PoseLandmark.LEFT_HIP,
            mp.solutions.holistic.PoseLandmark.RIGHT_HIP
        ]

        errors = []

        for point in ref_points:
            user_pos = user_results.pose_landmarks.landmark[point]
            correct_pos = correct_landmarks.pose_landmarks.landmark[point]

            error = np.sqrt((user_pos.x - correct_pos.x) ** 2 +
                            (user_pos.y - correct_pos.y) ** 2)
            errors.append(error)

            # Deduct points for significant errors
            if error > 0.1:
                score -= 15
                part_name = self.get_landmark_name(point)
                if part_name and part_name not in feedback:
                    feedback.append(f"Adjust your {part_name} position")

        # Check shoulder alignment
        user_left_shoulder = user_results.pose_landmarks.landmark[
            mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER]
        user_right_shoulder = user_results.pose_landmarks.landmark[
            mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER]

        correct_left_shoulder = correct_landmarks.pose_landmarks.landmark[
            mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER]
        correct_right_shoulder = correct_landmarks.pose_landmarks.landmark[
            mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER]

        # Compare shoulder tilt
        user_tilt = user_right_shoulder.y - user_left_shoulder.y
        correct_tilt = correct_right_shoulder.y - correct_left_shoulder.y
        tilt_diff = abs(user_tilt - correct_tilt)

        if tilt_diff > 0.05:
            score -= 20
            if tilt_diff > 0.1:
                feedback.append("Straighten your shoulders - they appear uneven")

        return max(0, score), feedback

    def compare_hands(self, user_results, correct_landmarks, transform):
        """Compare hand positions and finger shapes with focus on sign language"""
        feedback = []
        hand_score = 0
        max_hand_score = 0

        # Left hand comparison - more robust checks
        left_score = 0
        if (hasattr(user_results, 'left_hand_landmarks')) and user_results.left_hand_landmarks:
            correct_left = getattr(correct_landmarks, 'left_hand_landmarks', None)
            if correct_left and len(correct_left.landmark) == 21:  # Ensure full hand detection
                score, hand_feedback = self.compare_single_hand(
                    user_results.left_hand_landmarks,
                    correct_left,
                    "left hand",
                    transform
                )
                left_score = score
                max_hand_score += 50
                feedback.extend(hand_feedback)
            else:
                # If no correct hand data, still compare basic position
                wrist_pos = self.estimate_wrist_position(correct_landmarks, 'left')
                if wrist_pos:
                    score, hand_feedback = self.compare_hand_position_only(
                        user_results.left_hand_landmarks,
                        wrist_pos,
                        "left hand",
                        transform
                    )
                    left_score = score * 0.7  # Reduced weight for partial comparison
                    max_hand_score += 35
                    feedback.extend(hand_feedback)

            # Right hand comparison - same logic
        right_score = 0
        if (hasattr(user_results, 'right_hand_landmarks') and user_results.right_hand_landmarks):
            correct_right = getattr(correct_landmarks, 'right_hand_landmarks', None)
            if correct_right and len(correct_right.landmark) == 21:  # Ensure full hand detection
                score, hand_feedback = self.compare_single_hand(
                    user_results.right_hand_landmarks,
                    correct_right,
                    "right hand",
                    transform
                )
                right_score = score
                max_hand_score += 50
                feedback.extend(hand_feedback)
            else:
                # If no correct hand data, still compare basic position
                wrist_pos = self.estimate_wrist_position(correct_landmarks, 'right')
                if wrist_pos:
                    score, hand_feedback = self.compare_hand_position_only(
                        user_results.right_hand_landmarks,
                        wrist_pos,
                        "right hand",
                        transform
                    )
                    right_score = score * 0.7  # Reduced weight for partial comparison
                    max_hand_score += 35
                    feedback.extend(hand_feedback)

        # Calculate total hand score (scale to 70 points total)
        if max_hand_score > 0:
            hand_score = (left_score + right_score) / max_hand_score * 70
        else:
            hand_score = 0


        return hand_score, feedback

    def estimate_wrist_position(self, landmarks, hand_type):
        """Estimate wrist position from pose when hand landmarks aren't available"""
        if not landmarks or not landmarks.pose_landmarks:
            return None

        try:
            if hand_type == 'left':
                wrist = landmarks.pose_landmarks.landmark[
                    mp.solutions.holistic.PoseLandmark.LEFT_WRIST]
            else:
                wrist = landmarks.pose_landmarks.landmark[
                    mp.solutions.holistic.PoseLandmark.RIGHT_WRIST]
            return (wrist.x, wrist.y)
        except:
            return None

    def compare_hand_position_only(self, user_hand, correct_wrist_pos, hand_name, transform):
        """Simplified comparison when only wrist position is known"""
        if not correct_wrist_pos:
            return 0, [f"Cannot determine {hand_name} position"]

        # Transform correct wrist position
        aligned_pos = transform @ np.array([correct_wrist_pos[0], correct_wrist_pos[1], 1])

        # Get user wrist position (landmark 0 is wrist)
        user_pos = user_hand.landmark[0]
        error = np.sqrt((user_pos.x - aligned_pos[0]) ** 2 + (user_pos.y - aligned_pos[1]) ** 2)

        # Calculate score (more lenient for position-only comparison)
        score = max(0, 100 - (error * 150))

        feedback = []
        if error > 0.15:
            feedback.append(f"Move your {hand_name} closer to the correct position")

        return score, feedback

    def compare_single_hand(self, user_hand, correct_hand, hand_name, transform):
        """Generate clear, actionable hand feedback"""
        if correct_hand is None:
            return 0, [f"Please make sure your {hand_name} is visible"]

        feedback = []

        # Check wrist position first (most important)
        wrist_error = self.get_landmark_error(user_hand, correct_hand, 0, transform)
        if wrist_error > 0.15:
            x_diff, y_diff = self.get_error_direction(user_hand, correct_hand, 0, transform)
            direction = []
            if abs(x_diff) > 0.05:
                direction.append("left" if x_diff > 0 else "right")
            if abs(y_diff) > 0.05:
                direction.append("down" if y_diff > 0 else "up")
            if direction:
                feedback.append(f"Move your {hand_name} wrist {' and '.join(direction)}")

        # Check finger tips
        finger_tips = [
            (4, "thumb tip"),
            (8, "index finger"),
            (12, "middle finger"),
            (16, "ring finger"),
            (20, "pinky finger")
        ]

        for tip_idx, tip_name in finger_tips:
            error = self.get_landmark_error(user_hand, correct_hand, tip_idx, transform)
            if error > 0.1:
                x_diff, y_diff = self.get_error_direction(user_hand, correct_hand, tip_idx, transform)

                direction = []
                if abs(x_diff) > 0.05:
                    direction.append("left" if x_diff > 0 else "right")
                if abs(y_diff) > 0.05:
                    direction.append("down" if y_diff > 0 else "up")

                if direction:
                    feedback.append(f"Adjust your {hand_name} {tip_name} {' and '.join(direction)}")

        # Check palm orientation
        palm_error = self.get_palm_orientation_error(user_hand, correct_hand, transform)
        if palm_error > 0.2:
            feedback.append(f"Rotate your {hand_name} palm {'inward' if palm_error > 0 else 'outward'}")

        return min(100, 100 - (wrist_error * 150)), feedback[:3]  # Return max 3 feedback items

    def get_landmark_error(self, user_hand, correct_hand, idx, transform):
        """Calculate error for a specific landmark"""
        user_pos = user_hand.landmark[idx]
        correct_pos = correct_hand.landmark[idx]
        aligned_pos = transform @ np.array([correct_pos.x, correct_pos.y, 1])
        return np.sqrt((user_pos.x - aligned_pos[0]) ** 2 + (user_pos.y - aligned_pos[1]) ** 2)

    def get_error_direction(self, user_hand, correct_hand, idx, transform):
        """Get x and y error direction"""
        user_pos = user_hand.landmark[idx]
        correct_pos = correct_hand.landmark[idx]
        aligned_pos = transform @ np.array([correct_pos.x, correct_pos.y, 1])
        return (user_pos.x - aligned_pos[0], user_pos.y - aligned_pos[1])

    def get_palm_orientation_error(self, user_hand, correct_hand, transform):
        """Calculate palm rotation error"""
        # Vector from wrist to middle finger base
        user_vec = np.array([
            user_hand.landmark[9].x - user_hand.landmark[0].x,
            user_hand.landmark[9].y - user_hand.landmark[0].y
        ])

        correct_vec = np.array([
            correct_hand.landmark[9].x - correct_hand.landmark[0].x,
            correct_hand.landmark[9].y - correct_hand.landmark[0].y
        ])

        # Normalize vectors
        user_vec = user_vec / np.linalg.norm(user_vec)
        correct_vec = correct_vec / np.linalg.norm(correct_vec)

        # Calculate angle difference
        return np.arccos(np.clip(np.dot(user_vec, correct_vec), -1.0, 1.0))

    def get_sign_language_feedback(self, error_type, body_part):
        """Generate specific sign language feedback with visual cues"""
        position_feedback = {
            'high': "too high",
            'low': "too low",
            'left': "too far left",
            'right': "too far right",
            'forward': "too far forward",
            'back': "too far back"
        }

        finger_feedback = {
            'thumb': {
                'extended': "Your thumb should be extended outward",
                'bent': "Bend your thumb more",
                'touching': "Your thumb should touch your {} finger",
                'spread': "Spread your thumb wider from your hand"
            },
            'index': {
                'straight': "Keep your index finger straight",
                'bent': "Bend your index finger at the knuckle",
                'together': "Your index finger should touch your {} finger",
                'apart': "Separate your index finger from your {} finger"
            },
            'middle': {
                'straight': "Middle finger should be straight",
                'bent': "Bend your middle finger more",
                'arched': "Create more arch in your middle finger"
            },
            'ring': {
                'curled': "Curl your ring finger more",
                'extended': "Extend your ring finger fully"
            },
            'pinky': {
                'extended': "Keep your pinky finger extended",
                'bent': "Your pinky should be bent more",
                'apart': "Separate your pinky from your ring finger"
            }
        }

        handshape_feedback = [
            "Maintain a firm handshape (not too loose)",
            "Keep your fingers together unless the sign requires separation",
            "Your palm should face {} for this sign",
            "Relax your hand slightly - tension is visible",
            "The handshape should be {} for this sign"
        ]

        movement_feedback = {
            'speed': {
                'fast': "Move faster for this sign",
                'slow': "Slow down your movement",
                'jerky': "Make the movement smoother"
            },
            'path': {
                'straight': "Move in a straight line",
                'circular': "Make a circular motion",
                'zigzag': "Follow a zigzag path",
                'updown': "Move up then down"
            },
            'direction': {
                'inward': "Move toward your body",
                'outward': "Move away from your body",
                'left': "Move more to your left",
                'right': "Move more to your right"
            }
        }

        # Generate feedback based on error type
        if error_type == 'position':
            direction = self.get_position_direction(body_part[1])
            return f"Your {body_part[0]} is {position_feedback.get(direction, 'misaligned')}"

        elif error_type == 'finger':
            finger = body_part[0]
            state = body_part[1]
            if finger in finger_feedback and state in finger_feedback[finger]:
                return finger_feedback[finger][state]
            return f"Adjust your {finger} position"

        elif error_type == 'handshape':
            return random.choice(handshape_feedback)

        elif error_type == 'movement':
            move_type = body_part[0]
            quality = body_part[1]
            if move_type in movement_feedback and quality in movement_feedback[move_type]:
                return movement_feedback[move_type][quality]
            return "Adjust your movement"

        return "Adjust your sign"

    def get_landmark_name(self, idx):
        """Simplified to only return names for relevant body parts"""
        names = {
            11: "left shoulder",
            12: "right shoulder",
            13: "left elbow",
            14: "right elbow",
            15: "left wrist",
            16: "right wrist",
            17: "left pinky",
            18: "right pinky",
            19: "left index",
            20: "right index"
        }
        return names.get(idx, "")

    def get_reference_points(self, landmarks):
        """Get reference points using shoulders, elbows, and wrists for better arm proportion alignment"""
        if not hasattr(landmarks, 'pose_landmarks') or landmarks.pose_landmarks is None:
            return None

        try:
            # Get all upper body points for better proportion alignment
            points = [
                [landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER].x,
                 landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER].y],
                [landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER].x,
                 landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER].y],
                [landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.LEFT_ELBOW].x,
                 landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.LEFT_ELBOW].y],
                [landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_ELBOW].x,
                 landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_ELBOW].y],
                [landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.LEFT_WRIST].x,
                 landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.LEFT_WRIST].y],
                [landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_WRIST].x,
                 landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_WRIST].y],
                [landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.NOSE].x,
                 landmarks.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.NOSE].y]
            ]

            return np.array(points)
        except Exception as e:
            print(f"Error getting reference points: {e}")
            return None

    def update_label(self, label, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img))

    def prev_frame(self):
        """Move to previous frame and reset auto-advance tracking"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.update_progress_text()
            self.good_match_start_time = None  # Reset auto-advance timer

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.timer.isActive():
            self.timer.stop()
        if self.holistic:
            self.holistic.close()
        pygame.mixer.quit()  # Clean up audio
        event.accept()


class GoogleDriveManager:
    def __init__(self):
        self.creds = None
        self.service = None
        self.root_folder_id = None
        self.initialize_drive()
        self.find_root_folders()

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

    def find_root_folders(self):
        """Find the root folders we need (FinalProject)."""
        results = self.service.files().list(
            q="name='FinalProject' and mimeType='application/vnd.google-apps.folder' and 'root' in parents and trashed=false",
            pageSize=1,
            fields="files(id, name)"
        ).execute()

        folder = results.get('files', [None])[0]

        if folder:
            self.root_folder_id = folder['id']
            print(f"Found FinalProject folder with ID: {self.root_folder_id}")
        else:
            raise Exception("FinalProject folder not found in root.")

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


class VideoPlayerWidget(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.media_player = QMediaPlayer()
        self.media_player.setNotifyInterval(100)  # Update every 100ms
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Title label
        title_label = QLabel(self.title)
        title_label.setStyleSheet("""
            font-weight: bold;
            font-size: 16px;
            color: #4a4a4a;
            margin-bottom: 5px;
        """)
        layout.addWidget(title_label)

        # Video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(400, 300)
        self.media_player.setVideoOutput(self.video_widget)
        layout.addWidget(self.video_widget)

        # Control buttons
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(5, 5, 5, 5)

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_play_pause)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #6e8efb;
                border-radius: 4px;
                padding: 5px;
                min-width: 30px;
            }
            QPushButton:hover {
                background-color: #5a7df4;
            }
        """)

        self.stop_button = QPushButton()
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #6e8efb;
                border-radius: 4px;
                padding: 5px;
                min-width: 30px;
            }
            QPushButton:hover {
                background-color: #5a7df4;
            }
        """)

        # Position slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)
        self.position_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #d3d3d3;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 12px;
                margin: -4px 0;
                background: #6e8efb;
                border-radius: 6px;
            }
            QSlider::sub-page:horizontal {
                background: #6e8efb;
                border-radius: 3px;
            }
        """)

        # Speed control
        speed_container = QWidget()
        speed_layout = QHBoxLayout(speed_container)
        speed_layout.setContentsMargins(0, 0, 0, 0)

        speed_label = QLabel("Speed:")
        speed_label.setStyleSheet("color: #4a4a4a;")

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "0.75x", "1.0x (Normal)", "1.25x", "1.5x", "1.75x", "2.0x"])
        self.speed_combo.setCurrentText("1.0x (Normal)")
        self.speed_combo.currentTextChanged.connect(self.change_speed)
        self.speed_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
                min-width: 100px;
                color: black;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: black;
                selection-background-color: #6e8efb;
                selection-color: white;
            }
            QComboBox:hover {
                border: 1px solid #6e8efb;
            }
        """)

        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_combo)

        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(speed_container)

        layout.addLayout(control_layout)
        layout.addWidget(self.position_slider)

        # Connect media player signals
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.stateChanged.connect(self.state_changed)

    def change_speed(self, speed_text):
        """Change playback speed based on combo box selection."""
        try:
            speed = float(speed_text.split('x')[0].strip())
            self.media_player.setPlaybackRate(speed)
            print(f"Playback speed set to: {speed}x")
        except Exception as e:
            print(f"Error setting playback speed: {str(e)}")

    def play_video(self, file_path):
        """Play a video file."""
        if os.path.exists(file_path):
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
            self.speed_combo.setCurrentText("1.0x (Normal)")
            self.media_player.setPlaybackRate(1.0)
            self.media_player.play()

    def toggle_play_pause(self):
        """Toggle between play and pause."""
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            if self.media_player.mediaStatus() == QMediaPlayer.NoMedia:
                return
            self.media_player.play()

    def stop(self):
        """Stop playback."""
        self.media_player.stop()

    def set_position(self, position):
        """Set video position."""
        self.media_player.setPosition(position)

    def position_changed(self, position):
        """Update position slider."""
        self.position_slider.setValue(position)

    def duration_changed(self, duration):
        """Update duration slider range."""
        self.position_slider.setRange(0, duration)

    def state_changed(self, state):
        """Update play/pause button icon based on state."""
        if state == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))


class ProgressDashboard(QWidget):
    def __init__(self, user_data, parent=None):
        super().__init__(parent)
        self.user_data = user_data
        self.setup_ui()

    def refresh_data(self, new_user_data):
        """Refresh the dashboard with new data"""
        self.user_data = new_user_data
        # Clear old tabs
        self.tabs.clear()
        # Recreate tabs with updated data
        self.setup_overview_tab()
        self.setup_words_tab()
        self.setup_games_tab()

    def draw_stacked_bar_chart(self, size, data_series, labels, series_names, colors):
        """Draw a stacked bar chart with the given data"""
        pixmap = QPixmap(size)
        pixmap.fill(Qt.white)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Chart dimensions
        margin = 40
        chart_width = size.width() - 2 * margin
        chart_height = size.height() - 2 * margin

        # Find max value for scaling
        max_value = max([sum(values) for values in zip(*data_series)]) if data_series else 1

        # Draw grid lines
        pen = QPen(QColor(200, 200, 200))
        pen.setWidth(1)
        painter.setPen(pen)

        # Draw horizontal grid lines
        for i in range(5):
            y = int(size.height() - margin - (i * chart_height / 4))  # Explicitly convert to int
            painter.drawLine(margin, y, size.width() - margin, y)

            # Draw value label
            value = (i * max_value / 4)
            painter.setPen(Qt.black)
            painter.drawText(QRectF(0, y - 10, margin - 5, 20),
                             Qt.AlignRight, f"{value:.0f}")
            painter.setPen(pen)

        # Draw bars
        if data_series and labels:
            bar_width = chart_width / (len(labels) * 1.5)
            x_positions = [margin + i * (bar_width * 1.5) for i in range(len(labels))]

            for series_idx, series in enumerate(data_series):
                color = QColor(colors[series_idx % len(colors)])
                painter.setBrush(color)
                painter.setPen(Qt.NoPen)

                for i, value in enumerate(series):
                    # Calculate height for this segment
                    segment_height = (value / max_value) * chart_height

                    # Calculate y position (stack segments)
                    y_base = size.height() - margin
                    for prev_series in data_series[:series_idx]:
                        y_base -= (prev_series[i] / max_value) * chart_height

                    y = y_base - segment_height

                    # Draw bar segment
                    painter.drawRect(QRectF(x_positions[i], y, bar_width, segment_height))

                    # Draw label for first series
                    if series_idx == 0:
                        painter.setPen(Qt.black)
                        painter.drawText(QRectF(x_positions[i], size.height() - margin + 5,
                                                bar_width, 20), Qt.AlignCenter, labels[i])
                        painter.setPen(Qt.NoPen)

        # Draw axes
        pen = QPen(Qt.black)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(margin, size.height() - margin,
                         size.width() - margin, size.height() - margin)  # X-axis
        painter.drawLine(margin, size.height() - margin,
                         margin, margin)  # Y-axis

        # Draw legend
        legend_x = margin + 10
        legend_y = margin + 10
        for i, name in enumerate(series_names):
            color = QColor(colors[i % len(colors)])
            painter.setBrush(color)
            painter.drawRect(legend_x, legend_y + i * 20, 15, 15)
            painter.setPen(Qt.black)
            painter.drawText(legend_x + 20, legend_y + i * 20 + 12, name)

        painter.end()
        return pixmap

    def draw_line_chart(self, size, values, labels, y_label, color):
        """Draw a line chart with improved visibility and styling."""
        pixmap = QPixmap(size)
        pixmap.fill(Qt.white)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Chart dimensions with more padding
        margin = 50
        chart_width = size.width() - 2 * margin
        chart_height = size.height() - 2 * margin

        if not values:
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "No data available")
            painter.end()
            return pixmap

        max_value = max(values) if values else 1
        min_value = min(values) if values else 0

        # Add padding if all values are equal
        if max_value == min_value:
            if max_value == 0:
                max_value = 100
                min_value = 0
            else:
                max_value *= 1.1
                min_value *= 0.9

        # Draw grid lines - lighter and less prominent
        pen = QPen(QColor(220, 220, 220))
        pen.setWidth(1)
        painter.setPen(pen)

        # Draw horizontal grid lines and value labels
        for i in range(5):
            y = int(size.height() - margin - (i * chart_height / 4))
            painter.drawLine(margin, y, size.width() - margin, y)

            # Draw value label
            value = min_value + (i * (max_value - min_value) / 4)
            painter.setPen(Qt.black)
            painter.drawText(QRectF(0, y - 10, margin - 5, 20),
                             Qt.AlignRight, f"{value:.0f}")
            painter.setPen(pen)

        # Calculate points with more spacing
        points = []
        x_step = chart_width / max(1, (len(values) - 1))

        for i, value in enumerate(values):
            x = int(margin + (i * x_step))
            y = int(size.height() - margin -
                    ((value - min_value) / (max_value - min_value) * chart_height))
            points.append(QPointF(x, y))

        # Draw line - thicker and more visible
        if len(points) > 1:
            pen = QPen(QColor(color))
            pen.setWidth(4)  # Thicker line
            painter.setPen(pen)
            painter.drawPolyline(QPolygonF(points))

            # Draw points - larger and more visible
            painter.setBrush(QColor(color))
            for point in points:
                painter.drawEllipse(point, 6, 6)  # Larger points

                # Draw value label - more prominent
                idx = points.index(point)
                painter.setPen(Qt.black)
                painter.setFont(QFont("Arial", 10))
                painter.drawText(QRectF(point.x() - 25, point.y() - 25, 50, 20),
                                 Qt.AlignCenter, f"{values[idx]:.0f}")

        # Draw axes - thicker and more visible
        pen = QPen(Qt.black)
        pen.setWidth(2)
        painter.setPen(pen)

        # X-axis
        painter.drawLine(margin, size.height() - margin,
                         size.width() - margin, size.height() - margin)

        # Y-axis
        painter.drawLine(margin, size.height() - margin,
                         margin, margin)

        # Draw axis labels - larger and more readable
        painter.setFont(QFont("Arial", 12))

        # X-axis label
        painter.drawText(QRectF(0, size.height() - margin + 20,
                                size.width(), margin - 20),
                         Qt.AlignCenter, "Game Sessions")

        # Y-axis label (rotated)
        painter.save()
        painter.translate(15, size.height() / 2)
        painter.rotate(-90)
        painter.drawText(QRectF(0, 0, size.height(), margin - 5),
                         Qt.AlignCenter, y_label)
        painter.restore()

        # Chart title
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        painter.drawText(QRectF(0, 10, size.width(), 30),
                         Qt.AlignCenter, f"{y_label} Over Time")

        painter.end()
        return pixmap

    def draw_dual_axis_chart(self, size, primary_data, secondary_data, labels,
                             primary_label, secondary_label, primary_color, secondary_color):
        """Draw a dual-axis chart with improved visibility."""
        pixmap = QPixmap(size)
        pixmap.fill(Qt.white)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Chart dimensions with more padding
        margin = 50
        chart_width = size.width() - 2 * margin
        chart_height = size.height() - 2 * margin

        # Handle empty data case
        if not primary_data or not secondary_data:
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "No data available")
            painter.end()
            return pixmap

        # Find ranges for both datasets
        try:
            primary_max = max(primary_data) if primary_data else 1
            secondary_max = max(secondary_data) if secondary_data else 1

            # Ensure we don't divide by zero and add some padding
            if primary_max == 0:
                primary_max = 1
            if secondary_max == 0:
                secondary_max = 1

            primary_max *= 1.1  # Add 10% padding
            secondary_max *= 1.1
        except Exception as e:
            print(f"Error calculating chart ranges: {e}")
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "Error displaying data")
            painter.end()
            return pixmap

        # Draw grid lines - lighter and less prominent
        pen = QPen(QColor(220, 220, 220))
        pen.setWidth(1)
        painter.setPen(pen)

        # Draw horizontal grid lines and value labels (primary axis)
        for i in range(5):
            y = int(size.height() - margin - (i * chart_height / 4))
            painter.drawLine(margin, y, size.width() - margin, y)

            # Draw primary axis value
            value = (i * primary_max / 4)
            painter.setPen(Qt.black)
            painter.drawText(QRectF(0, y - 10, margin - 5, 20),
                             Qt.AlignRight, f"{value:.0f}")
            painter.setPen(pen)

        # Draw secondary axis on right side - more distinct
        secondary_pen = QPen(QColor(100, 100, 100))
        secondary_pen.setWidth(1)
        painter.setPen(secondary_pen)

        for i in range(5):
            y = int(size.height() - margin - (i * chart_height / 4))
            value = (i * secondary_max / 4)
            painter.drawText(QRectF(size.width() - margin + 5, y - 10, margin - 5, 20),
                             Qt.AlignLeft, f"{value:.0f}")
            # Draw small tick mark
            painter.drawLine(size.width() - margin, y, size.width() - margin + 5, y)

        # Calculate points for both datasets
        primary_points = []
        secondary_points = []
        x_step = chart_width / max(1, (len(primary_data) - 1))

        for i in range(len(primary_data)):
            x = int(margin + (i * x_step))

            # Primary data point
            y_primary = int(size.height() - margin -
                            (primary_data[i] / primary_max * chart_height))
            primary_points.append(QPointF(x, y_primary))

            # Secondary data point
            y_secondary = int(size.height() - margin -
                              (secondary_data[i] / secondary_max * chart_height))
            secondary_points.append(QPointF(x, y_secondary))

        # Draw primary data line - thick and solid
        if len(primary_points) > 1:
            pen = QPen(QColor(primary_color))
            pen.setWidth(4)
            painter.setPen(pen)
            painter.drawPolyline(QPolygonF(primary_points))

            # Draw primary data points - filled circles
            painter.setBrush(QColor(primary_color))
            for point in primary_points:
                painter.drawEllipse(point, 6, 6)

        # Draw secondary data line - thick and dashed
        if len(secondary_points) > 1:
            pen = QPen(QColor(secondary_color))
            pen.setWidth(4)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawPolyline(QPolygonF(secondary_points))

            # Draw secondary data points - open circles
            painter.setBrush(Qt.NoBrush)
            for point in secondary_points:
                painter.drawEllipse(point, 6, 6)

        # Draw axes - more prominent
        pen = QPen(Qt.black)
        pen.setWidth(2)
        painter.setPen(pen)

        # X-axis
        painter.drawLine(margin, size.height() - margin,
                         size.width() - margin, size.height() - margin)

        # Y-axes
        painter.drawLine(margin, size.height() - margin,
                         margin, margin)

        # Secondary Y-axis
        painter.drawLine(size.width() - margin, size.height() - margin,
                         size.width() - margin, margin)

        # Draw axis labels - larger and more readable
        painter.setFont(QFont("Arial", 12))

        # X-axis label
        painter.drawText(QRectF(0, size.height() - margin + 20,
                                size.width(), margin - 20),
                         Qt.AlignCenter, "Game Sessions")

        # Primary axis label (left)
        painter.save()
        painter.translate(15, size.height() / 2)
        painter.rotate(-90)
        painter.drawText(QRectF(0, 0, size.height(), margin - 5),
                         Qt.AlignCenter, primary_label)
        painter.restore()

        # Secondary axis label (right) - different color
        painter.save()
        painter.setPen(QColor(secondary_color))
        painter.translate(size.width() - 15, size.height() / 2)
        painter.rotate(-90)
        painter.drawText(QRectF(0, 0, size.height(), margin - 5),
                         Qt.AlignCenter, secondary_label)
        painter.restore()

        # Draw legend - more prominent and better positioned
        legend_x = margin + 10
        legend_y = margin + 10
        legend_font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(legend_font)

        # Primary legend
        painter.setPen(QColor(primary_color))
        painter.drawLine(legend_x, legend_y + 8, legend_x + 20, legend_y + 8)
        painter.drawText(QRectF(legend_x + 25, legend_y, 150, 20),
                         Qt.AlignLeft, primary_label)

        # Secondary legend
        painter.setPen(QColor(secondary_color))
        painter.drawLine(legend_x, legend_y + 28, legend_x + 20, legend_y + 28)
        painter.drawText(QRectF(legend_x + 25, legend_y + 20, 150, 20),
                         Qt.AlignLeft, secondary_label)

        # Chart title
        painter.setPen(Qt.black)
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        painter.drawText(QRectF(0, 10, size.width(), 30),
                         Qt.AlignCenter, f"{primary_label} vs {secondary_label}")

        painter.end()
        return pixmap

    def setup_ui(self):
        self.setWindowTitle("Progress Dashboard")

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

        # Header
        header = QLabel(f"📊 {self.user_data['username']}'s Progress Dashboard")
        header.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #4a6baf;
            padding-bottom: 10px;
            border-bottom: 2px solid #6e8efb;
        """)
        scroll_layout.addWidget(header)

        # Create tab widget
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

        # Add tabs
        self.setup_overview_tab()
        self.setup_words_tab()
        self.setup_games_tab()

        # Set the scroll content
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    def setup_overview_tab(self):
        """Setup the overview tab with main stats and charts"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        # Stats cards in a grid
        stats_grid = QGridLayout()
        stats_grid.setSpacing(15)

        # Learning Progress Card - Updated for new structure
        learning_card = QGroupBox("📚 Learning Progress")
        learning_card.setStyleSheet(self.get_card_style("#6e8efb"))
        learning_layout = QVBoxLayout(learning_card)

        practice_stats = self.user_data['stats']['practice_stats']
        total_sessions = practice_stats['sessions']
        words_attempted = practice_stats['words_attempted']
        words_correct = practice_stats['words_correct']
        accuracy = (words_correct / words_attempted * 100) if words_attempted > 0 else 0

        learning_layout.addWidget(QLabel(f"🔄 Practice Sessions: {total_sessions}"))
        learning_layout.addWidget(QLabel(f"📝 Words Attempted: {words_attempted}"))
        learning_layout.addWidget(QLabel(f"🎯 Accuracy: {accuracy:.1f}%"))
        stats_grid.addWidget(learning_card, 0, 0)

        # Game Progress Card - Updated for new structure
        game_card = QGroupBox("🎮 Game Progress")
        game_card.setStyleSheet(self.get_card_style("#a777e3"))
        game_layout = QVBoxLayout(game_card)

        game_stats = self.user_data['stats']['game_stats']
        quiz_stats = game_stats['quiz']
        matching_stats = game_stats['matching']
        bubble_stats = game_stats['bubble']

        game_layout.addWidget(QLabel(f"❓ Quiz Attempts: {quiz_stats['attempts']}"))
        game_layout.addWidget(QLabel(f"🃏 Matching Attempts: {matching_stats['attempts']}"))
        game_layout.addWidget(QLabel(f"🎯 Bubble Attempts: {bubble_stats['attempts']}"))
        stats_grid.addWidget(game_card, 0, 1)

        # Streak Card - Updated for new structure
        streak_card = QGroupBox("🔥 Streaks")
        streak_card.setStyleSheet(self.get_card_style("#4CAF50"))
        streak_layout = QVBoxLayout(streak_card)

        streak = practice_stats['current_streak']
        longest_streak = practice_stats['longest_streak']
        goal_streak = practice_stats['goal_streak']
        last_practiced = practice_stats['last_practiced'] or "Never"

        streak_layout.addWidget(QLabel(f"📅 Current Streak: {streak} days"))
        streak_layout.addWidget(QLabel(f"🏆 Longest Streak: {longest_streak} days"))
        streak_layout.addWidget(QLabel(f"✅ Goal Streak: {goal_streak} days"))
        streak_layout.addWidget(QLabel(f"🕒 Last Practiced: {last_practiced}"))
        stats_grid.addWidget(streak_card, 1, 0)

        # Words Learned Card - Updated for new structure
        words_card = QGroupBox("📖 Words Learned")
        words_card.setStyleSheet(self.get_card_style("#FF9800"))
        words_layout = QVBoxLayout(words_card)

        known_words = len(self.user_data.get('known_words', []))
        struggled_words = len(self.user_data.get('struggled_words', []))
        total_words = len(self.user_data.get('word_progress', {}))

        words_layout.addWidget(QLabel(f"✅ Known Words: {known_words}"))
        words_layout.addWidget(QLabel(f"❌️ Struggling With: {struggled_words}"))
        words_layout.addWidget(QLabel(f"📊 Total Tracked: {total_words}"))
        stats_grid.addWidget(words_card, 1, 1)

        layout.addLayout(stats_grid)

        # Charts Section - Updated to use new data structure
        charts_frame = QGroupBox("📈 Progress Charts")
        charts_frame.setStyleSheet(self.get_card_style("#6e8efb"))
        charts_layout = QVBoxLayout(charts_frame)

        # Weekly Activity Chart
        weekly_chart = self.create_weekly_progress_chart()
        charts_layout.addWidget(weekly_chart)

        # Accuracy Trend Chart
        accuracy_chart = self.create_accuracy_trend_chart()
        charts_layout.addWidget(accuracy_chart)

        # Words Learned Chart
        words_chart = self.create_words_learned_chart()
        charts_layout.addWidget(words_chart)

        layout.addWidget(charts_frame)
        self.tabs.addTab(tab, "Overview")

    def get_card_style(self, color):
        """Return consistent styling for cards"""
        return f"""
            QGroupBox {{
                border: 2px solid {color};
                border-radius: 8px;
                margin-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {color};
                font-weight: bold;
            }}
        """

    def create_weekly_progress_chart(self):
        """Create weekly progress chart using new data structure"""
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)

        # Title
        title = QLabel("Weekly Activity")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        chart_layout.addWidget(title)

        # Get all activities from practice and games
        all_activities = []

        # Add practice sessions
        practice_history = self.user_data.get('practice_history', [])
        all_activities.extend([{
            'type': 'practice',
            'date': session['date'],
            'duration': session.get('duration', 0),
            'words': session.get('words_attempted', 0)
        } for session in practice_history])

        # Add game sessions
        game_history = self.user_data.get('game_history', {})
        for game_type in game_history:
            all_activities.extend([{
                'type': game_type,
                'date': session['date'],
                'duration': session.get('duration', 0),
                'score': session.get('score', 0)
            } for session in game_history[game_type]])

        # Group by week
        weekly_data = {}
        for activity in all_activities:
            date = datetime.fromisoformat(activity['date']).date()
            year, week, _ = date.isocalendar()
            week_key = f"{year}-W{week:02d}"

            if week_key not in weekly_data:
                weekly_data[week_key] = {
                    'date': date,
                    'practice_sessions': 0,
                    'game_sessions': 0,
                    'total_duration': 0,
                    'words_practiced': 0,
                    'game_points': 0
                }

            if activity['type'] == 'practice':
                weekly_data[week_key]['practice_sessions'] += 1
                weekly_data[week_key]['total_duration'] += activity.get('duration', 0)
                weekly_data[week_key]['words_practiced'] += activity.get('words', 0)
            else:
                weekly_data[week_key]['game_sessions'] += 1
                weekly_data[week_key]['total_duration'] += activity.get('duration', 0)
                weekly_data[week_key]['game_points'] += activity.get('score', 0)

        # Sort by week and prepare data for chart
        sorted_weeks = sorted(weekly_data.keys())
        week_labels = [f"Week {i + 1}" for i in range(len(sorted_weeks))]
        practice_data = [weekly_data[week]['practice_sessions'] for week in sorted_weeks]
        game_data = [weekly_data[week]['game_sessions'] for week in sorted_weeks]

        # Create chart
        chart = QLabel()
        chart.setFixedHeight(200)
        chart.setStyleSheet("background-color: white; border: 1px solid #ddd;")

        if sorted_weeks:
            pixmap = self.draw_stacked_bar_chart(
                chart.size(),
                [practice_data, game_data],
                week_labels,
                ["Practice", "Games"],
                ["#6e8efb", "#4CAF50"]
            )
        else:
            pixmap = QPixmap(chart.size())
            pixmap.fill(Qt.white)
            painter = QPainter(pixmap)
            painter.drawText(chart.rect(), Qt.AlignCenter, "No activity data available")
            painter.end()

        chart.setPixmap(pixmap)
        chart_layout.addWidget(chart)

        return chart_widget

    def create_accuracy_trend_chart(self):
        """Create accuracy trend chart using new practice stats"""
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)

        # Title
        title = QLabel("Practice Accuracy Trend")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        chart_layout.addWidget(title)

        # Get practice history
        practice_history = self.user_data.get('practice_history', [])

        # Prepare accuracy data
        dates = []
        accuracies = []
        for session in practice_history:
            date = datetime.fromisoformat(session['date']).date()
            attempted = session.get('words_attempted', 0)
            correct = session.get('words_correct', 0)
            if attempted > 0:
                accuracy = (correct / attempted) * 100
                dates.append(date)
                accuracies.append(accuracy)

        # Create chart
        chart = QLabel()
        chart.setFixedHeight(200)
        chart.setStyleSheet("background-color: white; border: 1px solid #ddd;")

        if accuracies:
            date_labels = [date.strftime("%m/%d") for date in dates]
            pixmap = self.draw_line_chart(
                chart.size(),
                accuracies,
                date_labels,
                "Accuracy (%)",
                "#6e8efb"
            )
        else:
            pixmap = QPixmap(chart.size())
            pixmap.fill(Qt.white)
            painter = QPainter(pixmap)
            painter.drawText(chart.rect(), Qt.AlignCenter, "No practice data available")
            painter.end()

        chart.setPixmap(pixmap)
        chart_layout.addWidget(chart)

        return chart_widget

    def create_words_learned_chart(self):
        """Create chart showing words learned over time using new structure"""
        chart_widget = QWidget()
        layout = QVBoxLayout(chart_widget)

        # Title
        title = QLabel("Words Learned Over Time")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Get word progress data
        word_progress = self.user_data.get('word_progress', {})

        # Create timeline of when words were first known
        known_dates = []
        for word, data in word_progress.items():
            if 'first_known' in data and data['first_known']:
                known_dates.append(datetime.fromisoformat(data['first_known']).date())

        # Count words known by date
        date_counts = {}
        for date in known_dates:
            date_str = date.strftime("%Y-%m-%d")
            date_counts[date_str] = date_counts.get(date_str, 0) + 1

        # Create cumulative counts
        sorted_dates = sorted(date_counts.keys())
        cumulative_counts = []
        current_total = 0
        for date in sorted_dates:
            current_total += date_counts[date]
            cumulative_counts.append(current_total)

        # Create chart
        chart = QLabel()
        chart.setFixedHeight(200)
        chart.setStyleSheet("background-color: white; border: 1px solid #ddd;")

        if sorted_dates:
            date_labels = [datetime.strptime(d, "%Y-%m-%d").strftime("%m/%d") for d in sorted_dates]
            pixmap = self.draw_line_chart(
                chart.size(),
                cumulative_counts,
                date_labels,
                "Words Known",
                "#a777e3"
            )
        else:
            pixmap = QPixmap(chart.size())
            pixmap.fill(Qt.white)
            painter = QPainter(pixmap)
            painter.drawText(chart.rect(), Qt.AlignCenter, "No word learning data available")
            painter.end()

        chart.setPixmap(pixmap)
        layout.addWidget(chart)

        return chart_widget

    def setup_words_tab(self):
        """Setup the words tab with known words and progress using new structure"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Known words section
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

        known_words = self.user_data.get('known_words', [])
        if not known_words:
            empty_label = QLabel("You haven't mastered any words yet. Keep practicing!")
            empty_label.setStyleSheet("font-size: 14px; color: #666;")
            known_layout.addWidget(empty_label)
        else:
            words_layout = QGridLayout()
            words_layout.setSpacing(15)
            for i, word in enumerate(known_words):
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

        # Struggled words section
        struggled_frame = QGroupBox("️❌ Struggled Words")
        struggled_frame.setStyleSheet("""
            QGroupBox {
                border: 2px solid #FF0000;
                border-radius: 8px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #FF0000;
                font-weight: bold;
            }
        """)
        struggled_layout = QVBoxLayout(struggled_frame)

        struggled_words = self.user_data.get('struggled_words', [])
        if not struggled_words:
            empty_label = QLabel("No words marked as struggling with. Great job!")
            empty_label.setStyleSheet("font-size: 14px; color: #666;")
            struggled_layout.addWidget(empty_label)
        else:
            words_layout = QGridLayout()
            words_layout.setSpacing(10)
            for i, word in enumerate(struggled_words):
                word_label = QLabel(word)
                word_label.setStyleSheet("""
                    QLabel {
                        font-size: 16px; 
                        padding: 8px 12px;
                        background-color: #FFF3E0;
                        border-radius: 4px;
                    }
                """)
                words_layout.addWidget(word_label, i // 4, i % 4)
            struggled_layout.addLayout(words_layout)
        layout.addWidget(struggled_frame)

        # Word progress table
        progress_frame = QGroupBox("📊 Word Progress Details")
        progress_frame.setStyleSheet(known_frame.styleSheet().replace("4CAF50", "#6e8efb").replace("388E3C", "#4a6baf"))
        progress_layout = QVBoxLayout(progress_frame)

        word_progress = self.user_data.get('word_progress', {})
        if not word_progress:
            empty_label = QLabel("No word progress data available yet.")
            empty_label.setStyleSheet("font-size: 14px; color: #666;")
            progress_layout.addWidget(empty_label)
        else:
            table = QTableWidget()
            table.setColumnCount(5)
            table.setHorizontalHeaderLabels(["Word", "Attempts", "Correct", "Accuracy", "Last Practiced"])
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

            # Sort words by most recently practiced
            sorted_words = sorted(word_progress.items(),
                                  key=lambda x: x[1].get('last_attempted', ''),
                                  reverse=True)
            table.setRowCount(len(sorted_words))

            for row, (word, data) in enumerate(sorted_words):
                attempts = data.get('attempts', 0)
                correct = data.get('correct', 0)
                accuracy = (correct / attempts * 100) if attempts > 0 else 0
                last_practiced = data.get('last_attempted', 'Never')

                table.setItem(row, 0, QTableWidgetItem(word))
                table.setItem(row, 1, QTableWidgetItem(str(attempts)))
                table.setItem(row, 2, QTableWidgetItem(str(correct)))

                accuracy_item = QTableWidgetItem(f"{accuracy:.1f}%")
                if accuracy >= 80:
                    accuracy_item.setForeground(QColor(46, 125, 50))  # Green
                elif accuracy >= 50:
                    accuracy_item.setForeground(QColor(251, 192, 45))  # Yellow
                else:
                    accuracy_item.setForeground(QColor(198, 40, 40))  # Red
                table.setItem(row, 3, accuracy_item)

                table.setItem(row, 4, QTableWidgetItem(
                    last_practiced.split('T')[0] if 'T' in last_practiced else last_practiced))

            table.resizeColumnsToContents()
            progress_layout.addWidget(table)

        layout.addWidget(progress_frame)
        layout.addStretch()

        self.tabs.addTab(tab, "Words")

    def create_accuracy_chart(self):
        """Create accuracy trend chart using real user data"""
        chart_widget = QWidget()
        layout = QVBoxLayout(chart_widget)

        # Title
        title = QLabel("Accuracy Trend")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Get practice history (if available)
        practice_history = self.user_data.get('practice_history', [])

        # Calculate accuracy for each session
        dates = []
        accuracies = []
        for session in practice_history:
            date = datetime.fromisoformat(session['date']).date()
            attempted = session.get('words_attempted', 0)
            correct = session.get('words_correct', 0)
            if attempted > 0:
                accuracy = (correct / attempted) * 100
                dates.append(date)
                accuracies.append(accuracy)

        # Create chart
        chart = QLabel()
        chart.setFixedHeight(200)
        chart.setStyleSheet("background-color: white; border: 1px solid #ddd;")

        # Draw chart
        if accuracies:
            # Convert dates to strings for labels
            date_labels = [date.strftime("%m/%d") for date in dates]
            pixmap = self.draw_line_chart(chart.size(), accuracies, date_labels)
        else:
            pixmap = QPixmap(chart.size())
            pixmap.fill(Qt.white)
            painter = QPainter(pixmap)
            painter.drawText(chart.rect(), Qt.AlignCenter, "No accuracy data available")
            painter.end()

        chart.setPixmap(pixmap)
        layout.addWidget(chart)

        return chart_widget

    def draw_bar_chart(self, size, values, labels=None):
        """Draw a bar chart with the given values and labels"""
        pixmap = QPixmap(size)
        pixmap.fill(Qt.white)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Chart dimensions
        margin = 20
        chart_width = size.width() - 2 * margin
        chart_height = size.height() - 2 * margin

        # Find max value for scaling
        max_value = max(values) if values else 1

        # Draw grid lines
        pen = QPen(QColor(200, 200, 200))
        pen.setWidth(1)
        painter.setPen(pen)

        # Draw horizontal grid lines
        for i in range(5):
            y = size.height() - margin - (i * chart_height / 4)
            painter.drawLine(margin, y, size.width() - margin, y)

            # Draw value label
            value = (i * max_value / 4)
            painter.setPen(Qt.black)
            painter.drawText(QRectF(0, y - 10, margin - 5, 20),
                             Qt.AlignRight, f"{value:.0f}")
            painter.setPen(pen)

        # Draw bars
        bar_width = chart_width / (len(values) * 1.5)  # Leave space between bars
        color = QColor(74, 107, 175)  # Main blue color

        for i, value in enumerate(values):
            # Calculate bar dimensions
            bar_height = (value / max_value) * chart_height
            x = margin + i * (bar_width * 1.5)
            y = size.height() - margin - bar_height

            # Draw bar
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawRect(QRectF(x, y, bar_width, bar_height))

            # Draw value label
            painter.setPen(Qt.black)
            painter.drawText(QRectF(x, y - 20, bar_width, 20),
                             Qt.AlignCenter, str(value))

            # Draw label if provided
            if labels and i < len(labels):
                painter.drawText(QRectF(x, size.height() - margin + 5, bar_width, 20),
                                 Qt.AlignCenter, labels[i])

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

    def setup_games_tab(self):
        """Setup the games tab with detailed statistics for each game type"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        # Create tab widget for different games
        game_tabs = QTabWidget()
        game_tabs.setStyleSheet("""
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

        # Quiz Game stats
        quiz_tab = QWidget()
        quiz_layout = QVBoxLayout(quiz_tab)
        self.setup_quiz_stats(quiz_layout)
        game_tabs.addTab(quiz_tab, "❓ Quiz")

        # Matching Game stats
        matching_tab = QWidget()
        matching_layout = QVBoxLayout(matching_tab)
        self.setup_matching_stats(matching_layout)
        game_tabs.addTab(matching_tab, "🃏 Matching")

        # Bubble Shooter stats
        bubble_tab = QWidget()
        bubble_layout = QVBoxLayout(bubble_tab)
        self.setup_bubble_stats(bubble_layout)
        game_tabs.addTab(bubble_tab, "🎯 Bubble")

        layout.addWidget(game_tabs)
        self.tabs.addTab(tab, "Games")

    def setup_quiz_stats(self, layout):
        """Setup quiz game statistics with charts using new structure"""
        stats_frame = QGroupBox("❓ Quiz Game Statistics")
        stats_frame.setStyleSheet(self.get_card_style("#6e8efb"))
        stats_layout = QVBoxLayout(stats_frame)

        quiz_stats = self.user_data['stats']['game_stats']['quiz']
        quiz_history = self.user_data.get('game_history', {}).get('quiz', [])

        # Create stats in a vertical layout
        stats_layout.addWidget(QLabel(f"🔄 Attempts: {quiz_stats['attempts']}"))
        stats_layout.addWidget(QLabel(f"✅ Correct Answers: {quiz_stats['correct']}"))
        stats_layout.addWidget(QLabel(f"🎯 Highest Score: {quiz_stats['highest_score']}"))
        stats_layout.addWidget(QLabel(f"📊 Last Score: {quiz_stats['last_score']}"))
        stats_layout.addWidget(QLabel(f"🎯 Accuracy: {quiz_stats['accuracy']:.1f}%"))

        layout.addWidget(stats_frame)

        # Score trend chart
        if quiz_history:
            dates = [datetime.fromisoformat(session['date']).strftime("%m/%d")
                     for session in quiz_history]
            scores = [session.get('score', 0) for session in quiz_history]
            accuracies = [(session.get('correct', 0) / session.get('total', 1) * 100)
                          if session.get('total', 0) > 0 else 0
                          for session in quiz_history]

            chart = QLabel()
            chart.setFixedHeight(200)
            chart.setStyleSheet("background-color: white; border: 1px solid #ddd;")

            pixmap = self.draw_dual_axis_chart(
                chart.size(),
                scores,
                accuracies,
                dates,
                "Score",
                "Accuracy (%)",
                "#6e8efb",
                "#4CAF50"
            )
            chart.setPixmap(pixmap)
            stats_layout.addWidget(chart)
        else:
            no_data = QLabel("No quiz game data available yet")
            no_data.setStyleSheet("font-size: 14px; color: #666;")
            stats_layout.addWidget(no_data)

        layout.addWidget(stats_frame)

    def setup_matching_stats(self, layout):
        """Setup matching game statistics with charts using new structure"""
        stats_frame = QGroupBox("🃏 Matching Game Statistics")
        stats_frame.setStyleSheet(self.get_card_style("#4CAF50"))
        stats_layout = QVBoxLayout(stats_frame)

        matching_stats = self.user_data['stats']['game_stats']['matching']
        matching_history = self.user_data.get('game_history', {}).get('matching', [])

        # Basic stats
        stats_grid = QGridLayout()
        stats_grid.addWidget(QLabel(f"🔄 Attempts: {matching_stats['attempts']}"))
        stats_grid.addWidget(QLabel(f"✅ Matches: {matching_stats['matches']}"))
        stats_grid.addWidget(QLabel(f"⏱️ Best Time: {matching_stats['best_time'] or 'N/A'}"))
        stats_grid.addWidget(QLabel(f"📊 Last Score: {matching_stats['last_score']}"))
        stats_grid.addWidget(QLabel(f"🎯 Accuracy: {matching_stats['accuracy']:.1f}%"))
        stats_layout.addLayout(stats_grid)

        # Performance chart
        if matching_history:
            dates = [datetime.fromisoformat(session['date']).strftime("%m/%d")
                     for session in matching_history]
            matches = [session.get('matches', 0) for session in matching_history]
            times = [session.get('time', 0) for session in matching_history]

            chart = QLabel()
            chart.setFixedHeight(200)
            chart.setStyleSheet("background-color: white; border: 1px solid #ddd;")

            pixmap = self.draw_dual_axis_chart(
                chart.size(),
                matches,
                times,
                dates,
                "Matches",
                "Time (s)",
                "#4CAF50",
                "#FF9800"
            )
            chart.setPixmap(pixmap)
            stats_layout.addWidget(chart)
        else:
            no_data = QLabel("No matching game data available yet")
            no_data.setStyleSheet("font-size: 14px; color: #666;")
            stats_layout.addWidget(no_data)

        layout.addWidget(stats_frame)

    def setup_bubble_stats(self, layout):
        """Setup bubble shooter statistics with charts using new structure"""
        stats_frame = QGroupBox("🎯 Bubble Shooter Statistics")
        stats_frame.setStyleSheet(self.get_card_style("#FF9800"))
        stats_layout = QVBoxLayout(stats_frame)

        bubble_stats = self.user_data['stats']['game_stats']['bubble']
        bubble_history = self.user_data.get('game_history', {}).get('bubble', [])

        # Basic stats
        stats_grid = QGridLayout()
        stats_grid.addWidget(QLabel(f"🔄 Attempts: {bubble_stats['attempts']}"))
        stats_grid.addWidget(QLabel(f"💥 Bubbles Popped: {bubble_stats['bubbles_popped']}"))
        stats_grid.addWidget(QLabel(f"🏆 Highest Level: {bubble_stats['highest_level']}"))
        stats_grid.addWidget(QLabel(f"📊 Last Score: {bubble_stats['last_score']}"))
        stats_grid.addWidget(QLabel(f"🎯 Accuracy: {bubble_stats['accuracy']:.1f}%"))
        stats_layout.addLayout(stats_grid)

        # Performance chart
        if bubble_history:
            dates = [datetime.fromisoformat(session['date']).strftime("%m/%d")
                     for session in bubble_history]
            scores = [session.get('score', 0) for session in bubble_history]
            levels = [session.get('level', 1) for session in bubble_history]

            chart = QLabel()
            chart.setFixedHeight(200)
            chart.setStyleSheet("background-color: white; border: 1px solid #ddd;")

            pixmap = self.draw_dual_axis_chart(
                chart.size(),
                scores,
                levels,
                dates,
                "Score",
                "Level",
                "#FF9800",
                "#FF5722"
            )
            chart.setPixmap(pixmap)
            stats_layout.addWidget(chart)
        else:
            no_data = QLabel("No bubble shooter data available yet")
            no_data.setStyleSheet("font-size: 14px; color: #666;")
            stats_layout.addWidget(no_data)

        layout.addWidget(stats_frame)

    def create_quiz_chart(self):
        """Create a chart for quiz game performance"""
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)

        # Title
        title = QLabel("Quiz Performance Over Time")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        chart_layout.addWidget(title)

        # Get session history
        session_history = self.user_data.get('game_history', {}).get('quiz', [])

        # Prepare data for chart
        dates = []
        scores = []
        accuracies = []

        for session in session_history:
            dates.append(datetime.fromisoformat(session['date']).strftime("%m/%d"))
            scores.append(session.get('score', 0))
            total = session.get('total', 1)
            correct = session.get('correct', 0)
            accuracies.append((correct / total) * 100 if total > 0 else 0)

        # Create chart with real data
        chart = QLabel()
        chart.setFixedHeight(200)
        chart.setStyleSheet("background-color: white; border: 1px solid #ddd;")

        if scores and accuracies:
            pixmap = self.draw_dual_axis_chart(
                chart.size(),
                scores,
                accuracies,
                dates,
                "Score",
                "Accuracy (%)",
                "#6e8efb",
                "#4CAF50"
            )
        else:
            pixmap = QPixmap(chart.size())
            pixmap.fill(Qt.white)
            painter = QPainter(pixmap)
            painter.drawText(chart.rect(), Qt.AlignCenter, "No quiz data available")
            painter.end()

        chart.setPixmap(pixmap)
        chart_layout.addWidget(chart)

        return chart_widget

    def create_matching_chart(self):
        """Create a chart for matching game performance"""
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)

        # Title
        title = QLabel("Matching Game Performance")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        chart_layout.addWidget(title)

        # Get session history
        session_history = self.user_data.get('game_history', {}).get('matching', [])

        # Prepare data for chart
        dates = []
        scores = []
        times = []

        for session in session_history:
            date = datetime.fromisoformat(session['date']).strftime("%m/%d")
            dates.append(date)
            scores.append(session.get('score', 0))
            times.append(session.get('time', 0))  # Time in seconds

        # Create chart with real data
        chart = QLabel()
        chart.setFixedHeight(200)
        chart.setStyleSheet("background-color: white; border: 1px solid #ddd;")

        if scores:
            pixmap = self.draw_dual_axis_chart(
                chart.size(),
                scores,
                times,
                dates,
                "Score",
                "Time (s)",
                "#6e8efb",
                "#FF9800"
            )
        else:
            pixmap = QPixmap(chart.size())
            pixmap.fill(Qt.white)
            painter = QPainter(pixmap)
            painter.drawText(chart.rect(), Qt.AlignCenter, "No matching game data available")
            painter.end()

        chart.setPixmap(pixmap)
        chart_layout.addWidget(chart)

        return chart_widget

    def create_bubble_chart(self):
        """Create a chart for bubble shooter performance with empty data handling"""
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)

        # Title
        title = QLabel("Bubble Shooter Performance")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        chart_layout.addWidget(title)

        # Get session history
        session_history = self.user_data.get('game_history', {}).get('bubble', [])

        # Prepare data for chart - handle empty case
        dates = []
        scores = []
        matches = []

        for session in session_history:
            dates.append(datetime.fromisoformat(session['date']).strftime("%m/%d"))
            scores.append(session.get('score', 0))
            matches.append(session.get('matches', 0))

        # Create chart with real data or empty state
        chart = QLabel()
        chart.setFixedHeight(200)
        chart.setStyleSheet("background-color: white; border: 1px solid #ddd;")

        if scores and matches:
            pixmap = self.draw_dual_axis_chart(
                chart.size(),
                scores,
                matches,
                dates,
                "Score",
                "Matches",
                "#6e8efb",
                "#FF5722"
            )
        else:
            pixmap = QPixmap(chart.size())
            pixmap.fill(Qt.white)
            painter = QPainter(pixmap)
            painter.drawText(chart.rect(), Qt.AlignCenter, "No bubble shooter data available")
            painter.end()

        chart.setPixmap(pixmap)
        chart_layout.addWidget(chart)

        return chart_widget

    def create_real_bubble_chart(self):
        """Create a chart with real bubble shooter performance data"""
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)

        # Title
        title = QLabel("Bubble Shooter Performance")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        chart_layout.addWidget(title)

        # Get session history
        session_history = self.user_data.get('game_history', {}).get('bubble', [])

        # Prepare data for chart
        dates = []
        scores = []

        for session in session_history:
            dates.append(datetime.fromisoformat(session['date']).strftime("%m/%d"))
            scores.append(session.get('score', 0))  # Default to 0 if score missing

        # If no scores, create some dummy data for the chart
        if not scores:
            dates = ["01/01", "01/02", "01/03"]
            scores = [0, 0, 0]
            title.setText("Bubble Shooter Performance (No Data Yet)")

        # Create chart with real data
        chart = QLabel()
        chart.setFixedHeight(200)
        chart.setStyleSheet("background-color: white; border: 1px solid #ddd;")

        pixmap = self.draw_line_chart(chart.size(), scores, dates)
        chart.setPixmap(pixmap)
        chart_layout.addWidget(chart)

        return chart_widget

    def create_real_chart(self, title, data):
        """Create a proper chart widget with the given data"""
        # Create a widget to hold our chart
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)

        # Title label
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        chart_layout.addWidget(title_label)

        # Create a simple bar chart using QWidget painting
        chart = QLabel()
        chart.setFixedHeight(200)
        chart.setStyleSheet("background-color: white; border: 1px solid #ddd;")

        # Generate a pixmap with the chart
        pixmap = self.draw_bar_chart(chart.size(), data)
        chart.setPixmap(pixmap)

        chart_layout.addWidget(chart)

        return chart_widget


class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Handly - Sign Language Learning")
        self.showMaximized()  # Open in full window

        # Initialize user data
        self.current_user = None
        self.user_data_file = "users.json"

        # Initialize Google Drive manager
        self.drive_manager = GoogleDriveManager()

        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)  # Remove spacing

        # Create stacked widget for different screens
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        # Setup all screens
        self.setup_login_screen()
        self.setup_intro_screen()

        # Show login screen first
        self.stacked_widget.setCurrentIndex(0)

    def setup_login_screen(self):
        """Setup the login screen as part of the main window"""
        login_page = QWidget()
        login_layout = QVBoxLayout(login_page)
        login_layout.setContentsMargins(20, 20, 20, 20)  # Remove margins for full-width styling
        login_layout.setSpacing(20)

        self.setup_title_section(login_layout)

        # Form container (white background)
        form_container = QWidget()
        form_container.setStyleSheet("background-color: white;")
        form_layout = QVBoxLayout(form_container)
        form_layout.setContentsMargins(20, 20, 20, 20)
        form_layout.setSpacing(20)

        # Form elements (same as before)
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter your username")
        form_layout.addWidget(QLabel("Username:"))

        form_layout.addWidget(self.username_input)

        # Buttons
        btn_layout = QHBoxLayout()
        self.sign_in_btn = QPushButton("Sign In")
        self.sign_in_btn.setStyleSheet("""
            QPushButton {
                background-color: #6e8efb;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #5a7df4;
            }
        """)
        self.sign_in_btn.clicked.connect(self.handle_login)
        btn_layout.addWidget(self.sign_in_btn)

        self.new_user_btn = QPushButton("New User")
        self.new_user_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.new_user_btn.clicked.connect(self.create_new_user)
        btn_layout.addWidget(self.new_user_btn)

        form_layout.addLayout(btn_layout)

        # Status label
        self.login_status_label = QLabel()
        self.login_status_label.setStyleSheet("color: red;")
        form_layout.addWidget(self.login_status_label)

        login_layout.addWidget(form_container)
        login_layout.addStretch()

        self.stacked_widget.addWidget(login_page)

    def setup_intro_screen(self):
        """Setup the intro screen as part of the main window"""
        intro_page = QWidget()
        intro_layout = QVBoxLayout(intro_page)
        intro_layout.setContentsMargins(0, 0, 0, 0)
        intro_layout.setSpacing(0)

        self.setup_title_section(intro_layout)

        # Video player section
        self.intro_player = QMediaPlayer()
        self.intro_video_widget = QVideoWidget()
        self.intro_player.setVideoOutput(self.intro_video_widget)
        intro_layout.addWidget(self.intro_video_widget, 1)  # Take remaining space

        # Continue button
        self.continue_btn = QPushButton("Let's Get Started!")
        self.continue_btn.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                padding: 15px 30px;
                background-color: #6e8efb;
                color: white;
                border-radius: 8px;
                min-width: 250px;
                margin: 30px;
            }
            QPushButton:hover {
                background-color: #5a7df4;
            }
        """)
        self.continue_btn.hide()
        self.continue_btn.clicked.connect(self.show_main_app)

        # Button container to center it
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.addWidget(self.continue_btn, 0, Qt.AlignCenter)
        btn_layout.setContentsMargins(0, 0, 0, 30)

        intro_layout.addWidget(btn_container)

        self.stacked_widget.addWidget(intro_page)

    def handle_login(self):
        """Handle user login with proper data validation"""
        username = self.username_input.text().strip()
        if not username:
            self.login_status_label.setText("Please enter a username")
            return

        user_data = self.load_user_data()
        if username in user_data:
            self.current_user = user_data[username]

            # Ensure the user data has all required fields
            if 'stats' not in self.current_user:
                self.current_user['stats'] = {
                    "game_stats": {
                        "quiz": {"attempts": 0, "correct": 0, "highest_score": 0, "last_score": 0, "accuracy": 0.0},
                        "matching": {"attempts": 0, "matches": 0, "best_time": None, "last_score": 0, "accuracy": 0.0},
                        "bubble": {"attempts": 0, "bubbles_popped": 0, "highest_level": 0, "last_score": 0,
                                   "accuracy": 0.0}
                    },
                    "practice_stats": {
                        "sessions": 0, "total_time": 0, "words_attempted": 0, "words_correct": 0,
                        "current_streak": 0, "longest_streak": 0, "last_practiced": None,
                        "daily_goal": 3, "goal_streak": 0
                    },
                    "efficiency": {
                        "words_per_hour": 0.0, "games_per_week": 0.0, "improvement_rate": 0.0
                    }
                }

            # Add other missing fields as needed...
            self.show_intro_screen()
        else:
            self.login_status_label.setText("User not found")

    def create_new_user(self):
        """Create a new user account with the updated data structure"""
        username = self.username_input.text().strip()
        if not username:
            self.login_status_label.setText("Please enter a username")
            return

        user_data = self.load_user_data()
        if username in user_data:
            self.login_status_label.setText("Username already exists")
            return

        # Create user with new data structure
        new_user = {
            "username": username,
            "stats": {
                # Consolidated game statistics
                "game_stats": {
                    "quiz": {
                        "attempts": 0,
                        "correct": 0,
                        "highest_score": 0,
                        "last_score": 0,
                        "total_score": 0,
                        "accuracy": 0.0
                    },
                    "matching": {
                        "attempts": 0,
                        "matches": 0,
                        "best_time": None,
                        "last_score": 0,
                        "total_score": 0,
                        "accuracy": 0.0
                    },
                    "bubble": {
                        "attempts": 0,
                        "bubbles_popped": 0,
                        "highest_level": 0,
                        "last_score": 0,
                        "total_score": 0,
                        "accuracy": 0.0
                    }
                },

                # Enhanced practice tracking
                "practice_stats": {
                    "sessions": 0,
                    "total_time": 0,
                    "words_attempted": 0,
                    "words_correct": 0,
                    "current_streak": 0,
                    "longest_streak": 0,
                    "last_practiced": None,
                    "daily_goal": 3,
                    "goal_streak": 0
                },

                # Calculated metrics
                "efficiency": {
                    "words_per_hour": 0.0,
                    "games_per_week": 0.0,
                    "improvement_rate": 0.0
                }
            },

            # Word tracking
            "known_words": [],
            "struggled_words": [],
            "word_progress": {},

            # Activity history
            "practice_history": [],
            "game_history": {
                "quiz": [],
                "matching": [],
                "bubble": []
            },

            # User preferences
            "preferences": {
                "voice_feedback": True,
                "auto_advance": True,
                "difficulty": "normal",
                "daily_goal": 3,
                "theme": "light"
            },

            # System tracking
            "meta": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }
        }

        user_data[username] = new_user
        self.current_user = new_user
        self.save_user_data()
        self.show_intro_screen()

    def load_user_data(self):
        """Load user data from file"""
        try:
            if os.path.exists(self.user_data_file):
                with open(self.user_data_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading user data: {e}")
            return {}

    def save_user_data(self):
        """Save the current user's data to file with proper error handling"""
        try:
            # Load all users or initialize if file doesn't exist
            user_data = {}
            if os.path.exists(self.user_data_file):
                try:
                    with open(self.user_data_file, 'r') as f:
                        user_data = json.load(f)
                except json.JSONDecodeError:
                    # Handle corrupt JSON file
                    user_data = {}

            # Ensure we have a current user to save
            if not self.current_user:
                print("Warning: No current user to save")
                return

            # Update current user's data
            user_data[self.current_user['username']] = self.current_user

            # Save back to file
            with open(self.user_data_file, 'w') as f:
                json.dump(user_data, f, indent=2)

        except Exception as e:
            print(f"Error saving user data: {str(e)}")
            # Optionally show error to user
            QMessageBox.warning(self, "Save Error", f"Could not save user data: {str(e)}")

    def embed_text(self, text, model_name='default'):
        """Generate embeddings for text using the local SignCLIP model."""
        try:
            # Convert single text to list if needed
            texts = [text] if isinstance(text, str) else text

            # Get embeddings using the local model
            embeddings = embed_text(texts, model_name)

            # Ensure proper shape
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)  # Ensure 2D array

            # Return first embedding if input was single text
            if len(texts) == 1 and len(embeddings) >= 1:
                return embeddings[0]  # Return first embedding as array
            return embeddings

        except Exception as e:
            print(f"Error generating local text embeddings: {str(e)}")
            return np.array([])

    def show_intro_screen(self):
        """Show the intro screen and play intro video"""
        self.stacked_widget.setCurrentIndex(1)

        # Try to load intro video from Google Drive
        intro_folder = self.drive_manager.get_folder_id("intro")
        if intro_folder:
            video_files = self.drive_manager.list_files(intro_folder['id'], mime_type='video/mp4')
            intro_video = next((f for f in video_files if f['name'].lower() == "intro.mp4"), None)

            if intro_video:
                temp_path = os.path.join(tempfile.gettempdir(), "intro_temp.mp4")
                self.drive_manager.download_file(intro_video['id'], temp_path)
                self.intro_player.setMedia(QMediaContent(QUrl.fromLocalFile(temp_path)))
                self.intro_player.play()

                # Show continue button when video ends
                self.intro_player.stateChanged.connect(self.handle_video_state_change)

    def handle_video_state_change(self, state):
        """Handle video state changes to show continue button"""
        if state == QMediaPlayer.StoppedState:
            self.continue_btn.show()

    def show_main_app(self):
        """Show the main application after intro"""
        self.initialize_main_app()
        self.stacked_widget.setCurrentIndex(2)

    def initialize_main_app(self):
        """Initialize the main application after intro"""
        self.words = ['hello', 'coach', 'clumsy', 'clueless', 'closet', 'close', 'clock', 'climb', 'click', 'clever',
                      'classroom', 'church', 'christ', 'choose', 'choke', 'china', 'bible', 'best', 'bell', 'behavior',
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

        # Initialize UI
        self.setWindowTitle("Handly - Sign Language Learning")
        # self.setGeometry(100, 100, 1200, 900)

        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)

        # Now setup the UI components
        self.setup_ui()
        self.load_thresholds()
        self.update_word_display()
        self.setup_navigation()  # Now main_layout exists
        self.update_score_display()

        self.setCentralWidget(self.main_widget)
        self.game_windows = []  # Track open game windows

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

        subtitle_label = QLabel("A friendly hand to help you understand!")
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
        """Show game selection menu that uses the unified start_game() method."""
        menu = QDialog(self)
        menu.setWindowTitle("Select Game")
        layout = QVBoxLayout(menu)

        # Quiz game button
        quiz_btn = QPushButton("❓ Sign Quiz")
        quiz_btn.setStyleSheet(self.game_button_style())
        quiz_btn.clicked.connect(lambda: self.start_game_and_close(menu, 'quiz'))

        # Matching game button
        match_btn = QPushButton("🃏 Sign Matching")
        match_btn.setStyleSheet(self.game_button_style())
        match_btn.clicked.connect(lambda: self.start_game_and_close(menu, 'matching'))

        # Bubble Shooter button
        bubble_btn = QPushButton("🎯 Bubble Shooter")
        bubble_btn.setStyleSheet(self.game_button_style())
        bubble_btn.clicked.connect(lambda: self.start_game_and_close(menu, 'bubble'))

        layout.addWidget(quiz_btn)
        layout.addWidget(match_btn)
        layout.addWidget(bubble_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(menu.close)
        layout.addWidget(close_btn)

        menu.exec_()

    def game_button_style(self):
        """Shared style for game buttons"""
        return """
            QPushButton {
                font-size: 16px;
                padding: 12px;
                margin: 5px;
                min-width: 200px;
            }
        """

    def start_game_and_close(self, menu, game_type):
        """Start game and close menu."""
        self.start_game(game_type)
        menu.accept()

    # In the start_game method of SignLanguageApp class:
    def start_game(self, game_type):
        """Start the specified game type with proper word selection."""
        # Get struggled words, remove duplicates and words already known
        struggled_words = list(set(
            word for word in self.current_user.get('struggled_words', [])
            if word not in self.current_user.get('known_words', [])
        ))

        # If not enough struggled words, supplement with random words
        remaining_words = list(set(
            word for word in self.words
            if word not in struggled_words and word not in self.current_user.get('known_words', [])
        ))

        # Determine minimum words needed based on game type
        if game_type == 'matching':
            min_words_needed = 8  # Matching game requires 8 unique words
        elif game_type == 'bubble':
            min_words_needed = 10  # Bubble shooter needs more words
        else:  # quiz
            min_words_needed = 8  # Other games can work with fewer words

        if len(struggled_words) < min_words_needed:
            needed = min_words_needed - len(struggled_words)
            struggled_words.extend(random.sample(remaining_words, min(needed, len(remaining_words))))

        # Create the appropriate game
        if game_type == 'quiz':
            game = SignQuizGame(struggled_words, self.current_user, self)
        elif game_type == 'matching':
            game = SignMatchingGame(struggled_words, self.current_user, self)
        elif game_type == 'bubble':
            game = BubbleShooterGame(struggled_words, self.current_user, self)
        else:
            raise ValueError(f"Unknown game type: {game_type}")

        # Set drive manager for all games
        game.drive_manager = self.drive_manager

        # Add game to windows list before connecting signals
        self.game_windows.append(game)

        def cleanup_game():
            try:
                if game in self.game_windows:
                    self.game_windows.remove(game)
                # Force UI update
                self.update_score_display()
                if hasattr(self, 'progress_page'):
                    self.progress_page.refresh_data(self.current_user)
                    self.update_score_display()
            except ValueError:
                pass  # Game was already removed

        game.finished.connect(cleanup_game)

        # Center and show the window
        game_geometry = game.frameGeometry()
        center_point = self.frameGeometry().center()
        game_geometry.moveCenter(center_point)
        game.move(game_geometry.topLeft())

        game.show()
        return game

    def show_progress_dashboard(self):
        """Show the progress dashboard page with updated data."""
        # Check if we already have a progress page
        if hasattr(self, 'progress_page'):
            # Refresh the existing page
            self.progress_page.refresh_data(self.current_user)
        else:
            # Create new progress page
            self.progress_page = ProgressDashboard(self.current_user)
            self.stacked_widget.addWidget(self.progress_page)

        # Show the page
        self.stacked_widget.setCurrentWidget(self.progress_page)

    def update_score_display(self):
        """Update the score display with real data from all games"""
        stats = self.current_user['stats']

        # Calculate total points from all games
        total_game_points = (stats['game_stats']['quiz'].get('total_points', 0) +
                             stats['game_stats']['matching'].get('total_points', 0) +
                             stats['game_stats']['bubble'].get('total_points', 0))

        # Create detailed score text
        score_text = (f"🏆 Points - Learning: {stats['practice_stats'].get('words_correct')} | "
                      f"Games: {total_game_points} "
                      f"(Quiz: {stats['game_stats']['quiz'].get('total_points', 0)}, "
                      f"Matching: {stats['game_stats']['matching'].get('total_points', 0)}, "
                      f"Bubble: {stats['game_stats']['bubble'].get('total_points', 0)}")

        # Update or create the score display
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
            # Add to layout if not already present
            if hasattr(self, 'title_layout'):
                self.title_layout.addWidget(self.score_display, 0, Qt.AlignRight)

        # Force immediate UI update
        QApplication.processEvents()

    def setup_progress_page(self):
        """Setup the progress dashboard page."""
        self.progress_page = ProgressDashboard(self.current_user)
        self.stacked_widget.addWidget(self.progress_page)

    # In the setup_demo_page method of SignLanguageApp class:
    def setup_demo_page(self):
        """Setup the demonstration and upload page."""
        demo_page = QWidget()

        # Create the main layout for the page
        main_layout = QVBoxLayout(demo_page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create scroll area with expanded viewport
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Set minimum height to ensure scroll area is tall enough
        scroll.setMinimumHeight(600)

        # Create the scroll content widget
        scroll_content = QWidget()
        scroll_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(0)

        # Title section
        self.setup_title_section(scroll_layout)

        # Current word section
        self.setup_word_section(scroll_layout)

        # Demo video section - make it slightly smaller
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
        self.demo_player.setMinimumHeight(300)  # Reduced from 400
        demo_video_layout.addWidget(self.demo_player)

        scroll_layout.addWidget(demo_video_frame)

        # Upload section - ensure it's always visible
        upload_frame = QFrame()
        upload_frame.setStyleSheet("""
            background-color: white;
            border-radius: 12px;
            padding: 15px;
        """)
        upload_layout = QVBoxLayout(upload_frame)
        upload_layout.setContentsMargins(10, 10, 10, 10)

        upload_title = QLabel("📤 Your Turn to Sign!")
        upload_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        upload_layout.addWidget(upload_title)

        upload_instructions = QLabel("Record yourself signing the word using the options below")
        upload_instructions.setStyleSheet("font-size: 14px; color: #666;")
        upload_layout.addWidget(upload_instructions)

        # Video input options - with fixed height
        self.input_options = QWidget()
        self.input_options.setFixedHeight(80)  # Fixed height for buttons
        input_layout = QHBoxLayout(self.input_options)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(20)

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
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.upload_btn.clicked.connect(self.upload_video)
        input_layout.addWidget(self.upload_btn, 0, Qt.AlignCenter)

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
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.record_btn.clicked.connect(self.toggle_recording)
        input_layout.addWidget(self.record_btn, 0, Qt.AlignCenter)

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

        scroll_layout.addWidget(upload_frame)

        # Add stretch to push content up and ensure scrollability
        scroll_layout.addStretch(1)

        # Set the scroll content and add to main layout
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        self.stacked_widget.addWidget(demo_page)

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
        scroll_layout.setContentsMargins(0, 0, 0, 0)  # Add margins
        scroll_layout.setSpacing(0)

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
        title_frame = QFrame()
        title_frame.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                        stop:0 #6e8efb, stop:1 #a777e3);
            border-radius: 15px;
            padding: 20px;
        """)

        # Use a grid layout for better control
        title_layout = QGridLayout(title_frame)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(0)  # Remove spacing between items

        # Text container (centered) - make it transparent
        text_container = QWidget()
        text_container.setStyleSheet("background: transparent;")  # Make container transparent
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(10)

        title_label = QLabel("👋 Handly")
        title_label.setStyleSheet("""
            color: white;
            font-size: 36px;
            font-weight: bold;
            background: transparent;
        """)
        title_label.setAlignment(Qt.AlignCenter)

        subtitle_label = QLabel("A friendly hand to help you understand!")
        subtitle_label.setStyleSheet("""
            color: white;
            font-size: 16px;
            background: transparent;
        """)
        subtitle_label.setAlignment(Qt.AlignCenter)

        text_layout.addWidget(title_label)
        text_layout.addWidget(subtitle_label)

        # Add text container to the center of the grid
        title_layout.addWidget(text_container, 0, 1, Qt.AlignCenter)

        # Image (right side) - only add if image exists
        image_label = QLabel()
        pixmap = QPixmap("handly_logo.png")  # Must be a PNG with transparency
        if not pixmap.isNull():
            pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(pixmap)
            image_label.setStyleSheet("background: transparent; border: none;")
            image_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            # Add image to the right column
            title_layout.addWidget(image_label, 0, 2, Qt.AlignRight)

        # Add empty widget to left column to balance the layout
        left_spacer = QWidget()
        left_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        left_spacer.setStyleSheet("background: transparent;")  # Make spacer transparent
        title_layout.addWidget(left_spacer, 0, 0)

        # Set column stretch factors
        title_layout.setColumnStretch(0, 1)  # Left spacer
        title_layout.setColumnStretch(1, 0)  # Text container (no stretch)
        title_layout.setColumnStretch(2, 1)  # Right spacer/image

        parent_layout.addWidget(title_frame)

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
        new_word_btn.clicked.connect(self.new_word)
        word_layout.addWidget(new_word_btn)
        parent_layout.addWidget(word_frame)

    def start_practice_mode(self, correct_video_path):
        """Start the frame-by-frame practice mode"""
        # Initialize practice_stats if it doesn't exist
        if 'practice_stats' not in self.current_user['stats']:
            self.current_user['stats']['practice_stats'] = {
                "sessions": 0,
                "total_time": 0,
                "words_attempted": 0,
                "words_correct": 0,
                "current_streak": 0,
                "longest_streak": 0,
                "last_practiced": None,
                "daily_goal": 3,
                "goal_streak": 0
            }

        # Increment the session count
        self.current_user['stats']['practice_stats']['sessions'] += 1

        # Create and show the practice widget
        self.practice_widget = PracticeWidget(correct_video_path)

        # Refresh progress data
        if hasattr(self, 'progress_page'):
            self.progress_page.refresh_data(self.current_user)

        self.practice_widget.show()

    def new_word(self):
        """Select a new random word and update the UI."""
        self.cleanup_temp_files()
        self.current_word = random.choice(self.words)

        # Update both word displays if they exist
        if self.demo_word_label is not None:
            self.demo_word_label.setText(f"📝 Current Sign: {self.current_word}")
        if self.results_word_label is not None:
            self.results_word_label.setText(f"📝 Current Sign: {self.current_word}")

        # Load the new demo video
        self.load_demo_video()

        # Clear any previous feedback
        self.feedback_text.clear()
        self.progress_bar.setValue(0)

        # Stop any playing videos
        self.user_player.media_player.stop()
        self.correct_player.media_player.stop()
        self.overlay_player.media_player.stop()

        # Switch back to the demo page (index 0)
        self.stacked_widget.setCurrentIndex(0)

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
        cache_path = "/Users/yaelbatat/Desktop/pythonProject/final_project/my_thresholds.json"

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
        """Update the UI with the current word and load its demo video."""
        if self.demo_word_label is not None:
            self.demo_word_label.setText(f"📝 Current Sign: {self.current_word}")
        if self.results_word_label is not None:
            self.results_word_label.setText(f"📝 Current Sign: {self.current_word}")
        self.load_demo_video()

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
        start_time = time.time()
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
        end_time = time.time()
        print(f"align_and_scale_video took {end_time - start_time:.2f} seconds")

    def video_to_pose(self, video_path, output_pose_path):

        """Convert a video to pose format using command line."""
        start_time = time.time()
        cmd = f"video_to_pose -i {video_path} --format mediapipe -o {output_pose_path}"
        ret = os.system(cmd)
        if ret != 0:
            raise ValueError(f"Pose conversion failed (return code {ret})")
        end_time = time.time()
        print(f"video_to_pose took {end_time - start_time:.2f} seconds")

    def read_pose(self, pose_path):
        """Read a pose file."""
        with open(pose_path, "rb") as f:
            return Pose.read(f.read())

    def embed_pose(self, pose, model_name='default'):
        """Generate embeddings for a pose using the local SignCLIP model."""
        start_time = time.time()
        try:
            # Convert single pose to list if needed
            poses = [pose] if not isinstance(pose, list) else pose

            # Prepare pose objects
            pose_objects = []
            for p in poses:
                if isinstance(p, Pose):
                    pose_objects.append(p)
                else:
                    try:
                        with open(p, "rb") as f:
                            pose_objects.append(Pose.read(f.read()))
                    except Exception as e:
                        print(f"Error processing pose file {p}: {str(e)}")
                        continue

            if not pose_objects:
                return np.zeros(768)  # Return zero vector if no valid poses

            # Get embeddings using the local model
            embeddings = embed_pose(pose_objects, model_name)

            # Ensure proper shape (always return 768-dim vector)
            if embeddings.ndim == 1:
                if embeddings.shape[0] == 1:  # Handle scalar case
                    return np.zeros(768)
                return embeddings.reshape(-1)  # Flatten to 1D
            elif embeddings.ndim == 2:
                if embeddings.shape[1] == 1:  # Handle (N,1) case
                    return np.zeros((embeddings.shape[0], 768))
                return embeddings
            else:
                return np.zeros(768)  # Fallback to zero vector

        except Exception as e:
            print(f"Error generating local pose embeddings: {str(e)}")
            return np.zeros(768)  # Return zero vector on error
        finally:
            end_time = time.time()
            print(f"embed_pose took {end_time - start_time:.2f} seconds")


    def analyze_video(self, file_path):
        start_time = time.time()
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
            # Load threshold for current word
            threshold = self.thresholds.get(self.current_word, 0.5)  # Default threshold if not found
            self.analysis_progress.setValue(10)
            self.analysis_status.setText("⏳ Loaded threshold")
            QApplication.processEvents()

            # Setup directories
            dirs = self.setup_word_directories()
            user_input_path = os.path.join(dirs["user_input"], "user_input.mp4")

            # Align user video (no fallback needed here)
            self.align_and_scale_video(file_path, aligned_user_path)
            shutil.copy(aligned_user_path, user_input_path)
            self.analysis_progress.setValue(20)
            self.analysis_status.setText("⏳ Aligned user video")
            QApplication.processEvents()

            # Get correct embeddings and videos
            correct_embeddings, correct_videos = self.get_correct_embeddings()
            if len(correct_embeddings) == 0:
                raise ValueError("No correct examples found for comparison")

            # Process user video
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

            # Normalize embeddings
            all_data = np.vstack([correct_embeddings, [user_embedding]])
            scaler = StandardScaler()
            all_data_normalized = scaler.fit_transform(all_data)

            correct_embeddings_normalized = all_data_normalized[:-1]
            user_embedding_normalized = all_data_normalized[-1]

            # Find nearest neighbors
            knn_model = NearestNeighbors(n_neighbors=min(6, len(correct_embeddings_normalized)))
            knn_model.fit(correct_embeddings_normalized)

            distances, indices = knn_model.kneighbors(user_embedding_normalized.reshape(1, -1))
            mean_distance = float(distances.mean())

            # Try to align videos in order of similarity until one succeeds
            aligned_correct_path = None
            comparison_path = None
            used_video_index = 0

            for i, video_index in enumerate(indices[0]):
                try:
                    current_video = correct_videos[video_index]
                    aligned_correct_path = os.path.join(temp_dir, f"correct_aligned_{i}.mp4")

                    self.analysis_progress.setValue(50 + i * 10)
                    self.analysis_status.setText(f"⏳ Aligning correct example {i + 1}/{len(indices[0])}")
                    QApplication.processEvents()

                    # Try to align this video
                    self.align_and_scale_video(current_video, aligned_correct_path)

                    # If we get here, alignment succeeded
                    used_video_index = i
                    comparison_path = os.path.join(temp_dir, f"comparison_{i}.mp4")

                    # Create comparison video and analyze frame-by-frame match
                    frame_matches = self.compare_and_overlay_videos(aligned_user_path, aligned_correct_path,
                                                                    comparison_path)

                    # Calculate percentage of frames with good match
                    total_frames = len(frame_matches)
                    good_frames = sum(1 for match in frame_matches if match == 'good')
                    good_percentage = (good_frames / total_frames) * 100 if total_frames > 0 else 0

                    # If 90% of frames match well, consider it correct regardless of embedding distance
                    if good_percentage >= 90:
                        mean_distance = threshold * 0.8  # Force correct classification
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

            # Prepare results
            results = {
                'prediction': 'Correct' if mean_distance < threshold else 'Incorrect',
                'confidence_score': max(0, min(1, 1 - (mean_distance / threshold))),
                'distance': mean_distance,
                'threshold': threshold,
                'closest_correct_example': aligned_correct_path,
                'user_video_path': aligned_user_path,
                'comparison_path': comparison_path,
                'temp_dir': temp_dir,
                'used_video_index': used_video_index,  # Track which video was actually used
                'frame_match_percentage': good_percentage  # Add frame match percentage to results
            }

            # After successful analysis, update user stats with new structure
            practice_stats = self.current_user['stats']['practice_stats']
            word_progress = self.current_user['word_progress']

            # Update practice stats

            practice_stats['words_attempted'] += 1

            if results['prediction'] == 'Correct':
                practice_stats['words_correct'] += 1

                # Update word progress
                if self.current_word not in word_progress:
                    word_progress[self.current_word] = {
                        'attempts': 0,
                        'correct': 0,
                        'last_attempted': datetime.now().isoformat(),
                        'confidence': 0.0
                    }

                word_progress[self.current_word]['attempts'] += 1
                word_progress[self.current_word]['correct'] += 1
                word_progress[self.current_word]['last_attempted'] = datetime.now().isoformat()
                word_progress[self.current_word]['confidence'] = results['confidence_score']

                # Mark as known if confidence is high
                if self.current_word not in self.current_user['known_words']:
                    self.current_user['known_words'].append(self.current_word)
                    word_progress[self.current_word]['first_known'] = datetime.now().isoformat()

                    # Update known words history
                    today = datetime.now().date().isoformat()
                    found = False
                    for entry in self.current_user.get('known_words_history', []):
                        if entry['date'] == today:
                            if self.current_word not in entry['words']:
                                entry['words'].append(self.current_word)
                            found = True
                            break

                    if not found:
                        if 'known_words_history' not in self.current_user:
                            self.current_user['known_words_history'] = []
                        self.current_user['known_words_history'].append({
                            'date': today,
                            'words': [self.current_word]
                        })
            else:
                # Update word progress for incorrect attempts
                if self.current_word not in word_progress:
                    word_progress[self.current_word] = {
                        'attempts': 0,
                        'correct': 0,
                        'last_attempted': datetime.now().isoformat(),
                        'confidence': 0.0
                    }

                word_progress[self.current_word]['attempts'] += 1
                word_progress[self.current_word]['last_attempted'] = datetime.now().isoformat()
                word_progress[self.current_word]['confidence'] = results['confidence_score']

                # Add to struggled words if not already known
                if (self.current_word not in self.current_user['known_words'] and
                        self.current_word not in self.current_user['struggled_words']):
                    self.current_user['struggled_words'].append(self.current_word)

            # Update streaks
            today = datetime.now().date()
            last_practiced = practice_stats['last_practiced']

            if last_practiced:
                last_date = datetime.fromisoformat(last_practiced).date()
                if (today - last_date).days == 1:  # Consecutive day
                    practice_stats['current_streak'] += 1
                    if practice_stats['current_streak'] > practice_stats['longest_streak']:
                        practice_stats['longest_streak'] = practice_stats['current_streak']
                elif (today - last_date).days > 1:  # Broken streak
                    practice_stats['current_streak'] = 1
            else:  # First practice
                practice_stats['current_streak'] = 1
                practice_stats['longest_streak'] = max(practice_stats['longest_streak'], 1)

            practice_stats['last_practiced'] = today.isoformat()

            # Update daily goal tracking
            if practice_stats['words_attempted'] >= practice_stats['daily_goal']:
                practice_stats['goal_streak'] += 1

            # Add to practice history
            practice_session = {
                'date': today.isoformat(),
                'duration': 0,  # Would need to track actual duration
                'words_attempted': 1,
                'words_correct': 1 if results['prediction'] == 'Correct' else 0,
                'new_words': [self.current_word] if results['prediction'] == 'Correct' else [],
                'focus_area': 'general'
            }
            self.current_user['practice_history'].append(practice_session)

            # Update meta info
            self.current_user['meta']['last_updated'] = datetime.now().isoformat()

            # Save user data
            self.save_user_data()
            self.update_score_display()

            self.analysis_progress.setValue(100)
            self.analysis_status.setText("✅ Analysis complete!")
            QApplication.processEvents()
            end_time = time.time()
            print(f"sanalyze_video took {end_time - start_time:.2f} seconds")

            QTimer.singleShot(1000, lambda: self.show_results(results))

        except Exception as e:
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)

            self.analysis_status.setText(f"❌ Error: {str(e)}")
            self.input_options.show()
            self.analysis_progress.hide()

            QMessageBox.critical(self, "Analysis Error", f"An error occurred: {str(e)}")

    def get_correct_embeddings(self):
        """
        Retrieves and processes correct example videos from Google Drive, generating consistent 768-dim embeddings.
        Returns:
            tuple: (embeddings_array, video_paths) where:
                - embeddings_array: numpy array of shape (n_examples, 768)
                - video_paths: list of local paths to the downloaded videos
        """
        start_time = time.time()
        embeddings = []
        video_paths = []
        temp_dir = tempfile.mkdtemp(prefix="handly_correct_")
        target_dim = 768  # SignCLIP embedding dimension

        try:
            # 1. Verify Google Drive connection
            if not hasattr(self, 'drive_manager') or not self.drive_manager:
                raise ValueError("Google Drive manager not initialized")

            # 2. Locate the correct folder structure
            folders = []
            for folder_name in ["videos", self.current_word, "Aligned", "Correct"]:
                parent_id = folders[-1]["id"] if folders else None
                folder = self.drive_manager.get_folder_id(folder_name, parent_id)
                if not folder:
                    raise FileNotFoundError(f"Missing required folder: {folder_name}")
                folders.append(folder)

            # 3. Download up to 3 correct examples
            correct_videos = self.drive_manager.list_files(folders[-1]["id"], mime_type='video/mp4')[:3]
            if not correct_videos:
                raise FileNotFoundError(f"No correct videos found for '{self.current_word}'")

            for i, video_file in enumerate(correct_videos):
                video_path = os.path.join(temp_dir, f"correct_{i}.mp4")
                pose_path = os.path.join(temp_dir, f"correct_{i}.pose")

                try:
                    # 4. Download video
                    self.drive_manager.download_file(video_file['id'], video_path)
                    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                        raise ValueError("Downloaded video file is empty")

                    # 5. Convert to pose format
                    try:
                        self.video_to_pose(video_path, pose_path)
                        if not os.path.exists(pose_path) or os.path.getsize(pose_path) == 0:
                            raise ValueError("Generated pose file is empty")
                    except Exception as pose_error:
                        print(f"Pose conversion failed for {video_path}: {str(pose_error)}")
                        # Fallback: Use video directly if pose conversion fails
                        video_paths.append(video_path)
                        embeddings.append(np.zeros(target_dim))
                        continue

                    # 6. Generate embedding
                    with open(pose_path, "rb") as f:
                        pose = Pose.read(f.read())

                    embedding = self.embed_pose(pose)

                    # 7. Validate embedding dimensions
                    if not isinstance(embedding, np.ndarray):
                        raise ValueError(f"Embedding is not numpy array: {type(embedding)}")

                    if embedding.size == 0:
                        raise ValueError("Empty embedding returned")

                    # Ensure proper shape (768,)
                    if embedding.shape == (1,):
                        print(f"Warning: Scalar embedding for {video_path}, using zeros")
                        embedding = np.zeros(target_dim)
                    elif embedding.shape == (1, target_dim):
                        embedding = embedding[0]  # Remove batch dimension
                    elif len(embedding.shape) == 2:
                        embedding = embedding.flatten()

                    if len(embedding) < target_dim:
                        # Pad with zeros if too short
                        new_embed = np.zeros(target_dim)
                        new_embed[:len(embedding)] = embedding
                        embedding = new_embed
                    elif len(embedding) > target_dim:
                        # Truncate if too long
                        embedding = embedding[:target_dim]

                    embeddings.append(embedding)
                    video_paths.append(video_path)
                    print(f"Successfully processed example {i + 1}/{len(correct_videos)}")

                except Exception as e:
                    print(f"Error processing video {i + 1}: {str(e)}")
                    # Add zero vector to maintain alignment with video_paths
                    embeddings.append(np.zeros(target_dim))
                    video_paths.append(video_path)
                    continue
                finally:
                    # Clean up pose file
                    if os.path.exists(pose_path):
                        try:
                            os.remove(pose_path)
                        except:
                            pass

            # 8. Convert to numpy array with proper shape
            if not embeddings:
                return np.zeros((0, target_dim)), []

            embeddings_array = np.stack(embeddings)
            if embeddings_array.shape[1] != target_dim:
                print(f"Adjusting final embeddings shape from {embeddings_array.shape} to (n, {target_dim})")
                if embeddings_array.shape[1] > target_dim:
                    embeddings_array = embeddings_array[:, :target_dim]
                else:
                    new_array = np.zeros((len(embeddings), target_dim))
                    new_array[:, :embeddings_array.shape[1]] = embeddings_array
                    embeddings_array = new_array

            return embeddings_array, video_paths

        except Exception as e:
            print(f"Fatal error in get_correct_embeddings: {str(e)}")
            return np.zeros((0, target_dim)), []

        finally:
            # Clean up remaining temporary files (except videos)
            for file in os.listdir(temp_dir):
                if file.endswith('.pose'):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except:
                        pass
        end_time = time.time()
        print(f"get_correct_embeddings took {end_time - start_time:.2f} seconds")



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
        threshold = 0.04

        # Colors for visualization
        CORRECT_COLOR = (255, 0, 0)  # Blue for correct pose
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
            while True:
                ret_user, user_frame = user_cap.read()
                ret_correct, correct_frame = correct_cap.read()

                if not ret_user or not ret_correct:
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
        """Clean all temporary files including aligned videos"""
        if hasattr(self, 'temp_video_files'):
            for temp_file in self.temp_video_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass
        # Also clean any alignment temp dirs
        temp_dir = os.path.join(tempfile.gettempdir(), "handly_align")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

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


class BubbleShooterGame(QMainWindow):
    finished = pyqtSignal()

    def __init__(self, words, user_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Handly - Bubble Shooter")
        self.resize(1200, 800)  # Default size, but can be resized
        self.setMinimumSize(800, 600)  # Minimum reasonable size
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a2a6c, stop:1 #b21f1f);
            }
        """)

        self.words = words
        self.user_data = user_data
        self.score = 0
        self.lives = 3
        self.level = 1
        self.current_target_word = None
        self.bubbles = []
        self.launcher_angle = math.pi / 2  # Start pointing straight up (90 degrees)
        self.grid_speed = 2
        self.grid_drop_counter = 0
        self.drive_manager = parent.drive_manager if parent else None
        self.temp_video_files = []
        self.matches_found = 0
        self.bubbles_popped = 0
        self.projectile_active = False  # Add this flag

        # Define a palette of distinct colors as RGB tuples
        self.color_palette_rgb = [
            (255, 138, 128),  # Soft Coral
            (255, 183, 77),  # Soft Orange
            (129, 199, 132),  # Light Green
            (128, 216, 255),  # Sky Blue
            (179, 157, 219),  # Lavender
            (244, 143, 177),  # Rosy Pink
            (174, 213, 129),  # Leaf Green
            (255, 204, 128),  # Peachy Orange
            (100, 181, 246),  # Soft Blue
            (255, 171, 145),  # Peach
            (186, 104, 200),  # Orchid Purple
            (255, 112, 67),  # Tangerine
            (77, 182, 172),  # Aqua Green
            (255, 202, 212),  # Pink Blush
            (103, 230, 220),  # Mint Teal
            (240, 147, 203)  # Light Fuchsia
        ]

        self.color_palette = [QColor(*rgb) for rgb in self.color_palette_rgb]
        self.word_colors = {}  # {word: (r,g,b)}
        self.active_words = set()  # Words currently on screen

        self.word_font = QFont("Arial", 12, QFont.Bold)
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animations)
        self.animations = []
        # Angle control (0=right, 90=up, 180=left)
        self.launcher_angle = math.pi / 2  # Start at 90 degrees (pointing up)
        self.angle_change_speed = math.pi / 60  # Rotation speed (3 degrees per frame)
        # Shooter position (moved left by 100 pixels from center)
        # Initialize shooter position
        self.shooter_x = self.width() // 2
        self.shooter_y = self.height() - 50

        # Rotation controls (now reversed)
        self.rotate_left = False  # Will move angle from 90 to 180
        self.rotate_right = False  # Will move angle from 90 to 0

        # Initialize stacked widget for screens
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        # Add these new attributes for keyboard controls
        self.shooter_x = self.width() // 2  # Initial shooter position (center)
        self.shooter_speed = 10  # Movement speed
        self.shooter_width = 60  # Width of the shooter base
        self.move_left = False
        self.move_right = False

        # Setup keyboard timer
        self.keyboard_timer = QTimer()
        self.keyboard_timer.timeout.connect(self.update_shooter_position)
        self.keyboard_timer.start(16)  # ~60 FPS
        # Add game timer for tracking duration
        self.game_start_time = None
        self.game_duration = 0

        # Track words encountered during this game
        self.words_encountered = set()
        # Setup all screens
        self.setup_instructions_page()
        self.setup_game_page()
        self.setup_game_over_page()
        # Add shots tracking
        self.shots_taken = 0

        # Setup victory page
        self.setup_victory_page()

        # Start with instructions
        self.show_instructions()

    def keyPressEvent(self, event):
        """Handle key presses for rotation and shooting"""
        if event.key() == Qt.Key_Left:
            self.rotate_left = True
        elif event.key() == Qt.Key_Right:
            self.rotate_right = True
        elif event.key() == Qt.Key_Space:
            self.shoot_bubble()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Handle key releases"""
        if event.key() == Qt.Key_Left:
            self.rotate_left = False
        elif event.key() == Qt.Key_Right:
            self.rotate_right = False
        else:
            super().keyReleaseEvent(event)

    def update_shooter_position(self):
        """Update launcher angle with reversed controls"""
        if self.rotate_left:
            # Move from 90 to 180 degrees (left arrow)
            self.launcher_angle = min(math.pi, self.launcher_angle + self.angle_change_speed)
        if self.rotate_right:
            # Move from 90 to 0 degrees (right arrow)
            self.launcher_angle = max(0, self.launcher_angle - self.angle_change_speed)

        self.update_canvas()

    def shoot_bubble(self):
        """Shoot from the fixed left position"""
        if self.projectile_active:  # Don't shoot if a projectile is already active
            return

        if not hasattr(self, 'projectiles'):
            self.projectiles = []

        self.projectile_active = True  # Set flag when shooting
        self.shots_taken += 1

        # Use the defined shooter position
        speed = 15
        dx = math.cos(self.launcher_angle) * speed
        dy = -math.sin(self.launcher_angle) * speed

        self.projectiles.append({
            'word': self.current_target_word,
            'color': QColor(255, 255, 255),
            'x': self.shooter_x,
            'y': self.shooter_y,
            'radius': 20,
            'dx': dx,
            'dy': dy,
            'trail': []
        })
        self.play_sound("shoot")

    def update_canvas(self):
        if hasattr(self, 'bubbles') and self.bubbles:
            # Calculate grid width from bubble positions
            rightmost = max(b['x'] for b in self.bubbles)
            leftmost = min(b['x'] for b in self.bubbles)
            grid_width = rightmost - leftmost + (self.bubbles[0]['radius'] * 2)
            # Center shooter within this width
            self.shooter_x = leftmost + (grid_width // 2)

        pixmap = QPixmap(self.width(), self.height())
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        # Update shooter position based on current canvas size

        self.shooter_y = self.canvas.height() - 50

        # Draw bubbles (keep existing code)
        self.draw_bubbles(painter)

        # Draw projectiles (keep existing code)
        for proj in getattr(self, 'projectiles', []):
            for i, (tx, ty) in enumerate(proj['trail']):
                alpha = 50 + i * 40
                radius = proj['radius'] * (0.5 + i * 0.1)
                painter.setBrush(QColor(255, 255, 255, alpha))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(tx, ty), radius, radius)

            painter.setBrush(proj['color'])
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(proj['x'], proj['y']), proj['radius'], proj['radius'])

        # ---- Enhanced Shooter Machine - Center Positioned ----
        shooter_x = self.width() // 2  # Center position
        shooter_y = self.height() - 150  # Fixed position from bottom

        # 1. Draw platform base
        platform_gradient = QLinearGradient(shooter_x - 80, shooter_y + 20,
                                            shooter_x + 80, shooter_y + 20)
        platform_gradient.setColorAt(0, QColor(100, 100, 100))
        platform_gradient.setColorAt(1, QColor(60, 60, 60))
        painter.setBrush(platform_gradient)
        painter.setPen(QPen(QColor(40, 40, 40), 3))
        platform_rect = QRectF(shooter_x - 80, shooter_y - 20, 160, 50)
        painter.drawRoundedRect(platform_rect, 15, 15)

        # 2. Draw rotating base
        base_gradient = QRadialGradient(shooter_x, shooter_y, 35)
        base_gradient.setColorAt(0, QColor(200, 200, 200))
        base_gradient.setColorAt(1, QColor(100, 100, 100))
        painter.setBrush(base_gradient)
        painter.setPen(QPen(QColor(60, 60, 60), 3))
        painter.drawEllipse(QPointF(shooter_x, shooter_y), 35, 35)

        # 3. Draw barrel
        barrel_length = 80
        barrel_end = QPointF(
            shooter_x + math.cos(self.launcher_angle) * barrel_length,
            shooter_y - math.sin(self.launcher_angle) * barrel_length
        )
        barrel_pen = QPen(QColor(150, 150, 150), 15)
        barrel_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(barrel_pen)
        painter.drawLine(QPointF(shooter_x, shooter_y), barrel_end)

        # 4. Draw aiming guide
        aim_length = 200
        aim_end = QPointF(
            shooter_x + math.cos(self.launcher_angle) * aim_length,
            shooter_y - math.sin(self.launcher_angle) * aim_length
        )
        painter.setPen(QPen(QColor(255, 255, 0, 150), 2, Qt.DashLine))
        painter.drawLine(QPointF(shooter_x, shooter_y), aim_end)

        # Update shooter position attributes
        self.shooter_x = shooter_x
        self.shooter_y = shooter_y

        # ---- Game Info Display ----
        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 16, QFont.Bold))

        # Target word
        target_text = f"Target: {self.current_target_word}"
        text_width = painter.fontMetrics().width(target_text)
        painter.drawText(self.width() - text_width - 20, 30, target_text)

        # Controls reminder
        controls_text = "CONTROLS: ← → to aim • SPACE to shoot"
        painter.setFont(QFont("Arial", 14))
        painter.drawText(20, 30, controls_text)

        # Score and level
        painter.setFont(QFont("Arial", 16, QFont.Bold))
        painter.drawText(20, 60, f"Score: {self.score}")
        painter.drawText(20, 90, f"Level: {self.level}")

        painter.end()
        self.canvas.setPixmap(pixmap)

    def setup_instructions_page(self):
        """Setup the game instructions page."""
        self.instructions_page = QWidget()
        layout = QVBoxLayout(self.instructions_page)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(30)

        # Title
        title = QLabel("🎯 Sign Language Bubble Shooter")
        title.setStyleSheet("""
            font-size: 36px;
            font-weight: bold;
            color: white;
            margin-bottom: 20px;
        """)
        layout.addWidget(title, 0, Qt.AlignCenter)

        # Game instructions
        instructions = QLabel(
            "How to Play:\n\n"
            "1. Match the bubble with the sign video at the top\n"
            "2. Shoot bubbles by clicking or moving mouse to aim\n"
            "3. Match 3+ bubbles to score points and clear them\n"
            "4. Don't let the bubbles reach the bottom!\n\n"
            "Tips:\n"
            "- The faster you match, the more points you get\n"
            "- Longer chains give bonus points\n"
            "- Watch the demo video carefully\n"
            "- You have 3 lives - use them wisely!"
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("""
            font-size: 20px;
            color: white;
            margin-bottom: 30px;
        """)
        layout.addWidget(instructions)

        # Button container
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(20)

        # Start button
        start_btn = QPushButton("🚀 Start Game")
        start_btn.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                padding: 15px 30px;
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        start_btn.clicked.connect(self.start_game)
        button_layout.addWidget(start_btn)

        # Home button
        home_btn = QPushButton("🏠 Main Menu")
        home_btn.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                padding: 15px 30px;
                background-color: #6e8efb;
                color: white;
                border-radius: 10px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #5a7df4;
            }
        """)
        home_btn.clicked.connect(self.close)
        button_layout.addWidget(home_btn)

        layout.addWidget(button_container)
        layout.addStretch()

        self.stacked_widget.addWidget(self.instructions_page)

    def setup_game_page(self):
        """Setup the main game play page with properly sized canvas."""
        self.game_page = QWidget()
        layout = QVBoxLayout(self.game_page)
        layout.setContentsMargins(0, 0, 0, 0)  # No margins
        layout.setSpacing(0)

        # Top bar with game info
        top_bar = QWidget()
        top_bar.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.3);
            border-radius: 0;
            padding: 10px;
        """)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(20, 5, 20, 5)

        # Score display
        self.score_label = QLabel(f"Score: {self.score}")
        self.score_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 5px 10px;
                background-color: rgba(0, 0, 0, 0.5);
                border-radius: 10px;
            }
        """)
        top_layout.addWidget(self.score_label)

        # Level display
        self.level_label = QLabel(f"Level: {self.level}")
        self.level_label.setStyleSheet(self.score_label.styleSheet())
        top_layout.addWidget(self.level_label)

        # Lives display
        self.lives_container = QWidget()
        lives_layout = QHBoxLayout(self.lives_container)
        lives_layout.setSpacing(5)

        lives_label = QLabel("Lives:")
        lives_label.setStyleSheet("color: white;")
        lives_layout.addWidget(lives_label)

        self.hearts = []
        for i in range(3):
            heart = QLabel("❤️")
            heart.setObjectName(f"heart_{i}")
            heart.setStyleSheet("font-size: 24px; color: red;")
            self.hearts.append(heart)
            lives_layout.addWidget(heart)
        top_layout.addWidget(self.lives_container)

        layout.addWidget(top_bar)

        # Video container
        self.video_container = QWidget()
        self.video_container.setFixedHeight(180)
        self.video_container.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.5);
            padding: 10px;
        """)
        self.video_layout = QHBoxLayout(self.video_container)
        self.video_layout.setContentsMargins(20, 0, 20, 0)
        self.video_layout.setAlignment(Qt.AlignCenter)

        # Video player
        self.video_output = QVideoWidget()
        self.video_output.setFixedSize(200, 150)
        self.video_output.setStyleSheet("background-color: black; border-radius: 10px;")

        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_output)

        # Replay button
        self.replay_btn = QPushButton()
        self.replay_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.replay_btn.setFixedSize(40, 40)
        self.replay_btn.setStyleSheet("""
            QPushButton {
                background-color: #6e8efb;
                border-radius: 20px;
                border: none;
            }
            QPushButton:hover {
                background-color: #5a7df4;
            }
        """)
        self.replay_btn.clicked.connect(self.replay_demo)
        self.replay_btn.setToolTip("Replay demo")

        self.video_layout.addWidget(self.video_output)
        self.video_layout.addWidget(self.replay_btn)
        layout.addWidget(self.video_container)

        # Game canvas - will match bubble grid width
        self.canvas = QLabel()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setStyleSheet("background-color: rgba(0, 0, 0, 0.2);")
        layout.addWidget(self.canvas)

        # Mouse tracking
        self.canvas.setMouseTracking(True)
        self.setMouseTracking(True)

        self.stacked_widget.addWidget(self.game_page)

    def setup_game_over_page(self):
        """Setup the game over page."""
        self.game_over_page = QWidget()
        layout = QVBoxLayout(self.game_over_page)
        layout.setAlignment(Qt.AlignCenter)

        # Game over text
        game_over_label = QLabel("Game Over!")
        game_over_label.setStyleSheet("""
            QLabel {
                font-size: 48px;
                font-weight: bold;
                color: white;
                margin-bottom: 30px;
            }
        """)
        layout.addWidget(game_over_label)

        # Score display
        self.final_score_label = QLabel()
        self.final_score_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: white;
                margin-bottom: 30px;
            }
        """)
        layout.addWidget(self.final_score_label)

        # Button container
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)

        # Play Again button
        play_again_btn = QPushButton("Play Again")
        play_again_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 18px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        play_again_btn.clicked.connect(self.restart_game)
        button_layout.addWidget(play_again_btn)

        # Home button
        home_btn = QPushButton("Return Home")
        home_btn.setStyleSheet("""
            QPushButton {
                background-color: #6e8efb;
                color: white;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 18px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #5a7df4;
            }
        """)
        home_btn.clicked.connect(self.close)
        button_layout.addWidget(home_btn)

        layout.addWidget(button_container)
        layout.addStretch()

        self.stacked_widget.addWidget(self.game_over_page)

    def show_instructions(self):
        """Show the instructions page."""
        self.stacked_widget.setCurrentWidget(self.instructions_page)

    def start_game(self):
        """Start the game from instructions."""
        self.game_start_time = time.time()
        self.stacked_widget.setCurrentWidget(self.game_page)
        self.load_target_word()
        self.generate_bubble_grid()

        # Ensure we have focus for keyboard events
        self.setFocus()

        # Restart timers if they aren't running
        if not hasattr(self, 'timer') or not self.timer.isActive():
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_game)
            self.timer.start(16)  # ~60 FPS

        if not hasattr(self, 'animation_timer') or not self.animation_timer.isActive():
            self.animation_timer.start(16)

        self.update_lives_display()

    def showEvent(self, event):
        """Handle window show event to ensure focus."""
        super().showEvent(event)
        self.setFocus()

    def end_game(self):
        """Handle game over and save statistics."""
        if hasattr(self, 'game_ended') and self.game_ended:
            return  # Already ended

        self.game_ended = True
        self.game_duration = time.time() - self.game_start_time

        # Update game stats in new user data structure
        bubble_stats = self.user_data['stats']['game_stats']['bubble']
        bubble_stats['attempts'] += 1
        bubble_stats['bubbles_popped'] += self.bubbles_popped
        bubble_stats['last_score'] = self.score
        bubble_stats['total_score'] += self.score

        # Update highest level if achieved
        if self.level > bubble_stats['highest_level']:
            bubble_stats['highest_level'] = self.level

        # Calculate and update accuracy
        if self.shots_taken > 0:
            accuracy = self.bubbles_popped / self.shots_taken
            bubble_stats['accuracy'] = accuracy

        # Add to game history
        game_history = {
            'date': datetime.now().isoformat(),
            'score': self.score,
            'level': self.level,
            'bubbles': self.bubbles_popped,
            'duration': int(self.game_duration),
            'words': list(self.words_encountered),
            'shots': self.shots_taken,
            'accuracy': accuracy if self.shots_taken > 0 else 0,
            'result': 'win' if self.check_win_condition() else 'loss'
        }
        self.user_data['game_history']['bubble'].append(game_history)

        # Update practice stats if this counts as practice
        practice_stats = self.user_data['stats']['practice_stats']
        if len(self.words_encountered) > 0:
            practice_stats['total_time'] += int(self.game_duration / 60)  # in minutes

            # Update streak information
            today = datetime.now().date()
            last_practiced = practice_stats['last_practiced']

            if last_practiced:
                last_date = datetime.fromisoformat(last_practiced).date()
                if (today - last_date).days == 1:  # Consecutive day
                    practice_stats['current_streak'] += 1
                    if practice_stats['current_streak'] > practice_stats['longest_streak']:
                        practice_stats['longest_streak'] = practice_stats['current_streak']
                elif (today - last_date).days > 1:  # Broken streak
                    practice_stats['current_streak'] = 1
            else:  # First practice
                practice_stats['current_streak'] = 1
                practice_stats['longest_streak'] = max(practice_stats['longest_streak'], 1)

            practice_stats['last_practiced'] = today.isoformat()

            # Update daily goal tracking
            if len(self.words_encountered) >= practice_stats['daily_goal']:
                practice_stats['goal_streak'] += 1

        # Update efficiency metrics
        efficiency = self.user_data['stats']['efficiency']
        if self.game_duration > 0:
            efficiency['words_per_hour'] = len(self.words_encountered) / (self.game_duration / 3600)
            # Simple implementation - could be enhanced
            efficiency['games_per_week'] = min(7, efficiency.get('games_per_week', 0) + 1)
            efficiency['improvement_rate'] = (self.score - bubble_stats.get('last_score', 0)) / max(1, bubble_stats.get(
                'last_score', 1))

        # Update meta info
        self.user_data['meta']['last_updated'] = datetime.now().isoformat()

        # Save through parent if available
        if hasattr(self, 'parent') and self.parent():
            self.parent().save_user_data()

    def show_game_over(self):
        """Show the game over screen with final stats."""
        self.final_score_label.setText(
            f"Final Score: {self.score}\nLevel Reached: {self.level}\nBubbles Popped: {self.bubbles_popped}")
        self.stacked_widget.setCurrentWidget(self.game_over_page)
        self.game_over_page.setFocus()  # Set focus to the game over page

    def load_target_word(self):
        """Load a new target word from words currently present and accessible in the grid"""
        # First find all words in the current grid that aren't being popped
        available_words = set()
        for bubble in self.bubbles:
            if bubble['pop_animation'] == 0:
                available_words.add(bubble['word'])

        # If no bubbles left, we've cleared the level - generate new grid first
        if not available_words:
            self.generate_bubble_grid()
            return self.load_target_word()

        # Now find which of these words are actually reachable
        accessible_words = set()
        for word in available_words:
            # Find all bubbles with this word
            word_bubbles = [b for b in self.bubbles
                            if b['word'] == word and b['pop_animation'] == 0]

            # Check if any of them are reachable
            for bubble in word_bubbles:
                if self.has_clear_path(self.shooter_x, self.shooter_y, bubble['x'], bubble['y']):
                    accessible_words.add(word)
                    break

        # If we have accessible words, pick one randomly
        if accessible_words:
            self.current_target_word = random.choice(list(accessible_words))
        else:
            # If no words are accessible, pick the one closest to being reachable
            # This is a fallback - we should try to make sure this case doesn't happen
            closest_word = None
            min_distance = float('inf')

            for word in available_words:
                word_bubbles = [b for b in self.bubbles
                                if b['word'] == word and b['pop_animation'] == 0]

                for bubble in word_bubbles:
                    distance = math.sqrt((self.shooter_x - bubble['x']) ** 2 +
                                         (self.shooter_y - bubble['y']) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_word = word

            self.current_target_word = closest_word

        # Ensure the target word has a color assigned
        self.assign_color(self.current_target_word)

        # Load video demonstration
        avatars_folder = self.drive_manager.get_folder_id("Avatars")
        if not avatars_folder:
            return

        video_name = f"{self.current_target_word}.mp4"
        video_files = self.drive_manager.list_files(avatars_folder['id'], mime_type='video/mp4')
        video_file = next((f for f in video_files if f['name'].lower() == video_name.lower()), None)

        if not video_file:
            # Fallback to Hello.mp4 if specific word video not found
            video_file = next((f for f in video_files if f['name'].lower() == "hello.mp4"), None)
            if not video_file:
                return

        # Download and play the video
        temp_path = os.path.join(tempfile.gettempdir(), video_file['name'])
        self.drive_manager.download_file(video_file['id'], temp_path)
        self.temp_video_files.append(temp_path)

        # Stop any current playback
        self.media_player.stop()

        # Set new media
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(temp_path)))
        self.media_player.play()

        # Enable replay button
        self.replay_btn.setEnabled(True)
        # Track this word as encountered
        self.words_encountered.add(self.current_target_word)

    def replay_demo(self):
        """Replay the current target word's demo video."""
        if hasattr(self, 'current_target_word') and self.current_target_word:
            # Stop any current playback
            self.media_player.stop()

            # Set the media again to restart from beginning
            if hasattr(self, 'media_player') and self.media_player.media().isNull():
                # If media was cleared, reload it
                self.load_target_word()
            else:
                # Just replay the current media
                self.media_player.setPosition(0)
                self.media_player.play()

    def generate_bubble_grid(self):
        """Create the initial grid of bubbles"""
        self.bubbles = []
        rows = 5
        bubble_radius = 30
        vertical_spacing = bubble_radius * 1.9

        # Calculate available width (accounting for small margins)
        available_width = self.width() - 40

        # Calculate how many columns fit and exact spacing
        cols = max(8, int(available_width / (bubble_radius * 2.1)))
        exact_spacing = available_width / (cols - 0.5)  # Account for hexagonal offset

        # Calculate starting positions
        start_x = 20  # Left margin
        start_y = 100  # From top

        # Get available words (exclude target word to ensure variety)
        available_words = [w for w in self.words if w != self.current_target_word]
        if len(available_words) < 5:
            available_words = self.words.copy()
            if self.current_target_word in available_words:
                available_words.remove(self.current_target_word)

        # Assign colors
        for word in available_words + [self.current_target_word]:
            self.assign_color(word)

        # Create grid
        for row in range(rows):
            row_offset = (row % 2) * (exact_spacing / 2)

            for col in range(cols):
                x = start_x + col * exact_spacing + row_offset
                y = start_y + row * vertical_spacing

                # Select word - ensure target word appears at least once per row
                if col == 0 and row % 2 == 0:  # First column every other row
                    word = self.current_target_word
                else:
                    word = random.choice(available_words)

                # Get color
                color = self.get_color_for_word(word)

                self.bubbles.append({
                    'word': word,
                    'color': color,
                    'original_color': color,
                    'x': x,
                    'y': y,
                    'radius': bubble_radius,
                    'pop_animation': 0,
                    'bounce_offset': 0,
                    'bounce_direction': 1
                })

                self.active_words.add(word)

        # Adjust shooter position to match new grid
        if self.bubbles:
            rightmost = max(b['x'] for b in self.bubbles)
            leftmost = min(b['x'] for b in self.bubbles)
            grid_width = rightmost - leftmost + (bubble_radius * 2)
            self.shooter_x = leftmost + (grid_width // 2)
            self.shooter_y = self.height() - 50

    def _force_accessible_target(self):
        """Ensure target word appears in at least one accessible position"""
        # First clear any existing target bubbles
        self.bubbles = [b for b in self.bubbles if b['word'] != self.current_target_word]

        # Generate positions that are definitely reachable
        accessible_positions = [
            (self.width() // 2, 100),  # Center top
            (self.width() // 3, 150),  # Left side
            (2 * self.width() // 3, 150)  # Right side
        ]

        # Add target bubble to accessible position
        for x, y in accessible_positions:
            if self.has_clear_path(self.shooter_x, self.shooter_y, x, y):
                self.bubbles.append({
                    'word': self.current_target_word,
                    'color': self.get_color_for_word(self.current_target_word),
                    'original_color': self.get_color_for_word(self.current_target_word),
                    'x': x,
                    'y': y,
                    'radius': 30,
                    'pop_animation': 0,
                    'bounce_offset': 0,
                    'bounce_direction': 1
                })
                break

    def is_target_accessible(self):
        """Check if at least one target bubble is reachable"""
        if not self.bubbles or not self.current_target_word:
            return False

        # Find all target bubbles
        target_bubbles = [b for b in self.bubbles
                          if b['word'] == self.current_target_word and b['pop_animation'] == 0]

        if not target_bubbles:
            return False

        # Check path to each target bubble
        shooter_x = self.width() // 2
        shooter_y = self.height()

        for bubble in target_bubbles:
            if self.has_clear_path(shooter_x, shooter_y, bubble['x'], bubble['y']):
                return True

        return False

    def has_clear_path(self, x1, y1, x2, y2):
        """More robust path checking with angle validation"""
        # First check direct line of sight
        if not self._has_line_of_sight(x1, y1, x2, y2):
            return False

        # Then check if angle is within shooter's range
        angle = math.atan2(y1 - y2, x2 - x1)  # Angle in radians
        return (math.pi / 4 <= angle <= 3 * math.pi / 4)  # Between 45 and 135 degrees

    def _has_line_of_sight(self, x1, y1, x2, y2):
        """Check if there's a clear straight path between two points"""
        steps = 100
        bubble_radius = 30 if not self.bubbles else self.bubbles[0]['radius']

        for i in range(steps + 1):
            x = x1 + (x2 - x1) * i / steps
            y = y1 + (y2 - y1) * i / steps

            # Check collision with any bubble
            for bubble in self.bubbles:
                if bubble['pop_animation'] > 0:
                    continue

                distance = math.sqrt((x - bubble['x']) ** 2 + (y - bubble['y']) ** 2)
                if distance < bubble_radius:
                    return False

        return True

    def release_word_color(self, word):
        """Release a word's color when it's no longer on screen"""
        if word in self.active_words:
            self.active_words.remove(word)

    def assign_color(self, word):
        """Assign a unique color to a word, reusing colors only when necessary"""
        if word in self.word_colors:
            return  # Color already assigned

        # Find an available color not currently in use
        used_colors = {self.word_colors[w] for w in self.active_words if w in self.word_colors}
        available_colors = [rgb for rgb in self.color_palette_rgb if rgb not in used_colors]

        if available_colors:
            # Use an available color from palette
            self.word_colors[word] = random.choice(available_colors)
        else:
            # Recycle least recently used color (simple approach)
            self.word_colors[word] = random.choice(self.color_palette_rgb)

        self.active_words.add(word)

    def get_color_for_word(self, word):
        """Get QColor object for a word"""
        if word not in self.word_colors:
            self.assign_color(word)
        rgb = self.word_colors[word]
        return QColor(*rgb)

    def play_sound(self, sound_type):
        """Play sound effects for the game using Pygame"""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            # Create simple sound effects programmatically
            if sound_type == "shoot":
                # Create a short "pop" sound
                sound = pygame.mixer.Sound(buffer=self._generate_pop_sound())
            elif sound_type == "match":
                # Create a "ting" success sound
                sound = pygame.mixer.Sound(buffer=self._generate_ting_sound())
            elif sound_type == "miss":
                # Create a "buzz" error sound
                sound = pygame.mixer.Sound(buffer=self._generate_buzz_sound())
            elif sound_type == "levelup":
                # Create a fanfare sound
                sound = pygame.mixer.Sound(buffer=self._generate_fanfare_sound())
            else:
                return

            sound.play()

        except Exception as e:
            print(f"Error playing sound: {e}")

    def _generate_pop_sound(self):
        """Generate a simple pop sound"""
        arr = np.zeros(44100 // 2, dtype=np.int16)  # 0.5 second sound
        for i in range(len(arr)):
            arr[i] = 32767 * np.sin(2 * np.pi * 440 * i / 44100) * \
                     np.exp(-0.0005 * i)  # Exponential decay
        return arr.tobytes()

    def _generate_ting_sound(self):
        """Generate a success ting sound"""
        arr = np.zeros(44100 // 3, dtype=np.int16)  # 0.33 second sound
        for i in range(len(arr)):
            freq = 880 + 440 * np.sin(2 * np.pi * 2 * i / 44100)
            arr[i] = 32767 * np.sin(2 * np.pi * freq * i / 44100) * \
                     np.exp(-0.001 * i)
        return arr.tobytes()

    def _generate_buzz_sound(self):
        """Generate an error buzz sound"""
        arr = np.zeros(44100 // 4, dtype=np.int16)  # 0.25 second sound
        for i in range(len(arr)):
            freq = 220 + 110 * np.sin(2 * np.pi * 10 * i / 44100)
            arr[i] = 32767 * np.sin(2 * np.pi * freq * i / 44100) * \
                     (1 - i / len(arr))  # Linear decay
        return arr.tobytes()

    def _generate_fanfare_sound(self):
        """Generate a level-up fanfare"""
        arr = np.zeros(44100, dtype=np.int16)  # 1 second sound
        for i in range(len(arr)):
            if i < len(arr) // 3:
                freq = 440 + 220 * np.sin(2 * np.pi * 1 * i / 44100)
            elif i < 2 * len(arr) // 3:
                freq = 660 + 220 * np.sin(2 * np.pi * 1 * i / 44100)
            else:
                freq = 880 + 220 * np.sin(2 * np.pi * 1 * i / 44100)
            arr[i] = 32767 * np.sin(2 * np.pi * freq * i / 44100) * \
                     (1 - 0.5 * i / len(arr))
        return arr.tobytes()

    def update_game(self):
        """Main game loop"""
        # Move projectiles
        for proj in getattr(self, 'projectiles', []):
            proj['x'] += proj['dx']
            proj['y'] += proj['dy']

            # Add to trail
            proj['trail'].append((proj['x'], proj['y']))
            if len(proj['trail']) > 5:
                proj['trail'].pop(0)

            # Check wall collisions with proper bounce physics
            if proj['x'] <= proj['radius']:  # Left wall
                proj['x'] = proj['radius']  # Prevent sticking
                proj['dx'] = abs(proj['dx'])  # Bounce right
            elif proj['x'] >= self.width() - proj['radius']:  # Right wall
                proj['x'] = self.width() - proj['radius']  # Prevent sticking
                proj['dx'] = -abs(proj['dx'])  # Bounce left

            # Check ceiling collision
            if proj['y'] <= proj['radius']:
                proj['y'] = proj['radius']
                proj['dy'] = abs(proj['dy'])  # Bounce down

            # Check if projectile went off bottom of screen
            if proj['y'] > self.height() + proj['radius']:
                self.projectiles.remove(proj)
                self.projectile_active = False
                continue

            # Check bubble collisions
            for bubble in self.bubbles:
                if bubble['pop_animation'] > 0:
                    continue

                distance = math.sqrt((proj['x'] - bubble['x']) ** 2 + (proj['y'] - bubble['y']) ** 2)
                if distance < proj['radius'] + bubble['radius']:
                    if proj['word'] == bubble['word']:
                        matches = self.remove_matching_bubbles(bubble['word'], bubble['x'], bubble['y'])
                        self.projectiles.remove(proj)
                        self.projectile_active = False  # Clear flag when projectile hits
                        break
                    else:
                        self.lives -= 1
                        self.update_lives_display()
                        self.projectiles.remove(proj)
                        self.projectile_active = False  # Clear flag when projectile hits
                        if self.lives <= 0:
                            self.game_over()
                        else:
                            bubble['color'] = QColor(255, 0, 0)
                            QTimer.singleShot(200, lambda b=bubble: self.reset_bubble_color(b))
                        break

        # Move grid down periodically
        self.grid_drop_counter += 1
        if self.grid_drop_counter >= 60:
            self.grid_drop_counter = 0
            for bubble in self.bubbles:
                bubble['y'] += self.grid_speed

            # Check if grid reached bottom
            lowest_bubble = max(bubble['y'] for bubble in self.bubbles) if self.bubbles else 0

            if lowest_bubble > self.height() - 300:
                self.game_over()

        self.update_score_display()
        self.update_canvas()
        # Check for game over
        if self.lives <= 0:
            self.end_game()
            return
        # Check for win condition
        if self.check_win_condition():
            self.show_victory_screen()
            return

        # Check for game over
        if self.lives <= 0:
            self.show_game_over()
            return

    def reset_bubble_color(self, bubble):
        """Reset bubble color after wrong match feedback"""
        if bubble in self.bubbles:
            bubble['color'] = bubble['original_color']

    def update_animations(self):
        """Update all animations in the game"""
        needs_redraw = False

        # Update bubble pop animations
        for bubble in self.bubbles[:]:  # Use a copy for iteration
            if bubble['pop_animation'] > 0:
                bubble['pop_animation'] += 0.5
                needs_redraw = True
                if bubble['pop_animation'] >= 10:
                    self.bubbles.remove(bubble)

            # Update bounce animations
            if bubble['bounce_offset'] != 0:
                bubble['bounce_offset'] += bubble['bounce_direction'] * 2
                if abs(bubble['bounce_offset']) >= 10:
                    bubble['bounce_direction'] *= -1
                if bubble['bounce_offset'] == 0:
                    bubble['bounce_direction'] = 0
                needs_redraw = True

        if needs_redraw:
            self.update_canvas()

    def remove_matching_bubbles(self, word, x, y):
        """Remove all connected bubbles with the same word using flood fill algorithm"""
        if not self.bubbles:
            return 0

        # Find all connected bubbles with the same word
        bubble_radius = self.bubbles[0]['radius']
        connected_bubbles = []
        queue = []

        # Find the bubble that was hit
        for bubble in self.bubbles:
            if (bubble['word'] == word and
                    math.sqrt((x - bubble['x']) ** 2 + (y - bubble['y']) ** 2) < bubble_radius * 2):
                queue.append(bubble)
                break

        # Flood fill algorithm to find connected same-word bubbles
        while queue:
            current = queue.pop()
            if current in connected_bubbles:
                continue

            connected_bubbles.append(current)

            # Check all 4 directions (up, down, left, right)
            directions = [
                (0, -1),  # Up
                (0, 1),  # Down
                (-1, 0),  # Left
                (1, 0)  # Right
            ]

            for dx, dy in directions:
                # Calculate approximate position of adjacent bubble
                check_x = current['x'] + dx * bubble_radius * 2
                check_y = current['y'] + dy * bubble_radius * 2

                # Find bubbles in this position with same word
                for bubble in self.bubbles:
                    if (bubble not in connected_bubbles and
                            bubble['word'] == word and
                            bubble['pop_animation'] == 0 and
                            math.sqrt(
                                (check_x - bubble['x']) ** 2 + (check_y - bubble['y']) ** 2) < bubble_radius * 1.5):
                        queue.append(bubble)

        # Mark bubbles for removal with animation
        for bubble in connected_bubbles:
            bubble['pop_animation'] = 1  # Start pop animation
            self.bubbles_popped += 1
            self.release_word_color(bubble['word'])

        # Calculate score - base points multiplied by combo multiplier
        combo_multiplier = min(5, len(connected_bubbles))  # Cap multiplier at 5x
        points_earned = 10 * self.level * combo_multiplier
        self.score += points_earned

        # Play sound effect based on combo size
        if len(connected_bubbles) >= 3:
            self.play_sound("match")
        else:
            self.play_sound("pop")

        # Check if we should level up (after every 10 matches)
        self.matches_found += 1
        if self.matches_found > 0 and self.matches_found % 10 == 0:
            self.level_up()

        self.load_target_word()
        return len(connected_bubbles)

    def level_up(self):
        """Increase difficulty with visual feedback"""
        self.level += 1
        self.grid_speed += 0.5

        # Visual feedback for level up
        for bubble in self.bubbles:
            bubble['bounce_offset'] = 15
            bubble['bounce_direction'] = -1

        self.play_sound("levelup")
        self.generate_bubble_grid()  # Add more bubbles

    def update_lives_display(self):
        """Update the lives display with hearts and X for lost lives"""
        for i, heart in enumerate(self.hearts):
            if i < self.lives:
                # Show colored heart for remaining lives
                heart.setText("❤")
                heart.setStyleSheet("""
                    QLabel {
                        font-size: 24px;
                        color: red;
                        margin: 0 2px;
                    }
                """)
            else:
                # Show gray X for lost lives
                heart.setText("✖")
                heart.setStyleSheet("""
                    QLabel {
                        font-size: 24px;
                        color: #888;
                        margin: 0 2px;
                    }
                """)

    def update_score_display(self):
        """Update the score display with data from the new structure"""
        stats = self.user_data['stats']
        game_stats = stats['game_stats']
        practice_stats = stats['practice_stats']

        # Calculate accuracy
        accuracy = 0
        if practice_stats['words_attempted'] > 0:
            accuracy = (practice_stats['words_correct'] /
                        practice_stats['words_attempted']) * 100

        # Create detailed score text
        score_text = (f"📊 Stats: "
                      f"Sessions: {practice_stats['sessions']} | "
                      f"Accuracy: {accuracy:.1f}% | "
                      f"Streak: {practice_stats['current_streak']} days\n"
                      f"🎮 Games: "
                      f"Quiz: {game_stats['quiz']['last_score']} | "
                      f"Matching: {game_stats['matching']['last_score']} | "
                      f"Bubble: {game_stats['bubble']['last_score']}")

        # Update or create the score display
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
            if hasattr(self, 'title_layout'):
                self.title_layout.addWidget(self.score_display, 0, Qt.AlignRight)

        # Force immediate UI update
        QApplication.processEvents()

    def draw_bubbles(self, painter):
        """Draw all bubbles using their assigned word colors"""
        try:
            for bubble in self.bubbles:
                draw_y = bubble['y'] + bubble['bounce_offset']

                if bubble['pop_animation'] > 0:
                    # Draw popping animation
                    radius = bubble['radius'] * (1 + bubble['pop_animation'] / 15)
                    alpha = 255 - (bubble['pop_animation'] * 25)
                    color = QColor(bubble['color'])
                    color.setAlpha(alpha)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(color)
                    painter.drawEllipse(QPointF(bubble['x'], draw_y), radius, radius)
                else:
                    # Draw shadow
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(0, 0, 0, 100))
                    painter.drawEllipse(QPointF(bubble['x'] + 3, draw_y + 3),
                                        bubble['radius'], bubble['radius'])

                    # Draw main bubble with assigned word color
                    painter.setBrush(bubble['color'])
                    painter.drawEllipse(QPointF(bubble['x'], draw_y),
                                        bubble['radius'], bubble['radius'])

                    # Draw highlight
                    highlight_color = QColor(255, 255, 255, 150)
                    painter.setBrush(highlight_color)
                    painter.drawEllipse(
                        QPointF(bubble['x'] - bubble['radius'] / 3,
                                draw_y - bubble['radius'] / 3),
                        bubble['radius'] / 3, bubble['radius'] / 3
                    )

                    # Draw word text
                    painter.setPen(Qt.white)
                    painter.setFont(self.word_font)
                    text_rect = QRectF(bubble['x'] - bubble['radius'],
                                       draw_y - bubble['radius'],
                                       bubble['radius'] * 2,
                                       bubble['radius'] * 2)
                    painter.drawText(text_rect, Qt.AlignCenter | Qt.TextWordWrap, bubble['word'])

        except Exception as e:
            print(f"Error drawing bubbles: {e}")

    def game_over(self):
        """Handle game over with new screen"""
        self.timer.stop()
        self.animation_timer.stop()
        self.end_game()  # Ensure stats are saved
        self.show_game_over()

    def check_win_condition(self):
        """Check if all bubbles have been popped"""
        for bubble in self.bubbles:
            if bubble['pop_animation'] == 0:  # If any bubble isn't popped
                return False
        return True

    def setup_victory_page(self):
        """Setup the victory page showing game results"""
        self.victory_page = QWidget()
        layout = QVBoxLayout(self.victory_page)
        layout.setAlignment(Qt.AlignCenter)

        # Victory title
        victory_label = QLabel("🎉 Victory! 🎉")
        victory_label.setStyleSheet("""
            QLabel {
                font-size: 48px;
                font-weight: bold;
                color: gold;
                margin-bottom: 30px;
            }
        """)
        layout.addWidget(victory_label)

        # Game stats display
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: white;
                margin-bottom: 30px;
            }
        """)
        layout.addWidget(self.stats_label)

        # Button container
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)

        # Play Again button
        play_again_btn = QPushButton("🔄 Play Again")
        play_again_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 18px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        play_again_btn.clicked.connect(self.restart_game)
        button_layout.addWidget(play_again_btn)

        # Home button
        home_btn = QPushButton("🏠 Return Home")
        home_btn.setStyleSheet("""
            QPushButton {
                background-color: #6e8efb;
                color: white;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 18px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #5a7df4;
            }
        """)
        home_btn.clicked.connect(self.close)
        button_layout.addWidget(home_btn)

        layout.addWidget(button_container)
        layout.addStretch()

        self.stacked_widget.addWidget(self.victory_page)

    def show_victory_screen(self):
        """Show victory screen with game stats"""
        self.end_game()
        # Calculate game duration
        duration = time.time() - self.game_start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)

        # Prepare stats text
        stats_text = (
            f"Final Score: {self.score}\n"
            f"Level Reached: {self.level}\n"
            f"Bubbles Popped: {self.bubbles_popped}\n"
            f"Time: {minutes}m {seconds}s\n"
            f"Accuracy: {self.calculate_accuracy():.1f}%"
        )

        self.stats_label.setText(stats_text)
        self.stacked_widget.setCurrentWidget(self.victory_page)
        self.victory_page.setFocus()  # Set focus to the victory page

    def calculate_accuracy(self):
        """Calculate shooting accuracy"""
        if self.shots_taken == 0:
            return 0.0
        return (self.bubbles_popped / self.shots_taken) * 100

    def restart_game(self):
        """Restart the game with fresh stats."""
        # Reset game state
        self.score = 0
        self.lives = 3
        self.level = 1
        self.bubbles_popped = 0
        self.words_encountered = set()
        self.shots_taken = 0
        self.matches_found = 0
        self.game_ended = False

        # Reset control variables
        self.rotate_left = False
        self.rotate_right = False
        self.launcher_angle = math.pi / 2  # Reset to default angle

        # Clear game objects
        self.bubbles = []
        self.projectiles = []

        # Stop and restart timers
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        if hasattr(self, 'animation_timer') and self.animation_timer.isActive():
            self.animation_timer.stop()
        if hasattr(self, 'keyboard_timer') and self.keyboard_timer.isActive():
            self.keyboard_timer.stop()

        # Restart timers
        self.keyboard_timer.start(16)
        self.animation_timer.start(16)

        # Start new game
        self.start_game()
        self.update_score_display()
        self.update_lives_display()

        # Ensure focus is on the game window
        self.setFocus()
        self.raise_()  # Bring window to front
        self.activateWindow()  # Activate window

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
        if self.lives > 0:  # Only call end_game if game was in progress
            self.end_game()
        event.accept()
        # Stop keyboard timer
        if hasattr(self, 'keyboard_timer') and self.keyboard_timer.isActive():
            self.keyboard_timer.stop()

        if hasattr(self, 'media_player') and self.media_player:
            self.media_player.stop()
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()

        if hasattr(self, 'animation_timer') and self.animation_timer.isActive():
            self.animation_timer.stop()

        if hasattr(self, 'media_player'):
            self.media_player.stop()

        event.accept()


class SignMatchingGame(QMainWindow):
    finished = pyqtSignal()

    def __init__(self, words, user_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sign Matching Game")
        self.setWindowModality(Qt.WindowModal)
        self.setFixedSize(1000, 700)

        self.words = words
        self.user_data = user_data
        self.score = 0
        self.matches_found = 0
        self.total_pairs = 8  # 4 word-video pairs (8 cards total)
        self.first_selection = None
        self.second_selection = None
        self.matched_indices = set()
        self.game_active = False
        self.temp_video_files = []
        self.drive_manager = parent.drive_manager if parent else None
        self.elapsed = 0
        # Game metrics
        self.start_time = None
        self.end_time = None
        self.moves = 0
        self.accuracy = 0.0
        self.current_streak = 0
        self.longest_streak = 0

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
        title = QLabel("Sign Matching Game")
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #4a6baf;
            margin-bottom: 20px;
        """)
        layout.addWidget(title, 0, Qt.AlignCenter)

        # Game instructions
        instructions = QLabel(
            "How to Play:\n\n"
            "1. Flip over cards to find matching pairs\n"
            "2. Match sign language videos to their English words\n"
            "3. Find all pairs to complete the game\n"
            "4. The faster you complete, the more points you earn\n\n"
            "Tips:\n"
            "- Try to remember card positions\n"
            "- Watch the full video when you find one\n"
            "- Complete quickly to earn bonus points"
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("""
            font-size: 18px;
            margin-bottom: 30px;
        """)
        layout.addWidget(instructions)

        # Start button
        start_btn = QPushButton("Start Game")
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
        home_btn.clicked.connect(self.close)
        layout.addWidget(home_btn, 0, Qt.AlignCenter)

        layout.addStretch()
        self.stacked_widget.addWidget(self.instructions_page)

    def setup_game_page(self):
        """Setup the main game play page."""
        self.game_page = QWidget()
        main_layout = QVBoxLayout(self.game_page)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Game info header
        header = QWidget()
        header_layout = QHBoxLayout(header)

        self.score_label = QLabel("Score: 0")
        self.score_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        header_layout.addWidget(self.score_label)

        self.timer_label = QLabel("Time: 0s")
        self.timer_label.setStyleSheet("font-size: 18px;")
        header_layout.addWidget(self.timer_label)

        header_layout.addStretch()

        quit_btn = QPushButton("Quit Game")
        quit_btn.setStyleSheet("""
            QPushButton {
                padding: 5px 10px;
                background-color: #f44336;
                color: white;
                border-radius: 4px;
            }
        """)
        quit_btn.clicked.connect(self.close)
        header_layout.addWidget(quit_btn)

        main_layout.addWidget(header)

        # Card grid
        self.card_grid = QGridLayout()
        self.card_grid.setSpacing(10)

        # Create card widgets
        self.cards = []
        for i in range(16):  # 8 pairs = 16 cards
            card = QPushButton()
            card.setFixedSize(120, 120)
            card.setStyleSheet("""
                QPushButton {
                    background-color: #6e8efb;
                    border-radius: 8px;
                    font-size: 24px;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #5a7df4;
                }
            """)
            card.clicked.connect(lambda _, idx=i: self.card_clicked(idx))
            self.cards.append(card)
            self.card_grid.addWidget(card, i // 4, i % 4)

        main_layout.addLayout(self.card_grid)
        main_layout.addStretch()

        self.stacked_widget.addWidget(self.game_page)

    def setup_results_page(self):
        """Setup the results page."""
        self.results_page = QWidget()
        layout = QVBoxLayout(self.results_page)
        layout.setAlignment(Qt.AlignCenter)

        # Results title
        title = QLabel("Game Results")
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

    def start_game(self):
        """Start the game with exactly 8 unique words (16 cards)."""
        # Reset game metrics
        self.start_time = time.time()
        self.end_time = None
        self.moves = 0
        self.matches_found = 0
        self.current_streak = 0
        self.longest_streak = 0
        self.score = 0
        self.matched_indices = set()
        self.first_selection = None
        self.second_selection = None
        self.game_active = False  # Will be set to True after setup

        # Reset all cards
        for card in self.cards:
            card.setEnabled(True)
            card.setText("?")
            card.setStyleSheet("""
                QPushButton {
                    background-color: #6e8efb;
                    border-radius: 8px;
                    font-size: 24px;
                    color: white;
                }
            """)

        # Check if we have enough words
        if len(self.words) < 8:
            QMessageBox.warning(self, "Not Enough Words",
                                "Need at least 8 different words to play!")
            self.close()
            return

        # Select exactly 8 unique words
        unique_words = random.sample(self.words, 8)

        # Create pairs - each word has both a word card and video card
        self.game_pairs = []
        for word in unique_words:
            self.game_pairs.append(("word", word))
            self.game_pairs.append(("video", word))

        # Verify we have exactly 16 items (8 pairs)
        if len(self.game_pairs) != 16:
            QMessageBox.critical(self, "Setup Error",
                                 f"Expected 16 cards, got {len(self.game_pairs)}")
            self.close()
            return

        # Shuffle the pairs
        random.shuffle(self.game_pairs)

        # Debug output
        print(f"Game setup with {len(unique_words)} unique words")
        print("Card assignments:")
        for i, (card_type, word) in enumerate(self.game_pairs):
            print(f"Card {i}: {card_type} - {word}")

        # Start game
        self.game_active = True
        self.start_time = time.time()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)

        self.update_score_display()
        self.stacked_widget.setCurrentWidget(self.game_page)

    def card_clicked(self, index):
        """Handle card clicks with visual feedback."""
        if (not self.game_active or
                index in self.matched_indices or
                (self.first_selection is not None and self.second_selection is not None) or
                index == self.first_selection):
            return

        # Show card content immediately
        self.show_card_content(index)

        # Handle selection logic
        if self.first_selection is None:
            self.first_selection = index
        else:
            self.second_selection = index
            QTimer.singleShot(1000, self.check_match)  # Wait before checking match

    def reset_card_style(self, index):
        """Reset card to default appearance."""
        self.cards[index].setStyleSheet("""
            QPushButton {
                background-color: #6e8efb;
                border-radius: 8px;
                font-size: 24px;
                color: white;
            }
        """)

    def show_card_content(self, index):
        """Show card content with visual feedback."""
        try:
            card_type, word = self.game_pairs[index]
            card = self.cards[index]

            if card_type == "word":
                card.setText(word)
                card.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;  /* Green for word cards */
                        border-radius: 8px;
                        font-size: 18px;
                        color: white;
                        border: 2px solid #388E3C;
                    }
                """)
            else:  # video
                card.setText("▶")
                card.setStyleSheet("""
                    QPushButton {
                        background-color: #FF9800;  /* Orange for video cards */
                        border-radius: 8px;
                        font-size: 24px;
                        color: white;
                        border: 2px solid #F57C00;
                    }
                """)
                # Play video preview after visual feedback
                QTimer.singleShot(100, lambda: self.play_video_preview(word))

        except Exception as e:
            print(f"Error showing card content: {str(e)}")
            self.reset_card_style(index)

    def update_timer(self):
        """Update the game timer."""
        self.elapsed = int(time.time() - self.start_time)
        self.timer_label.setText(f"Time: {self.elapsed}s")

    def play_video_preview(self, word):
        """Play a short preview of the sign video."""
        try:
            # Find the video in Google Drive
            avatars_folder = self.drive_manager.get_folder_id("Avatars")
            if not avatars_folder:
                return

            video_name = f"{word}.mp4"
            video_files = self.drive_manager.list_files(avatars_folder['id'], mime_type='video/mp4')
            video_file = next((f for f in video_files if f['name'].lower() == video_name.lower()), None)

            if not video_file:
                # Fallback to Hello.mp4 if specific word video not found
                video_file = next((f for f in video_files if f['name'].lower() == "hello.mp4"), None)
                if not video_file:
                    return

            # Download and play the video
            temp_path = os.path.join(tempfile.gettempdir(), video_file['name'])
            self.drive_manager.download_file(video_file['id'], temp_path)
            self.temp_video_files.append(temp_path)

            # Show preview dialog
            preview = QDialog(self)
            layout = QVBoxLayout(preview)

            player = VideoPlayerWidget("")
            player.play_video(temp_path)
            player.media_player.setPlaybackRate(1.0)
            layout.addWidget(player)

            # Close after 3 seconds
            QTimer.singleShot(3000, preview.close)
            preview.exec_()

        except Exception as e:
            print(f"Error playing video preview: {e}")

    def check_match(self):
        """Check if the two selected cards match."""
        self.moves += 1  # Count this attempt
        if self.first_selection is None or self.second_selection is None:
            return

        # Get the word from both cards
        _, word1 = self.game_pairs[self.first_selection]
        _, word2 = self.game_pairs[self.second_selection]

        if word1 == word2:  # Match!
            self.matches_found += 1
            self.current_streak += 1
            self.longest_streak = max(self.longest_streak, self.current_streak)
            self.score += 100  # Base points per match

            # Add time bonus
            elapsed = time.time() - self.start_time
            time_bonus = max(0, 50 - int(elapsed / 5))
            self.score += time_bonus

            # Mark cards as matched
            self.matched_indices.add(self.first_selection)
            self.matched_indices.add(self.second_selection)

            # Disable matched cards and change color
            for idx in [self.first_selection, self.second_selection]:
                self.cards[idx].setEnabled(False)
                self.cards[idx].setStyleSheet("""
                    QPushButton {
                        background-color: #8D83F6;
                        border-radius: 8px;
                        font-size: 24px;
                        color: white;
                        border: 2px solid #388E3C;
                    }
                """)

            # Reset selections immediately after match
            self.first_selection = None
            self.second_selection = None

            # Check if game is complete
            if self.matches_found >= self.total_pairs:
                self.game_complete()
        else:  # No match
            # Flip cards back over after a delay
            self.current_streak = 0
            QTimer.singleShot(1000, self.flip_cards_back)
            # Update accuracy
        self.accuracy = (self.matches_found / self.moves) if self.moves > 0 else 0
        self.update_score_display()

    def flip_cards_back(self):
        """Flip non-matching cards back to face down state."""
        # First reset the card appearances
        if self.first_selection is not None and self.first_selection not in self.matched_indices:
            self.cards[self.first_selection].setText("?")
            self.cards[self.first_selection].setStyleSheet("""
                QPushButton {
                    background-color: #6e8efb;
                    border-radius: 8px;
                    font-size: 24px;
                    color: white;
                }
            """)

        if self.second_selection is not None and self.second_selection not in self.matched_indices:
            self.cards[self.second_selection].setText("?")
            self.cards[self.second_selection].setStyleSheet("""
                QPushButton {
                    background-color: #6e8efb;
                    border-radius: 8px;
                    font-size: 24px;
                    color: white;
                }
            """)

        # Then reset selections
        self.first_selection = None
        self.second_selection = None

    def game_complete(self):
        """Handle game completion with full statistics tracking."""
        self.end_time = time.time()
        duration = int(self.end_time - self.start_time)

        # Calculate final score with time bonus
        base_score = self.matches_found * 100
        time_bonus = max(0, 500 - int(duration))  # Up to 500 bonus points
        self.score = base_score + time_bonus

        # Update user stats
        today = datetime.now().isoformat()

        # Update game-specific stats
        if 'game_stats' not in self.user_data['stats']:
            self.user_data['stats']['game_stats'] = {
                'matching': {
                    'attempts': 0,
                    'matches': 0,
                    'best_time': None,
                    'last_score': 0,
                    'accuracy': 0.0
                }
            }

        matching_stats = self.user_data['stats']['game_stats']['matching']
        matching_stats['attempts'] += 1
        matching_stats['matches'] += self.matches_found
        matching_stats['last_score'] = self.score

        # Update best time if this was faster
        if matching_stats['best_time'] is None or duration < matching_stats['best_time']:
            matching_stats['best_time'] = duration

        # Update accuracy (weighted average)
        total_matches = matching_stats['matches']
        total_attempts = matching_stats['attempts'] * (self.total_pairs // 2)  # Total possible matches
        matching_stats['accuracy'] = total_matches / total_attempts if total_attempts > 0 else 0

        # Update practice stats
        if 'practice_stats' not in self.user_data['stats']:
            self.user_data['stats']['practice_stats'] = {
                'sessions': 0,
                'total_time': 0,
                'words_attempted': 0,
                'words_correct': 0,
                'current_streak': 0,
                'longest_streak': 0,
                'last_practiced': None,
                'daily_goal': 3,
                'goal_streak': 0
            }

        practice_stats = self.user_data['stats']['practice_stats']

        practice_stats['total_time'] += duration // 60  # Convert to minutes

        # Update streaks
        last_practice = practice_stats['last_practiced']
        if last_practice:
            last_date = datetime.fromisoformat(last_practice).date()
            today_date = datetime.now().date()

            if (today_date - last_date).days == 1:
                practice_stats['current_streak'] += 1
            elif (today_date - last_date).days > 1:
                practice_stats['current_streak'] = 1
        else:
            practice_stats['current_streak'] = 1

        practice_stats['longest_streak'] = max(practice_stats['longest_streak'],
                                               practice_stats['current_streak'])
        practice_stats['last_practiced'] = today

        # Add to game history
        if 'game_history' not in self.user_data:
            self.user_data['game_history'] = {'matching': []}

        self.user_data['game_history']['matching'].append({
            'date': today,
            'matches': self.matches_found,
            'time': duration,
            'accuracy': self.accuracy,
            'score': self.score,
            'streak': self.current_streak,
            'moves': self.moves
        })

        # Save through parent if available
        if hasattr(self, 'parent') and self.parent():
            self.parent().save_user_data()

        # Show detailed results
        self.show_results()

    def show_results(self):
        """Show detailed game results with stats and buttons."""
        self.results_page = QWidget()
        layout = QVBoxLayout(self.results_page)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        # Game title
        title = QLabel("Game Results")
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #4a6baf;
            margin-bottom: 20px;
        """)
        layout.addWidget(title)

        # Results grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(30)
        grid.setVerticalSpacing(10)

        # Add stats to grid
        stats = [
            ("Matches Found", f"{self.matches_found}/{self.total_pairs // 2}"),
            ("Total Score", str(self.score)),
            ("Time Taken", f"{int(self.end_time - self.start_time)} seconds"),
            ("Moves Made", str(self.moves)),
            ("Accuracy", f"{self.accuracy:.1%}"),
            ("Longest Streak", str(self.longest_streak)),
            ("New Words Learned", str(len(self.get_new_words())))
        ]

        for row, (label, value) in enumerate(stats):
            grid.addWidget(QLabel(f"<b>{label}:</b>"), row, 0, Qt.AlignRight)
            grid.addWidget(QLabel(value), row, 1, Qt.AlignLeft)

        layout.addLayout(grid)

        # Buttons
        btn_layout = QHBoxLayout()

        play_again_btn = QPushButton("Play Again")
        play_again_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        play_again_btn.clicked.connect(self.start_game)
        btn_layout.addWidget(play_again_btn)

        home_btn = QPushButton("Return Home")
        home_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 10px 20px;
                background-color: #6e8efb;
                color: white;
                border-radius: 8px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #5a7df4;
            }
        """)
        home_btn.clicked.connect(self.close)
        btn_layout.addWidget(home_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()

        # Replace old results page if it exists
        if hasattr(self, 'results_page'):
            self.stacked_widget.removeWidget(self.results_page)

        self.stacked_widget.addWidget(self.results_page)
        self.stacked_widget.setCurrentWidget(self.results_page)

    def get_new_words(self):
        """Return list of words learned in this session."""
        new_words = []
        for idx in self.matched_indices:
            _, word = self.game_pairs[idx]
            if word not in self.user_data.get('known_words', []):
                new_words.append(word)
        return new_words

    def update_score_display(self):
        """Update the score display."""
        self.score_label.setText(f"Score: {self.score}")

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
        self.correct = 0
        self.start_time = None  # To track game duration
        self.words_encountered = set()  # Track words seen in this game

        # Setup central widget and UI
        self.setup_ui()

        # Initialize MediaPipe
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def setup_ui(self):
        """Setup all UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Create stacked widget for different screens
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        # Setup all pages
        self.setup_instructions_page()
        self.setup_quiz_page()
        self.setup_results_page()

        # Start with instructions
        self.show_instructions()

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
        self.correct = 0
        self.current_round = 0
        self.streak = 0
        self.max_streak = 0
        self.words_encountered.clear()
        self.start_time = time.time()
        self.words_encountered = set()  # Track all words seen in this game
        self.correctly_answered_words = set()  # Track words answered correctly

        self.update_score_display()
        self.stacked_widget.setCurrentWidget(self.quiz_page)
        self.next_question()

    def next_question(self):
        """Load the next question."""
        if self.current_round >= self.total_rounds:
            self.show_results()
            return

        self.current_round += 1

        # Filter out words that were already answered correctly
        available_words = [word for word in self.words if word not in self.correctly_answered_words]

        # If we've run out of unique words, allow repeats
        if not available_words:
            available_words = self.words

        self.current_word = random.choice(available_words)
        self.words_encountered.add(self.current_word)

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
        """Load the demo video for the current word."""
        try:
            # Find the Avatars folder in Google Drive
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
            self.video_player.play_video(temp_path)
            self.video_player.media_player.setPlaybackRate(1.0)  # Normal speed

            # Store the temp path for cleanup
            self.temp_video_files.append(temp_path)

            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load demo video: {str(e)}")
            return False

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
            self.correct += 1

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
        QTimer.singleShot(3000, self.next_question)

    def show_results(self):
        """Show final results and update user data."""
        game_duration = int(time.time() - self.start_time)
        accuracy = (self.correct / self.total_rounds) * 100 if self.total_rounds > 0 else 0

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

        # Update user data with new structure
        self.update_user_stats(game_duration, accuracy)

        # Save user data through parent or directly
        if hasattr(self, 'parent') and self.parent():
            self.parent().save_user_data()
        else:
            self.save_user_data_directly()

        # Emit signal to update main UI
        self.finished.emit()
        self.stacked_widget.setCurrentWidget(self.results_page)

    def update_user_stats(self, duration, accuracy):
        """Update the user's stats with the new structure."""
        stats = self.user_data['stats']
        game_stats = stats['game_stats']['quiz']
        practice_stats = stats['practice_stats']

        # Update game-specific stats
        game_stats['attempts'] += 1
        game_stats['correct'] += self.correct
        game_stats['last_score'] = self.score
        if self.score > game_stats['highest_score']:
            game_stats['highest_score'] = self.score
        game_stats['accuracy'] = accuracy

        # Update practice stats

        practice_stats['total_time'] += duration // 60  # Convert to minutes

        # Update streaks
        today = datetime.now().date()
        last_practiced = practice_stats.get('last_practiced')

        if last_practiced:
            last_date = datetime.fromisoformat(last_practiced).date()
            if (today - last_date).days == 1:  # Consecutive day
                practice_stats['current_streak'] += 1
                practice_stats['longest_streak'] = max(
                    practice_stats['longest_streak'],
                    practice_stats['current_streak']
                )

                # Check daily goal
                if self.correct >= practice_stats['daily_goal']:
                    practice_stats['goal_streak'] += 1
            elif (today - last_date).days > 1:  # Broken streak
                practice_stats['current_streak'] = 1
                if self.correct >= practice_stats['daily_goal']:
                    practice_stats['goal_streak'] = 1
                else:
                    practice_stats['goal_streak'] = 0
        else:  # First practice
            practice_stats['current_streak'] = 1
            if self.correct >= practice_stats['daily_goal']:
                practice_stats['goal_streak'] = 1

        practice_stats['last_practiced'] = today.isoformat()

        # Update efficiency metrics (will be calculated periodically)
        stats['efficiency']['words_per_hour'] = practice_stats['words_correct'] / max(1,
                                                                                      practice_stats['total_time'] / 60)
        stats['efficiency']['games_per_week'] = game_stats['attempts'] / max(1, (
                datetime.now() - datetime.fromisoformat(self.user_data['meta']['created'])).days / 7)

        # Add to game history
        game_history = {
            'date': datetime.now().isoformat(),
            'score': self.score,
            'correct': self.correct,
            'total': self.total_rounds,
            'duration': duration,
            'words': list(self.words_encountered),
            'streak': self.max_streak,
            'accuracy': accuracy
        }
        self.user_data['game_history']['quiz'].append(game_history)

        # Update meta info
        self.user_data['meta']['last_updated'] = datetime.now().isoformat()

    def save_user_data_directly(self):
        """Fallback method to save user data if parent isn't available."""
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())
