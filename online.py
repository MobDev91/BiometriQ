import time
import cv2
import dlib
import numpy as np
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox,
    QHBoxLayout, QSpacerItem, QSizePolicy, QGroupBox, QFrame, QScrollArea,
    QDialog, QLineEdit, QTextEdit, QDialogButtonBox, QComboBox, QCheckBox,
    QGridLayout, QTabWidget, QProgressBar, QSlider, QSpinBox, QFileDialog
)
from PySide6.QtGui import QPixmap, QIcon, QImage, QMovie, QFont
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QFile, QTextStream
from Backend.functions import Functions 
from Backend.model_manager import model_manager

try:
    from Backend.face_recognition_core import FaceRecognitionCore
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

class FaceRecognitionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Face Recognition Database")
        self.setModal(True)
        self.resize(600, 500)
        
        self.setStyleSheet(self.get_dialog_style())
        
        layout = QVBoxLayout()
        
        tab_widget = QTabWidget()
        
        # Add Person Tab
        add_tab = QWidget()
        add_layout = QVBoxLayout(add_tab)
        
        add_layout.addWidget(QLabel("Add Person to Database:"))
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter person's name")
        add_layout.addWidget(self.name_input)
        
        self.capture_btn = QPushButton("📸 Capture from Camera")
        self.capture_btn.setObjectName("faceRecognitionButton")
        self.capture_btn.clicked.connect(self.capture_face)
        add_layout.addWidget(self.capture_btn)
        
        self.upload_btn = QPushButton("📁 Upload Image")
        self.upload_btn.setObjectName("faceRecognitionButton")
        self.upload_btn.clicked.connect(self.upload_image)
        add_layout.addWidget(self.upload_btn)
        
        tab_widget.addTab(add_tab, "Add Person")
        
        # Database Stats Tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        
        refresh_btn = QPushButton("🔄 Refresh Stats")
        refresh_btn.setObjectName("faceRecognitionButton")
        refresh_btn.clicked.connect(self.refresh_stats)
        stats_layout.addWidget(refresh_btn)
        
        tab_widget.addTab(stats_tab, "Database Stats")
        
        # Search/Delete Tab
        manage_tab = QWidget()
        manage_layout = QVBoxLayout(manage_tab)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search person by name")
        manage_layout.addWidget(self.search_input)
        
        search_btn = QPushButton("🔍 Search")
        search_btn.setObjectName("faceRecognitionButton")
        search_btn.clicked.connect(self.search_person)
        manage_layout.addWidget(search_btn)
        
        self.search_results = QTextEdit()
        self.search_results.setReadOnly(True)
        manage_layout.addWidget(self.search_results)
        
        delete_btn = QPushButton("🗑️ Delete Person")
        delete_btn.setObjectName("faceRecognitionButton")
        delete_btn.clicked.connect(self.delete_person)
        manage_layout.addWidget(delete_btn)
        
        tab_widget.addTab(manage_tab, "Manage")
        
        layout.addWidget(tab_widget)
        
        close_btn = QPushButton("Close")
        close_btn.setObjectName("faceRecognitionButton")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        
        self.refresh_stats()
    
    def get_dialog_style(self):
        return """
        QPushButton#faceRecognitionButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 #8b5cf6, stop:1 #7c3aed);
            border: none;
            border-radius: 10px;
            color: white;
            font-weight: 500;
            font-size: 14px;
            padding: 10px 20px;
            min-width: 140px;
        }
        
        QPushButton#faceRecognitionButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 #7c3aed, stop:1 #6d28d9);
        }
        
        QPushButton#faceRecognitionButton:disabled {
            background: #374151;
            color: #9ca3af;
        }
        
        QTabWidget::pane {
            border: 1px solid rgba(75, 85, 99, 0.3);
            border-radius: 8px;
            background: rgba(31, 41, 55, 0.4);
            padding: 10px;
        }
        
        QTabBar::tab {
            background: rgba(55, 65, 81, 0.6);
            border: 1px solid rgba(75, 85, 99, 0.3);
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            color: #d1d5db;
            font-weight: 500;
        }
        
        QTabBar::tab:selected {
            background: rgba(139, 92, 246, 0.6);
            border-color: #8b5cf6;
            color: #f1f5f9;
        }
        
        QTabBar::tab:hover:!selected {
            background: rgba(75, 85, 99, 0.8);
            color: #f1f5f9;
        }
        
        QLineEdit {
            background: rgba(55, 65, 81, 0.8);
            border: 1px solid rgba(107, 114, 128, 0.5);
            border-radius: 6px;
            padding: 10px 12px;
            color: #f1f5f9;
            font-size: 14px;
            selection-background-color: #8b5cf6;
        }
        
        QLineEdit:focus {
            border-color: #8b5cf6;
            background: rgba(55, 65, 81, 1.0);
        }
        
        QLineEdit::placeholder {
            color: #9ca3af;
        }
        
        QTextEdit {
            background: rgba(17, 24, 39, 0.8);
            border: 1px solid rgba(75, 85, 99, 0.3);
            border-radius: 6px;
            color: #f1f5f9;
            font-size: 14px;
            padding: 10px;
            selection-background-color: #8b5cf6;
        }
        
        QTextEdit:focus {
            border-color: #8b5cf6;
        }
        
        QDialog {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #1f2937, stop:1 #111827);
            color: #f1f5f9;
        }
        """
    
    def capture_face(self):
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "Warning", "Please enter a name first.")
            return
        
        self.accept()
        if hasattr(self.parent(), 'capture_face_for_database'):
            self.parent().capture_face_for_database(self.name_input.text().strip())
    
    def upload_image(self):
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "Warning", "Please enter a name first.")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            if hasattr(self.parent(), 'add_face_from_image'):
                self.parent().add_face_from_image(file_path, self.name_input.text().strip())
            self.accept()
    
    def refresh_stats(self):
        if hasattr(self.parent(), 'face_recognition') and self.parent().face_recognition:
            stats = self.parent().face_recognition.get_database_stats()
            
            stats_text = f"""📊 Database Statistics:

Total Faces: {stats['total_faces']}

Age Groups:
"""
            for age_group, count in stats['age_groups'].items():
                stats_text += f"  • {age_group}: {count}\n"
            
            stats_text += "\nGender Distribution:\n"
            for gender, count in stats['gender_distribution'].items():
                stats_text += f"  • {gender}: {count}\n"
            
            stats_text += "\nSources:\n"
            for source, count in stats['sources'].items():
                stats_text += f"  • {source}: {count}\n"
            
            self.stats_text.setText(stats_text)
        else:
            self.stats_text.setText("Face recognition not available")
    
    def search_person(self):
        name = self.search_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a name to search.")
            return
        
        if hasattr(self.parent(), 'face_recognition') and self.parent().face_recognition:
            matches = self.parent().face_recognition.search_person(name)
            
            if matches:
                result_text = f"Found {len(matches)} matches for '{name}':\n\n"
                for i, (idx, meta) in enumerate(matches):
                    result_text += f"Match {i+1}:\n"
                    result_text += f"  • Name: {meta.get('name')}\n"
                    result_text += f"  • Age: {meta.get('age')}\n"
                    result_text += f"  • Gender: {meta.get('gender', 'Unknown')}\n"
                    result_text += f"  • Source: {meta.get('source')}\n"
                    result_text += f"  • Added: {meta.get('timestamp', 'Unknown')}\n\n"
            else:
                result_text = f"No matches found for '{name}'"
            
            self.search_results.setText(result_text)
        else:
            self.search_results.setText("Face recognition not available")
    
    def delete_person(self):
        name = self.search_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a name to delete.")
            return
        
        reply = QMessageBox.question(
            self, "Confirm Delete", 
            f"Are you sure you want to delete all faces for '{name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if hasattr(self.parent(), 'face_recognition') and self.parent().face_recognition:
                deleted_count = self.parent().face_recognition.delete_person(name)
                if deleted_count > 0:
                    self.parent().face_recognition.save_database()
                    QMessageBox.information(self, "Success", f"Deleted {deleted_count} faces for '{name}'")
                    self.refresh_stats()
                    self.search_results.clear()
                else:
                    QMessageBox.information(self, "Not Found", f"No faces found for '{name}'")

class ModelLoaderThread(QThread):
    models_loaded = Signal()
    face_recognition_loaded = Signal()

    def __init__(self):
        super().__init__()
        self.models_loaded_flag = False
        self.face_recognition_loaded_flag = False
        
    def run(self):
        if not self.models_loaded_flag:
            model_manager.load_models()
            self.models_loaded_flag = True
        self.models_loaded.emit()
        
        if not self.face_recognition_loaded_flag and FACE_RECOGNITION_AVAILABLE:
            try:
                self.face_recognition_loaded_flag = True
                self.face_recognition_loaded.emit()
            except Exception as e:
                pass

class CameraThread(QThread):
    frame_ready = Signal(object)
    camera_error = Signal(str)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None
        self._mutex = QThread().mutex if hasattr(QThread(), 'mutex') else None
        
    def run(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                for i in range(1, 4):
                    self.cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if self.cap.isOpened():
                        break
                        
            if not self.cap.isOpened():
                self.camera_error.emit("Cannot access any camera")
                return
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.camera_error.emit("Camera connected but cannot read frames")
                return
            
            self.running = True
            failed_reads = 0
            
            while self.running:
                ret, frame = self.cap.read()
                if ret and frame is not None and self.running:
                    self.frame_ready.emit(frame)
                    failed_reads = 0
                else:
                    failed_reads += 1
                    if failed_reads > 10:
                        if self.running:  # Seul émettre erreur si toujours en cours
                            self.camera_error.emit("Too many failed frame reads")
                        break
                        
                if self.running:  # Vérifier avant de dormir
                    self.msleep(33)
                
        except Exception as e:
            if self.running:  # Seul émettre erreur si toujours en cours
                self.camera_error.emit(f"Camera thread error: {str(e)}")
        finally:
            self.cleanup_camera()
            
    def stop(self):
        self.running = False
        
    def cleanup_camera(self):
        if self.cap:
            try:
                self.cap.release()
                self.cap = None
            except Exception:
                pass

class ModernOnlineWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("OnlineWindow")
        
        self.setStyleSheet(self.get_modern_style())

        self.frame_counter = 0
        self.prediction_interval = 30
        self.predicted_text = ""
        self.camera_retry_count = 0
        self.max_camera_retries = 3
        self.current_frame = None
        
        # ✅ NOUVEAU: Variables pour stocker les coordonnées du visage pour l'overlay
        self.current_face_box = None
        
        self.face_recognition = None
        self.face_recognition_enabled = False

        self.setWindowTitle("BiometriQ - Live Camera Analysis")
        
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        icon_path = "assets/Icons/favicon-black.png"
        self.setWindowIcon(QIcon(icon_path))

        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(30, 30, 30, 30)

        header_layout = QHBoxLayout()
        
        home_button = QPushButton("🏠 Home")
        home_button.setObjectName("homeButton")
        home_button.clicked.connect(self.go_home)
        header_layout.addWidget(home_button)
        
        title_label = QLabel("Live Camera Analysis")
        title_label.setObjectName("pageTitle")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        self.face_recognition_btn = QPushButton("👤 Face Recognition")
        self.face_recognition_btn.setObjectName("faceRecognitionButton")
        self.face_recognition_btn.clicked.connect(self.open_face_recognition_dialog)
        self.face_recognition_btn.setEnabled(False)
        header_layout.addWidget(self.face_recognition_btn)
        
        offline_button = QPushButton("📁 Switch to Image Upload")
        offline_button.setObjectName("switchButton")
        offline_button.clicked.connect(self.switch_to_offline_mode)
        header_layout.addWidget(offline_button)
        
        main_layout.addLayout(header_layout)

        control_section = QFrame()
        control_section.setObjectName("controlSection")
        control_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        control_layout = QHBoxLayout(control_section)
        control_layout.setSpacing(20)
        
        self.start_button = QPushButton("🎥 Start Camera")
        self.start_button.setObjectName("startButton")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_video_capture)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("⏹️ Stop Camera")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_video_capture)
        control_layout.addWidget(self.stop_button)
        
        self.face_rec_toggle = QCheckBox("Enable Face Recognition")
        self.face_rec_toggle.setObjectName("faceRecToggle")
        self.face_rec_toggle.setEnabled(False)
        self.face_rec_toggle.toggled.connect(self.toggle_face_recognition)
        control_layout.addWidget(self.face_rec_toggle)
        
        control_layout.addStretch()
        
        threshold_label = QLabel("Recognition Threshold:")
        threshold_label.setObjectName("thresholdLabel")
        control_layout.addWidget(threshold_label)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setObjectName("thresholdSlider")
        self.threshold_slider.setRange(30, 90)
        self.threshold_slider.setValue(60)
        self.threshold_slider.setEnabled(False)
        control_layout.addWidget(self.threshold_slider)
        
        self.threshold_value_label = QLabel("60%")
        self.threshold_value_label.setObjectName("thresholdValueLabel")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_value_label.setText(f"{v}%")
        )
        control_layout.addWidget(self.threshold_value_label)
        
        self.status_indicator = QLabel("●")
        self.status_indicator.setObjectName("statusIndicator")
        control_layout.addWidget(self.status_indicator)
        
        self.status_label = QLabel("Loading AI models...")
        self.status_label.setObjectName("statusLabel")
        control_layout.addWidget(self.status_label)
        
        main_layout.addWidget(control_section)

        content_scroll = QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content_scroll.setObjectName("contentScroll")
        content_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setSpacing(30)
        content_layout.setContentsMargins(20, 20, 20, 20)

        video_container = QFrame()
        video_container.setObjectName("videoContainer")
        video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout = QVBoxLayout(video_container)
        video_layout.setSpacing(15)
        
        video_label = QLabel("Camera Feed")
        video_label.setObjectName("sectionLabel")
        video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(video_label)
        
        self.video_label = QLabel()
        self.video_label.setObjectName("videoDisplay")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(600, 450)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setText("Camera feed will appear here\n\nClick 'Start Camera' to begin")
        video_layout.addWidget(self.video_label)
        
        content_layout.addWidget(video_container, 2)

        results_container = QFrame()
        results_container.setObjectName("resultsContainer")
        results_container.setMinimumWidth(350)
        results_container.setMaximumWidth(500)
        results_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        results_layout = QVBoxLayout(results_container)
        results_layout.setSpacing(20)
        
        results_label = QLabel("Live Analysis Results")
        results_label.setObjectName("sectionLabel")
        results_layout.addWidget(results_label)

        instructions_card = QFrame()
        instructions_card.setObjectName("instructionsCard")
        instructions_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        instructions_layout = QVBoxLayout(instructions_card)
        instructions_layout.setSpacing(10)
        
        instructions_title = QLabel("📋 Instructions")
        instructions_title.setObjectName("cardTitle")
        instructions_layout.addWidget(instructions_title)
        
        instructions_text = QLabel("• Ensure good lighting\n• Look directly at camera\n• Keep face visible\n• Shape & Emotion use dedicated models\n• Gender & Age require Face Recognition\n• Face Recognition enabled by default")
        instructions_text.setObjectName("instructionsText")
        instructions_text.setWordWrap(True)
        instructions_layout.addWidget(instructions_text)
        
        results_layout.addWidget(instructions_card)

        self.results_card = QFrame()
        self.results_card.setObjectName("resultsCard")
        self.results_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_card_layout = QVBoxLayout(self.results_card)
        results_card_layout.setSpacing(15)
        
        results_title = QLabel("🤖 AI Analysis")
        results_title.setObjectName("cardTitle")
        results_card_layout.addWidget(results_title)

        self.shape_display = QLabel("🔍 Face Shape: Analyzing...")
        self.shape_display.setObjectName("resultDisplay")
        self.shape_display.setWordWrap(True)
        results_card_layout.addWidget(self.shape_display)

        self.gender_display = QLabel("👤 Gender: Analyzing...")
        self.gender_display.setObjectName("resultDisplay")
        self.gender_display.setWordWrap(True)
        results_card_layout.addWidget(self.gender_display)

        self.emotion_display = QLabel("😊 Emotion: Analyzing...")
        self.emotion_display.setObjectName("resultDisplay")
        self.emotion_display.setWordWrap(True)
        results_card_layout.addWidget(self.emotion_display)

        self.age_display = QLabel("🎂 Age: Analyzing...")
        self.age_display.setObjectName("resultDisplay")
        self.age_display.setWordWrap(True)
        results_card_layout.addWidget(self.age_display)

        self.face_recognition_display = QLabel("👥 Identity: Face Recognition Disabled")
        self.face_recognition_display.setObjectName("faceRecognitionDisplay")
        self.face_recognition_display.setWordWrap(True)
        results_card_layout.addWidget(self.face_recognition_display)

        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setObjectName("confidenceLabel")
        self.confidence_label.setWordWrap(True)
        results_card_layout.addWidget(self.confidence_label)

        self.results_card.hide()
        results_layout.addWidget(self.results_card)
        
        content_layout.addWidget(results_container, 1)
        
        content_scroll.setWidget(content_widget)
        main_layout.addWidget(content_scroll)
        self.setLayout(main_layout)

        self.model_loader_thread = ModelLoaderThread()
        self.model_loader_thread.models_loaded.connect(self.on_models_loaded)
        self.model_loader_thread.face_recognition_loaded.connect(self.on_face_recognition_loaded)
        self.model_loader_thread.start()

        self.camera_thread = None
        self.face_detector = dlib.get_frontal_face_detector()
        
        try:
            self.landmark_predictor = dlib.shape_predictor("Utilities/Face-Detection/shape_predictor_68_face_landmarks.dat")
        except Exception:
            self.landmark_predictor = None

        if FACE_RECOGNITION_AVAILABLE:
            try:
                self.face_recognition = FaceRecognitionCore("face_database.h5")
                self.face_recognition_btn.setEnabled(True)
                self.face_rec_toggle.setEnabled(True)
                self.threshold_slider.setEnabled(True)
                
                # ✅ NOUVEAU: Activer par défaut car requis pour genre et âge
                self.face_rec_toggle.setChecked(True)
                self.face_recognition_enabled = True
                
            except Exception as e:
                self.face_recognition = None

    # ✅ NOUVELLE FONCTION: Dessiner les overlays sur le frame
    def draw_face_overlays(self, frame):
        """Dessine un carré vert autour du visage détecté"""
        if self.current_face_box is None:
            return frame
        
        # Copier le frame pour ne pas modifier l'original
        overlay_frame = frame.copy()
        
        # Récupérer les coordonnées de la boîte du visage
        x, y, w, h = self.current_face_box
        
        # 🟢 DESSINER UNIQUEMENT LA BOÎTE VERTE autour du visage
        cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return overlay_frame

    def get_modern_style(self):
        return """
        QWidget#OnlineWindow {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #0f1419, stop:1 #1a202c);
            color: #f1f5f9;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        
        QPushButton#homeButton, QPushButton#switchButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #374151, stop:1 #4b5563);
            border: 1px solid #6b7280;
            border-radius: 8px;
            color: #f9fafb;
            font-weight: 500;
            font-size: 14px;
            padding: 10px 20px;
            min-width: 120px;
        }
        
        QPushButton#homeButton:hover, QPushButton#switchButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #4b5563, stop:1 #6b7280);
            border-color: #9ca3af;
        }
        
        QPushButton#faceRecognitionButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #8b5cf6, stop:1 #7c3aed);
            border: none;
            border-radius: 10px;
            color: white;
            font-weight: 500;
            font-size: 14px;
            padding: 10px 20px;
            min-width: 140px;
        }
        
        QPushButton#faceRecognitionButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #7c3aed, stop:1 #6d28d9);
        }
        
        QPushButton#faceRecognitionButton:disabled {
            background: #374151;
            color: #9ca3af;
        }
        
        QPushButton#startButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #10b981, stop:1 #059669);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            font-size: 14px;
            padding: 12px 24px;
            min-width: 130px;
        }
        
        QPushButton#startButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #059669, stop:1 #047857);
        }
        
        QPushButton#startButton:disabled {
            background: #374151;
            color: #9ca3af;
        }
        
        QPushButton#stopButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #ef4444, stop:1 #dc2626);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            font-size: 14px;
            padding: 12px 24px;
            min-width: 130px;
        }
        
        QPushButton#stopButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #dc2626, stop:1 #b91c1c);
        }
        
        QPushButton#stopButton:disabled {
            background: #374151;
            color: #9ca3af;
        }
        
        QLabel#pageTitle {
            color: #f1f5f9;
            font-size: 28px;
            font-weight: 700;
            margin: 0 20px;
        }
        
        QFrame#controlSection {
            background: rgba(31, 41, 55, 0.6);
            border: 1px solid rgba(75, 85, 99, 0.3);
            border-radius: 12px;
            padding: 15px;
            margin: 10px 0;
        }
        
        QCheckBox#faceRecToggle {
            color: #f1f5f9;
            font-weight: 500;
            spacing: 10px;
        }
        
        QCheckBox#faceRecToggle::indicator {
            width: 18px;
            height: 18px;
            border-radius: 9px;
            border: 2px solid #64748b;
            background: transparent;
        }
        
        QCheckBox#faceRecToggle::indicator:checked {
            background: #10b981;
            border-color: #10b981;
        }
        
        QCheckBox#faceRecToggle::indicator:hover {
            border-color: #10b981;
        }
        
        QLabel#thresholdLabel, QLabel#thresholdValueLabel {
            color: #cbd5e1;
            font-size: 14px;
            font-weight: 500;
        }
        
        QSlider#thresholdSlider {
            min-height: 20px;
        }
        
        QSlider#thresholdSlider::groove:horizontal {
            border: 1px solid #64748b;
            height: 6px;
            background: #374151;
            border-radius: 3px;
        }
        
        QSlider#thresholdSlider::handle:horizontal {
            background: #10b981;
            border: 1px solid #059669;
            width: 16px;
            height: 16px;
            border-radius: 8px;
            margin: -6px 0;
        }
        
        QSlider#thresholdSlider::handle:horizontal:hover {
            background: #059669;
        }
        
        QSlider#thresholdSlider::handle:horizontal:pressed {
            background: #047857;
        }
        
        QLabel#statusIndicator {
            color: #f59e0b;
            font-size: 16px;
            font-weight: bold;
            margin-right: 8px;
        }
        
        QLabel#statusLabel {
            color: #d1d5db;
            font-size: 14px;
            font-weight: 500;
        }
        
        QScrollArea#contentScroll {
            border: none;
            background: transparent;
        }
        
        QScrollArea#contentScroll QScrollBar:vertical {
            background: rgba(55, 65, 81, 0.3);
            width: 12px;
            border-radius: 6px;
            margin: 0;
        }
        
        QScrollArea#contentScroll QScrollBar::handle:vertical {
            background: rgba(156, 163, 175, 0.5);
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollArea#contentScroll QScrollBar::handle:vertical:hover {
            background: rgba(156, 163, 175, 0.7);
        }
        
        QFrame#videoContainer, QFrame#resultsContainer {
            background: rgba(31, 41, 55, 0.4);
            border: 1px solid rgba(75, 85, 99, 0.2);
            border-radius: 12px;
            padding: 20px;
        }
        
        QLabel#sectionLabel {
            color: #e5e7eb;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            text-align: center;
        }
        
        QLabel#videoDisplay {
            background: #111827;
            border: 2px dashed #374151;
            border-radius: 8px;
            color: #9ca3af;
            font-size: 16px;
            font-weight: 500;
            padding: 40px;
            text-align: center;
        }
        
        QFrame#instructionsCard, QFrame#resultsCard {
            background: rgba(55, 65, 81, 0.3);
            border: 1px solid rgba(107, 114, 128, 0.2);
            border-radius: 8px;
            padding: 15px;
            margin: 8px 0;
        }
        
        QLabel#cardTitle {
            color: #f3f4f6;
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        QLabel#instructionsText {
            color: #d1d5db;
            font-size: 14px;
            line-height: 1.4;
        }
        
        QLabel#resultDisplay {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 6px;
            padding: 12px;
            color: #6ee7b7;
            font-weight: 500;
            margin: 6px 0;
            font-size: 14px;
        }
        
        QLabel#faceRecognitionDisplay {
            background: rgba(139, 92, 246, 0.1);
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 8px;
            padding: 15px;
            color: #a78bfa;
            font-weight: 500;
            margin: 8px 0;
            font-size: 15px;
        }
        
        QLabel#confidenceLabel {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 6px;
            padding: 10px;
            color: #93c5fd;
            font-weight: 500;
            font-size: 14px;
        }
        """

    def go_home(self):
        """Retourner à l'écran d'accueil de manière sécurisée"""
        # Arrêter la caméra d'abord si elle est active
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_video_capture()
        
        from main import ModernWelcomeWindow
        self.welcome_window = ModernWelcomeWindow()
        self.welcome_window.show()
        self.close()
    
    def switch_to_offline_mode(self):
        """Changer vers le mode offline de manière sécurisée"""
        # Arrêter la caméra d'abord si elle est active
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_video_capture()
        
        from offline import ModernOfflineWindow
        self.offline_window = ModernOfflineWindow()
        self.offline_window.show()
        self.close()
    
    def open_face_recognition_dialog(self):
        dialog = FaceRecognitionDialog(self)
        dialog.exec()
    
    def toggle_face_recognition(self, enabled):
        self.face_recognition_enabled = enabled
        if enabled:
            self.face_recognition_display.setText("👥 Identity: Face Recognition Active")
        else:
            self.face_recognition_display.setText("👥 Identity: Face Recognition Disabled")
    
    def start_video_capture(self):
        if self.camera_thread is None or not self.camera_thread.isRunning():
            self.camera_thread = CameraThread()
            self.camera_thread.frame_ready.connect(self.update_frame)
            self.camera_thread.camera_error.connect(self.handle_camera_error)
            self.camera_thread.start()
            
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Camera active")
            self.status_indicator.setStyleSheet("color: #10b981;")
            self.results_card.show()
    
    def stop_video_capture(self):
        """Arrêter la capture vidéo de manière sécurisée"""
        if self.camera_thread and self.camera_thread.isRunning():
            # Arrêter le thread proprement
            self.camera_thread.stop()
            
            # Attendre avec timeout pour éviter les blocages
            if not self.camera_thread.wait(3000):  # Timeout de 3 secondes
                print("Warning: Camera thread did not stop within timeout, forcing termination")
                self.camera_thread.terminate()
                self.camera_thread.wait(1000)  # Attendre 1 seconde de plus
            
            # Nettoyer les références
            self.camera_thread = None
            
        # Mettre à jour l'interface
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Camera stopped")
        self.status_indicator.setStyleSheet("color: #f59e0b;")
        
        # Nettoyer l'affichage vidéo
        self.video_label.clear()
        self.video_label.setText("Camera feed will appear here\n\nClick 'Start Camera' to begin")
        self.results_card.hide()
    
    def update_frame(self, frame):
        try:
            self.current_frame = frame.copy()
            
            # ✅ NOUVEAU: Appliquer les overlays avant d'afficher le frame
            frame_with_overlays = self.draw_face_overlays(frame)
            
            rgb_frame = cv2.cvtColor(frame_with_overlays, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            label_size = self.video_label.size()
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            self.video_label.setPixmap(scaled_pixmap)
            
            self.frame_counter += 1
            if self.frame_counter % self.prediction_interval == 0:
                self.process_frame_analysis(frame)
                
        except Exception as e:
            pass
    
    def process_frame_analysis(self, frame):
        try:
            # ✅ RÉINITIALISER la variable d'overlay au début
            self.current_face_box = None
            
            # Détecter les visages avec dlib pour les overlays
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)
            
            if len(faces) > 0:
                # Prendre le premier visage détecté
                face = faces[0]
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                
                # ✅ SAUVEGARDER la boîte du visage pour les overlays
                self.current_face_box = (x, y, w, h)
                
                # Toujours utiliser face_recognition pour genre et âge maintenant
                if self.face_recognition:
                    face_results = self.face_recognition.analyze_faces(frame)
                    
                    if face_results:
                        face_data = face_results[0]
                        
                        age_exact = face_data.get('age_exact', 'Unknown')
                        age_range = face_data.get('age_range', 'Unknown')
                        gender = face_data.get('gender', 'Unknown')
                        match = face_data.get('match')
                        similarity = face_data.get('similarity', 0.0)
                        
                        if age_exact != 'Unknown' and age_exact is not None:
                            self.age_display.setText(f"🎂 Age: {age_exact:.0f} years ({age_range})")
                        else:
                            self.age_display.setText("🎂 Age: Unknown")
                        
                        self.gender_display.setText(f"👤 Gender: {gender}")
                        
                        # Gestion de la reconnaissance d'identité seulement si activée
                        if self.face_recognition_enabled:
                            if match:
                                threshold = self.threshold_slider.value() / 100.0
                                if similarity > threshold:
                                    self.face_recognition_display.setText(f"👥 Identity: {match['name']} ({similarity:.2f})")
                                    self.confidence_label.setText(f"Match Confidence: {similarity:.3f}")
                                else:
                                    self.face_recognition_display.setText("👥 Identity: Unknown (Below threshold)")
                                    self.confidence_label.setText(f"Best Match: {similarity:.3f} (Below {threshold:.2f})")
                            else:
                                self.face_recognition_display.setText("👥 Identity: Unknown")
                                self.confidence_label.setText(f"Best Match: {similarity:.3f}")
                        else:
                            self.face_recognition_display.setText("👥 Identity: Face Recognition Disabled")
                            self.confidence_label.setText("Confidence: --")
                    else:
                        self.reset_face_recognition_results()
                else:
                    # Si pas de face_recognition, informer l'utilisateur
                    self.gender_display.setText("👤 Gender: Face Recognition Required")
                    self.age_display.setText("🎂 Age: Face Recognition Required")
                    self.face_recognition_display.setText("👥 Identity: Face Recognition Required")
                    self.confidence_label.setText("Confidence: --")
                
                # Analyser avec les modèles traditionnels
                face_img = frame[y:y+h, x:x+w]
                
                if face_img.size > 0:
                    self.update_basic_analysis_results(frame)
                else:
                    self.reset_basic_analysis_results()
            else:
                self.reset_basic_analysis_results()
                if not self.face_recognition_enabled:
                    self.reset_face_recognition_results()
                
        except Exception as e:
            pass
    
    def update_basic_analysis_results(self, frame):
        # Garder seulement shape et emotion - genre et âge gérés par InsightFace
        if model_manager.shape_model:
            predicted_class, predictions = Functions.predict_shape("online", frame, model_manager.shape_model)
            self.shape_display.setText(f"🔍 Face Shape: {predicted_class}")
        
        if model_manager.emotion_model:
            predicted_class, predictions = Functions.predict_emotion("online", frame, model_manager.emotion_model)
            self.emotion_display.setText(f"😊 Emotion: {predicted_class}")
        
        # ❌ SUPPRIMÉ: gender et age avec les anciens modèles - maintenant gérés par InsightFace
    
    def reset_basic_analysis_results(self):
        self.shape_display.setText("🔍 Face Shape: No face detected")
        self.emotion_display.setText("😊 Emotion: No face detected")
        
        # ✅ RÉINITIALISER la variable d'overlay
        self.current_face_box = None
        
        # Genre et âge maintenant toujours gérés par face_recognition
        if not self.face_recognition:
            self.gender_display.setText("👤 Gender: Face Recognition Required")
            self.age_display.setText("🎂 Age: Face Recognition Required")
    
    def reset_face_recognition_results(self):
        if not self.face_recognition_enabled:
            self.face_recognition_display.setText("👥 Identity: Face Recognition Disabled")
            self.confidence_label.setText("Confidence: --")
        else:
            self.face_recognition_display.setText("👥 Identity: No face detected")
            self.confidence_label.setText("Confidence: --")
    
    def handle_camera_error(self, error_message):
        QMessageBox.critical(self, "Camera Error", f"Camera error: {error_message}")
        self.stop_video_capture()
    
    def on_models_loaded(self):
        self.start_button.setEnabled(True)
        self.status_label.setText("Models loaded - Ready to start")
        self.status_indicator.setStyleSheet("color: #10b981;")
    
    def on_face_recognition_loaded(self):
        pass
    
    def capture_face_for_database(self, name):
        if self.current_frame is not None and self.face_recognition:
            try:
                face_results = self.face_recognition.analyze_faces(self.current_frame)
                
                if face_results:
                    face_data = face_results[0]
                    embedding = face_data['embedding']
                    age_range = face_data.get('age_range', 'Unknown')
                    gender = face_data.get('gender', 'Unknown')
                    confidence = face_data.get('confidence', 0.0)
                    
                    self.face_recognition.add_face(
                        embedding=embedding,
                        name=name,
                        age=age_range,
                        confidence=confidence,
                        source="camera",
                        gender=gender
                    )
                    
                    self.face_recognition.save_database()
                    QMessageBox.information(self, "Success", f"Face captured and added for {name}")
                else:
                    QMessageBox.warning(self, "No Face", "No face detected in current camera frame")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to capture face: {str(e)}")
        else:
            QMessageBox.warning(self, "No Frame", "No camera frame available for capture")
    
    def add_face_from_image(self, image_path, name):
        if self.face_recognition:
            try:
                image = cv2.imread(image_path)
                if image is None:
                    QMessageBox.warning(self, "Error", "Could not load image")
                    return
                
                face_results = self.face_recognition.analyze_faces(image)
                
                if face_results:
                    face_data = face_results[0]
                    embedding = face_data['embedding']
                    age_range = face_data.get('age_range', 'Unknown')
                    gender = face_data.get('gender', 'Unknown')
                    confidence = face_data.get('confidence', 0.0)
                    
                    self.face_recognition.add_face(
                        embedding=embedding,
                        name=name,
                        age=age_range,
                        confidence=confidence,
                        source="upload",
                        gender=gender
                    )
                    
                    self.face_recognition.save_database()
                    QMessageBox.information(self, "Success", f"Face added from image for {name}")
                else:
                    QMessageBox.warning(self, "No Face", "No face detected in uploaded image")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to add face from image: {str(e)}")
    
    def closeEvent(self, event):
        """Gérer la fermeture de l'application de manière sécurisée"""
        try:
            # Arrêter la caméra si elle est active
            if self.camera_thread and self.camera_thread.isRunning():
                self.camera_thread.stop()
                if not self.camera_thread.wait(2000):  # Timeout de 2 secondes
                    self.camera_thread.terminate()
                    self.camera_thread.wait(500)
                self.camera_thread = None
            
            # Nettoyer les ressources OpenCV
            cv2.destroyAllWindows()
            
            # Accepter l'événement de fermeture
            event.accept()
            
        except Exception as e:
            print(f"Error during close: {e}")
            event.accept()  # Forcer la fermeture même en cas d'erreur