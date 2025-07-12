import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QHBoxLayout, QGroupBox, QFormLayout, QMessageBox, QFrame, QScrollArea, 
    QSizePolicy, QDialog, QLineEdit, QTextEdit, QTabWidget, QCheckBox
)
from PySide6.QtGui import QPixmap, QIcon, QFont
from PySide6.QtCore import Qt, QThread, Signal, QFile, QTextStream
from Backend.model_manager import model_manager
from Backend.functions import Functions 

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

    def __init__(self):
        super().__init__()
        self.models_loaded_flag = False

    def run(self):
        if not self.models_loaded_flag:
            model_manager.load_models()
            self.models_loaded_flag = True
        self.models_loaded.emit()

class ModernOfflineWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("OfflineWindow")
        
        self.setStyleSheet(self.get_modern_style())

        self.setWindowTitle("BiometriQ - Image Analysis")
        
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        icon_path = "assets/Icons/favicon-black.png"
        self.setWindowIcon(QIcon(icon_path))

        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # Initialize face recognition
        self.face_recognition = None
        self.face_recognition_enabled = False
        self.current_image_path = None
        
        # Face recognition is now required for gender and age detection

        # Header
        header_layout = QHBoxLayout()
        
        home_button = QPushButton("🏠 Home")
        home_button.setObjectName("homeButton")
        home_button.clicked.connect(self.go_home)
        header_layout.addWidget(home_button)
        
        title_label = QLabel("Image Analysis")
        title_label.setObjectName("pageTitle")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Face Recognition button
        self.face_recognition_btn = QPushButton("👤 Face Recognition")
        self.face_recognition_btn.setObjectName("faceRecognitionButton")
        self.face_recognition_btn.clicked.connect(self.open_face_recognition_dialog)
        self.face_recognition_btn.setEnabled(False)
        header_layout.addWidget(self.face_recognition_btn)
        
        online_button = QPushButton("📷 Switch to Live Camera")
        online_button.setObjectName("switchButton")
        online_button.clicked.connect(self.switch_to_online_mode)
        header_layout.addWidget(online_button)
        
        main_layout.addLayout(header_layout)

        # Upload section
        upload_section = QFrame()
        upload_section.setObjectName("uploadSection")
        upload_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        upload_layout = QVBoxLayout(upload_section)
        upload_layout.setSpacing(10)
        
        upload_label = QLabel("Upload an image for analysis")
        upload_label.setObjectName("sectionLabel")
        upload_layout.addWidget(upload_label, alignment=Qt.AlignCenter)
        
        upload_button = QPushButton("📁 Choose Image")
        upload_button.setObjectName("uploadButton")
        upload_button.clicked.connect(self.upload_image)
        upload_layout.addWidget(upload_button, alignment=Qt.AlignCenter)
        
        main_layout.addWidget(upload_section)

        # Content area with scroll
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

        # Left side - Images
        images_container = QFrame()
        images_container.setObjectName("imagesContainer")
        images_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        images_layout = QVBoxLayout(images_container)
        images_layout.setSpacing(20)
        
        # Original image section
        original_section = QFrame()
        original_section.setObjectName("imageSection")
        original_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        original_layout = QVBoxLayout(original_section)
        original_layout.setSpacing(10)
        
        original_label = QLabel("Original Image")
        original_label.setObjectName("imageLabel")
        original_label.setAlignment(Qt.AlignCenter)
        original_layout.addWidget(original_label)
        
        self.uploaded_image_label = QLabel()
        self.uploaded_image_label.setObjectName("imageDisplay")
        self.uploaded_image_label.setAlignment(Qt.AlignCenter)
        self.uploaded_image_label.setMinimumSize(400, 300)
        self.uploaded_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.uploaded_image_label.setText("No image uploaded\n\nClick 'Choose Image' to start")
        original_layout.addWidget(self.uploaded_image_label)
        
        images_layout.addWidget(original_section)

        # Processed image section
        processed_section = QFrame()
        processed_section.setObjectName("imageSection")
        processed_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        processed_layout = QVBoxLayout(processed_section)
        processed_layout.setSpacing(10)
        
        processed_label = QLabel("Processed Face")
        processed_label.setObjectName("imageLabel")
        processed_label.setAlignment(Qt.AlignCenter)
        processed_layout.addWidget(processed_label)
        
        self.preprocessed_image_label = QLabel()
        self.preprocessed_image_label.setObjectName("imageDisplay")
        self.preprocessed_image_label.setAlignment(Qt.AlignCenter)
        self.preprocessed_image_label.setMinimumSize(400, 200)
        self.preprocessed_image_label.setMaximumHeight(300)
        self.preprocessed_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.preprocessed_image_label.setText("Detected face will appear here")
        processed_layout.addWidget(self.preprocessed_image_label)
        
        images_layout.addWidget(processed_section)

        content_layout.addWidget(images_container, 2)

        # Right side - Controls and Results
        controls_container = QFrame()
        controls_container.setObjectName("controlsContainer")
        controls_container.setMinimumWidth(350)
        controls_container.setMaximumWidth(500)
        controls_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setSpacing(20)
        
        # Face Recognition Controls
        face_rec_section = QFrame()
        face_rec_section.setObjectName("faceRecSection")
        face_rec_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        face_rec_layout = QVBoxLayout(face_rec_section)
        face_rec_layout.setSpacing(10)
        
        face_rec_label = QLabel("Face Recognition")
        face_rec_label.setObjectName("sectionLabel")
        face_rec_layout.addWidget(face_rec_label)
        
        self.face_rec_toggle = QCheckBox("Enable Face Recognition Analysis")
        self.face_rec_toggle.setObjectName("faceRecToggle")
        self.face_rec_toggle.setEnabled(False)
        self.face_rec_toggle.toggled.connect(self.toggle_face_recognition)
        face_rec_layout.addWidget(self.face_rec_toggle)
        
        face_rec_section.hide()
        controls_layout.addWidget(face_rec_section)
        
        # Analysis buttons
        self.analysis_group = QGroupBox("Analysis Tools")
        self.analysis_group.setObjectName("analysisGroup")
        self.analysis_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        analysis_layout = QVBoxLayout(self.analysis_group)
        analysis_layout.setSpacing(12)
        
        self.predict_shape_button = QPushButton("🔍 Analyze Face Shape")
        self.predict_gender_button = QPushButton("👤 Detect Gender")
        self.predict_emotion_button = QPushButton("😊 Recognize Emotion")
        self.predict_age_button = QPushButton("🎂 Predict Age")
        self.analyze_all_button = QPushButton("🚀 Analyze All")
        
        for button in [self.predict_shape_button, self.predict_gender_button, 
                      self.predict_emotion_button, self.predict_age_button, self.analyze_all_button]:
            button.setObjectName("analysisButton")
            button.setEnabled(False)
            button.hide()
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            analysis_layout.addWidget(button)

        # Connect buttons
        self.predict_shape_button.clicked.connect(self.predict_shape)
        self.predict_gender_button.clicked.connect(self.predict_gender)
        self.predict_emotion_button.clicked.connect(self.predict_emotion)
        self.predict_age_button.clicked.connect(self.predict_age)
        self.analyze_all_button.clicked.connect(self.analyze_all)
        
        self.analysis_group.hide()
        controls_layout.addWidget(self.analysis_group)

        # Results section
        self.results_group = QGroupBox("Analysis Results")
        self.results_group.setObjectName("resultsGroup")
        self.results_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.results_layout = QVBoxLayout(self.results_group)
        self.results_layout.setSpacing(12)

        # Status label
        self.status_label = QLabel("Upload an image to begin analysis")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        self.results_layout.addWidget(self.status_label)

        # Result labels
        self.shape_result = QLabel()
        self.gender_result = QLabel()
        self.emotion_result = QLabel()
        self.age_result = QLabel()
        self.face_recognition_result = QLabel()
        
        for label in [self.shape_result, self.gender_result, self.emotion_result, 
                     self.age_result, self.face_recognition_result]:
            label.setObjectName("resultLabel")
            label.setWordWrap(True)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            label.hide()
            self.results_layout.addWidget(label)

        controls_layout.addWidget(self.results_group)

        content_layout.addWidget(controls_container, 1)
        
        content_scroll.setWidget(content_widget)
        main_layout.addWidget(content_scroll)

        # Initialize model loading
        self.model_loader_thread = ModelLoaderThread()
        self.model_loader_thread.models_loaded.connect(self.on_models_loaded)

        self.file_path = None
        self.setLayout(main_layout)

        # Initialize face recognition
        if FACE_RECOGNITION_AVAILABLE:
            try:
                self.face_recognition = FaceRecognitionCore("face_database.h5")
                self.face_recognition_btn.setEnabled(True)
                self.face_rec_toggle.setEnabled(True)
                
                # ✅ NOUVEAU: Activer par défaut car requis pour genre et âge
                self.face_rec_toggle.setChecked(True)
                self.face_recognition_enabled = True
                
            except Exception as e:
                self.face_recognition = None

    def get_modern_style(self):
        return """
        QWidget#OfflineWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                       stop:0 #0f172a, stop:1 #1e293b);
            color: #f8fafc;
        }
        
        QScrollArea#contentScroll {
            background: transparent;
            border: none;
        }
        
        QScrollArea#contentScroll QScrollBar:vertical, QScrollArea#contentScroll QScrollBar:horizontal {
            background: rgba(30, 41, 59, 0.5);
            width: 12px;
            height: 12px;
            border-radius: 6px;
        }
        
        QScrollArea#contentScroll QScrollBar::handle:vertical, QScrollArea#contentScroll QScrollBar::handle:horizontal {
            background: rgba(148, 163, 184, 0.5);
            border-radius: 6px;
            min-height: 20px;
            min-width: 20px;
        }
        
        QScrollArea#contentScroll QScrollBar::handle:vertical:hover, QScrollArea#contentScroll QScrollBar::handle:horizontal:hover {
            background: rgba(148, 163, 184, 0.7);
        }
        
        QLabel#pageTitle {
            font-size: 32px;
            font-weight: 700;
            color: #f8fafc;
            margin: 10px 0;
        }
        
        QLabel#sectionLabel {
            font-size: 16px;
            color: #cbd5e1;
            font-weight: 500;
            margin-bottom: 10px;
        }
        
        QLabel#imageLabel {
            font-size: 16px;
            color: #94a3b8;
            font-weight: 600;
            margin: 5px 0;
            padding: 8px;
        }
        
        QFrame#uploadSection {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 15px;
            padding: 20px;
            max-height: 100px;
        }
        
        QFrame#imagesContainer, QFrame#controlsContainer {
            background: rgba(30, 41, 59, 0.3);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-radius: 15px;
            padding: 20px;
        }
        
        QFrame#imageSection {
            background: rgba(51, 65, 85, 0.3);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 12px;
            padding: 15px;
        }
        
        QFrame#faceRecSection {
            background: rgba(139, 92, 246, 0.1);
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 12px;
            padding: 15px;
        }
        
        QPushButton#uploadButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 #10b981, stop:1 #059669);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            font-size: 16px;
            padding: 12px 30px;
            min-width: 200px;
            max-width: 250px;
            min-height: 40px;
        }
        
        QPushButton#uploadButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 #059669, stop:1 #047857);
        }
        
        QPushButton#homeButton {
            background: rgba(147, 197, 253, 0.2);
            border: 2px solid #93c5fd;
            border-radius: 10px;
            color: #93c5fd;
            font-weight: 500;
            font-size: 14px;
            padding: 8px 16px;
            min-width: 80px;
        }
        
        QPushButton#homeButton:hover {
            background: rgba(147, 197, 253, 0.3);
            border-color: #dbeafe;
        }
        
        QPushButton#switchButton {
            background: rgba(55, 65, 81, 0.8);
            border: 2px solid #4b5563;
            border-radius: 10px;
            color: #f1f5f9;
            font-weight: 500;
            font-size: 14px;
            padding: 8px 16px;
            min-width: 180px;
        }
        
        QPushButton#switchButton:hover {
            background: rgba(75, 85, 99, 0.9);
            border-color: #6b7280;
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
        
        QLabel#imageDisplay {
            background: rgba(51, 65, 85, 0.4);
            border: 2px dashed #475569;
            border-radius: 12px;
            color: #94a3b8;
            font-size: 14px;
            padding: 20px;
        }
        
        QGroupBox#analysisGroup, QGroupBox#resultsGroup {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid #334155;
            border-radius: 15px;
            margin: 10px 0;
            padding-top: 20px;
            font-weight: 600;
            font-size: 16px;
            color: #f1f5f9;
        }
        
        QGroupBox#analysisGroup::title, QGroupBox#resultsGroup::title {
            subcontrol-origin: margin;
            left: 20px;
            padding: 0 10px;
            color: #f1f5f9;
        }
        
        QPushButton#analysisButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 #3b82f6, stop:1 #2563eb);
            border: none;
            border-radius: 10px;
            color: white;
            font-weight: 500;
            font-size: 15px;
            padding: 12px 20px;
            margin: 4px 0;
            text-align: left;
            min-height: 16px;
        }
        
        QPushButton#analysisButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 #2563eb, stop:1 #1d4ed8);
        }
        
        QPushButton#analysisButton:disabled {
            background: #374151;
            color: #9ca3af;
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
            background: #8b5cf6;
            border-color: #8b5cf6;
        }
        
        QCheckBox#faceRecToggle::indicator:hover {
            border-color: #8b5cf6;
        }
        
        QLabel#statusLabel {
            color: #10b981;
            font-weight: 500;
            font-size: 14px;
            padding: 12px;
            background: rgba(16, 185, 129, 0.1);
            border-radius: 8px;
            border: 1px solid rgba(16, 185, 129, 0.3);
            min-height: 20px;
        }
        
        QLabel#resultLabel {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 8px;
            padding: 12px;
            color: #3b82f6;
            font-weight: 500;
            margin: 4px 0;
            font-size: 14px;
            min-height: 20px;
        }
        """

    def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.JPG)")
        if file_path:
            self.file_path = file_path
            self.current_image_path = file_path
            
            pixmap = QPixmap(file_path)
            label_size = self.uploaded_image_label.size()
            scaled_pixmap = pixmap.scaled(label_size.width() - 40, label_size.height() - 40, 
                                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.uploaded_image_label.setPixmap(scaled_pixmap)
            self.uploaded_image_label.setText("")

            result = Functions.preprocess("offline", file_path)
            
            if result is not None:
                processed_image_path, normalized_face = result
                processed_pixmap = QPixmap(processed_image_path)
                processed_label_size = self.preprocessed_image_label.size()
                scaled_processed_pixmap = processed_pixmap.scaled(
                    processed_label_size.width() - 40, processed_label_size.height() - 40, 
                    Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preprocessed_image_label.setPixmap(scaled_processed_pixmap)
                self.preprocessed_image_label.setText("")

                self.analysis_group.show()
                for button in [self.predict_shape_button, self.predict_gender_button, 
                              self.predict_emotion_button, self.predict_age_button, self.analyze_all_button]:
                    button.show()

                # Show face recognition controls if available
                if self.face_recognition:
                    face_rec_section = self.findChild(QFrame, "faceRecSection")
                    if face_rec_section:
                        face_rec_section.show()

                for label in [self.shape_result, self.gender_result, self.emotion_result, 
                             self.age_result, self.face_recognition_result]:
                    label.hide()

                self.status_label.setText("✅ Image processed successfully! Ready for analysis.")
                
                if not self.model_loader_thread.models_loaded_flag:
                    self.status_label.setText("🔄 Loading AI models...")
                    self.model_loader_thread.start()
                else:
                    self.on_models_loaded()
            else:
                self.preprocessed_image_label.clear()
                self.preprocessed_image_label.setText("❌ Face detection failed\n\nPlease try another image")
                QMessageBox.warning(self, "Error", "No face detected in the image. Please try another image.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image_scaling()

    def update_image_scaling(self):
        if hasattr(self, 'file_path') and self.file_path:
            if self.uploaded_image_label.pixmap():
                pixmap = QPixmap(self.file_path)
                label_size = self.uploaded_image_label.size()
                scaled_pixmap = pixmap.scaled(label_size.width() - 40, label_size.height() - 40, 
                                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.uploaded_image_label.setPixmap(scaled_pixmap)
            
            if self.preprocessed_image_label.pixmap():
                try:
                    result = Functions.preprocess("offline", self.file_path)
                    if result is not None:
                        processed_image_path, _ = result
                        processed_pixmap = QPixmap(processed_image_path)
                        processed_label_size = self.preprocessed_image_label.size()
                        scaled_processed_pixmap = processed_pixmap.scaled(
                            processed_label_size.width() - 40, processed_label_size.height() - 40, 
                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self.preprocessed_image_label.setPixmap(scaled_processed_pixmap)
                except:
                    pass

    def on_models_loaded(self):
        # Message simple et concis
        self.status_label.setText("✅ Models loaded! Click analysis buttons to get predictions.")
        
        if model_manager.shape_model is not None:
            self.predict_shape_button.setEnabled(True)
        # Gender et age buttons toujours activés si face_recognition disponible
        if self.face_recognition is not None:
            self.predict_gender_button.setEnabled(True)
            self.predict_age_button.setEnabled(True)
        if model_manager.emotion_model is not None:
            self.predict_emotion_button.setEnabled(True)
        
        # Enable analyze all button if any models are available
        self.analyze_all_button.setEnabled(True)

    def toggle_face_recognition(self, enabled):
        self.face_recognition_enabled = enabled

    def predict_shape(self):
        if model_manager.shape_model is not None:
            predicted_class, predictions = Functions.predict_shape("offline", self.file_path, model_manager.shape_model)
            self.shape_result.setText(f"🔍 Face Shape: {predicted_class}")
            self.shape_result.show()

    def predict_gender(self):
        # Toujours utiliser face_recognition maintenant (gender.h5 supprimé)
        if self.face_recognition:
            self.analyze_with_face_recognition()
        else:
            # Informer l'utilisateur que la reconnaissance faciale est requise
            self.gender_result.setText("👤 Gender: Requires Face Recognition")
            self.gender_result.show()

    def predict_emotion(self):
        if model_manager.emotion_model is not None:
            predicted_class, predictions = Functions.predict_emotion("offline", self.file_path, model_manager.emotion_model)
            self.emotion_result.setText(f"😊 Emotion: {predicted_class}")
            self.emotion_result.show()

    def predict_age(self):
        # Toujours utiliser face_recognition maintenant pour l'âge aussi
        if self.face_recognition:
            self.analyze_with_face_recognition()
        else:
            # Informer l'utilisateur que la reconnaissance faciale est requise
            self.age_result.setText("🎂 Age: Requires Face Recognition")
            self.age_result.show()

    def analyze_all(self):
        self.predict_shape()
        self.predict_emotion()
        
        # Toujours utiliser InsightFace pour genre et âge maintenant
        if self.face_recognition:
            self.analyze_with_face_recognition()
        else:
            # Informer que la reconnaissance faciale est requise
            self.gender_result.setText("👤 Gender: Requires Face Recognition")
            self.age_result.setText("🎂 Age: Requires Face Recognition")
            self.gender_result.show()
            self.age_result.show()

    def analyze_with_face_recognition(self):
        if not self.face_recognition or not self.current_image_path:
            return
        
        try:
            image = cv2.imread(self.current_image_path)
            if image is None:
                return
            
            face_results = self.face_recognition.analyze_faces(image)
            
            if face_results:
                face_data = face_results[0]
                
                age_exact = face_data.get('age_exact', 'Unknown')
                age_range = face_data.get('age_range', 'Unknown')
                gender = face_data.get('gender', 'Unknown')
                match = face_data.get('match')
                similarity = face_data.get('similarity', 0.0)
                
                if age_exact != 'Unknown' and age_exact is not None:
                    self.age_result.setText(f"🎂 Age: {age_exact:.0f} years ({age_range})")
                else:
                    self.age_result.setText("🎂 Age: Unknown")
                self.age_result.show()
                
                self.gender_result.setText(f"👤 Gender: {gender}")
                self.gender_result.show()
                
                if match:
                    self.face_recognition_result.setText(f"👥 Identity: {match['name']} (Similarity: {similarity:.3f})")
                else:
                    self.face_recognition_result.setText(f"👥 Identity: Unknown (Best match: {similarity:.3f})")
                self.face_recognition_result.show()
            else:
                self.age_result.setText("🎂 Age: No face detected")
                self.age_result.show()
                self.gender_result.setText("👤 Gender: No face detected")
                self.gender_result.show()
                self.face_recognition_result.setText("👥 Identity: No face detected")
                self.face_recognition_result.show()
                
        except Exception as e:
            self.face_recognition_result.setText(f"👥 Identity: Analysis error")
            self.face_recognition_result.show()

    def open_face_recognition_dialog(self):
        dialog = FaceRecognitionDialog(self)
        dialog.exec()

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

    def go_home(self):
        from main import ModernWelcomeWindow
        self.welcome_window = ModernWelcomeWindow()
        self.welcome_window.show()
        self.close()

    def switch_to_online_mode(self):
        from online import ModernOnlineWindow
        self.online_window = ModernOnlineWindow()
        self.online_window.show()
        self.close()

    def closeEvent(self, event):
        self.go_home()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    offline_window = ModernOfflineWindow()
    offline_window.show()
    sys.exit(app.exec())