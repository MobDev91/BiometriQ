import sys
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PySide6.QtGui import QPixmap, QIcon, QFontDatabase, QFont
from PySide6.QtCore import Qt, QFile, QTextStream


class ModernWelcomeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("WelcomeWindow")
        
        # Apply modern styling
        self.setStyleSheet(self.get_modern_style())

        # Load the font file
        font_id = QFontDatabase.addApplicationFont("assets/Font/TitilliumWeb-Bold.ttf")
        if font_id != -1:
            QApplication.instance().setFont(QFont("Titillium Web", 10))
        else:
            QApplication.instance().setFont(QFont("Segoe UI", 10))

        self.setWindowTitle("BiometriQ - Modern Biometric System")
        
        # Make window resizable with minimum size
        self.setMinimumSize(600, 400)
        self.resize(700, 500)  # Default size, but resizable

        # Set Window Icon
        icon_path = "assets/Icons/favicon-black.png"
        self.setWindowIcon(QIcon(icon_path))

        # Center the window on the screen
        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # Title Section
        title_layout = QVBoxLayout()
        title_layout.setAlignment(Qt.AlignCenter)
        
        # App Title
        title_label = QLabel("BiometriQ")
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel("AI-Powered Facial Biometric Analysis")
        subtitle_label.setObjectName("subtitle")
        subtitle_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(subtitle_label)

        # Program Information
        program_label = QLabel("Master Embedded Artificial Intelligence")
        program_label.setObjectName("program")
        program_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(program_label)

        # Development Team
        dev_team_label = QLabel("Dev Team:\nBENYSSEF OUSSAMA\nEL FATHI ABDESSAMAD")
        dev_team_label.setObjectName("devTeam")
        dev_team_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(dev_team_label)

        main_layout.addLayout(title_layout)

        # Logo Section
        logo_container = QVBoxLayout()
        logo_container.setAlignment(Qt.AlignCenter)
        
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        try:
            pixmap = QPixmap("assets/Icons/logo.png")
            scaled_pixmap = pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
        except:
            logo_label.setText("🔍")
            logo_label.setStyleSheet("font-size: 64px;")
        
        logo_container.addWidget(logo_label)
        main_layout.addLayout(logo_container)

        # Features hint
        features_label = QLabel("• Face Recognition  • Gender Detection  • Emotion Analysis  • Shape Classification")
        features_label.setObjectName("features")
        features_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(features_label)

        # Buttons Section
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)
        button_layout.setAlignment(Qt.AlignCenter)

        # Online Mode Button
        online_button = QPushButton("📷 Live Camera")
        online_button.setObjectName("primaryButton")
        online_button.setMinimumSize(160, 50)
        online_button.clicked.connect(self.open_online_window)
        button_layout.addWidget(online_button)

        # Offline Mode Button
        offline_button = QPushButton("📁 Upload Image")
        offline_button.setObjectName("secondaryButton")
        offline_button.setMinimumSize(160, 50)
        offline_button.clicked.connect(self.open_offline_window)
        button_layout.addWidget(offline_button)

        main_layout.addLayout(button_layout)

        # Footer
        footer_label = QLabel("Choose your preferred analysis mode")
        footer_label.setObjectName("footer")
        footer_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(footer_label)

        self.setLayout(main_layout)

    def get_modern_style(self):
        return """
        QWidget#WelcomeWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                       stop:0 #0f172a, stop:0.5 #1e293b, stop:1 #0f172a);
        }
        
        QLabel#title {
            font-size: 36px;
            font-weight: 700;
            color: #f8fafc;
            margin: 0 0 10px 0;
        }
        
        QLabel#subtitle {
            font-size: 18px;
            color: #cbd5e1;
            font-weight: 400;
            margin-bottom: 20px;
        }
        
        QLabel#program {
            font-size: 16px;
            color: #a78bfa;
            font-weight: 600;
            margin: 10px 0;
            padding: 10px;
            background: rgba(139, 92, 246, 0.1);
            border-radius: 8px;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }
        
        QLabel#devTeam {
            font-size: 14px;
            color: #e2e8f0;
            font-weight: 500;
            margin: 10px 0;
            padding: 15px;
            background: rgba(30, 41, 59, 0.8);
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            line-height: 1.4;
        }
        
        QLabel#features {
            font-size: 14px;
            color: #94a3b8;
            margin: 20px 0;
            padding: 15px;
            background: rgba(30, 41, 59, 0.6);
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        
        QLabel#footer {
            font-size: 14px;
            color: #64748b;
            margin-top: 20px;
        }
        
        QPushButton#primaryButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 #3b82f6, stop:1 #2563eb);
            border: none;
            border-radius: 15px;
            color: white;
            font-weight: 600;
            font-size: 16px;
            padding: 15px 25px;
        }
        
        QPushButton#primaryButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 #2563eb, stop:1 #1d4ed8);
        }
        
        QPushButton#primaryButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 #1d4ed8, stop:1 #1e40af);
        }
        
        QPushButton#secondaryButton {
            background: rgba(55, 65, 81, 0.8);
            border: 2px solid #4b5563;
            border-radius: 15px;
            color: #f1f5f9;
            font-weight: 600;
            font-size: 16px;
            padding: 15px 25px;
        }
        
        QPushButton#secondaryButton:hover {
            background: rgba(75, 85, 99, 0.9);
            border-color: #6b7280;
        }
        
        QPushButton#secondaryButton:pressed {
            background: rgba(107, 114, 128, 0.9);
        }
        """

    def open_online_window(self):
        from online import ModernOnlineWindow
        self.online_window = ModernOnlineWindow()
        self.online_window.show()
        self.hide()

    def open_offline_window(self):
        from offline import ModernOfflineWindow
        self.offline_window = ModernOfflineWindow()
        self.offline_window.show()
        self.hide()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    welcome_window = ModernWelcomeWindow()
    welcome_window.show()
    sys.exit(app.exec())
