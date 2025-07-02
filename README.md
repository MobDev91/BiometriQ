<div id='top'></div>

# 🎯 BiometriQ

**_An Advanced Multimodal Facial Biometric System powered by AI_**

_Real-time facial recognition, emotion analysis, face shape prediction, and age/gender detection using state-of-the-art deep learning models._

<div align='center'>

![BiometriQ Logo](assets/Icons/logo.png)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PySide6](https://img.shields.io/badge/PySide6-6.6.1-green.svg)](https://doc.qt.io/qtforpython/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange.svg)](https://tensorflow.org)
[![InsightFace](https://img.shields.io/badge/InsightFace-0.7.3+-red.svg)](https://github.com/deepinsight/insightface)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [🚀 Features](#features)
- [🏗️ Project Architecture](#architecture)
- [⚙️ Installation](#installation)
- [🎮 Usage](#usage)
- [🤖 AI Models](#models)
- [🖥️ Interface](#interface)
- [📊 Performance](#performance)
- [🔧 Technical Details](#technical)
- [👥 Team](#team)
- [📄 License](#license)

---

<div id="features">

## 🚀 Features

### Core Capabilities

- 🎥 **Real-time Analysis** - Live camera feed processing with instant results
- 📁 **Image Analysis** - Upload and analyze static images
- 👤 **Face Recognition** - Advanced identity recognition with InsightFace
- 🎭 **Emotion Detection** - Multi-class emotion recognition (5 emotions)
- 🔍 **Face Shape Analysis** - Geometric face shape classification (5 categories)
- 👫 **Age & Gender Prediction** - Accurate demographic analysis
- 🗄️ **Face Database** - Persistent storage with HDF5 database
- ⚡ **GPU Acceleration** - CUDA and Metal support for enhanced performance

### Technical Highlights

- **Hybrid Architecture** - Combines custom models with InsightFace
- **Thread-Safe Processing** - Optimized for real-time performance
- **Modern UI** - Clean, responsive interface built with PySide6
- **Cross-Platform** - Supports Windows, macOS, and Linux
- **Modular Design** - Easy to extend and maintain

</div>

---

<div id="architecture">

## 🏗️ Project Architecture

```
BiometriQ/
├── 🎯 main.py                     # Application entry point
├── 📁 offline.py                  # Image upload & analysis mode
├── 📷 online.py                   # Real-time camera mode
├── 📋 requirements.txt            # Dependencies specification
├──
├── 🧠 Backend/
│   ├── functions.py               # Core processing functions
│   ├── model_manager.py           # AI model management
│   └── face_recognition_core.py   # InsightFace integration
├──
├── 🤖 Models/
│   ├── emotion.h5                 # Emotion recognition model
│   └── shape.h5                   # Face shape prediction model
├──
├── 🛠️ Utilities/
│   └── Face-Detection/
│       ├── mmod_human_face_detector.dat      # CNN face detector
│       └── shape_predictor_68_face_landmarks.dat  # 68 facial landmarks
├──
├── 🎨 assets/
│   └── Icons/
│       └── logo.png               # Application logo
├──
├── 🧪 test_examples/              # Sample images for testing
├── 📚 Notebooks/                  # Research & development notebooks
└── 🗄️ face_database.h5            # Face recognition database
```

### Key Components

#### 🎯 **Main Application** (`main.py`)

- Welcome screen with mode selection
- Navigation between online/offline modes
- Application lifecycle management

#### 🧠 **Backend Processing**

- **`functions.py`** - Image preprocessing, face detection, model inference
- **`model_manager.py`** - Efficient model loading and memory management
- **`face_recognition_core.py`** - InsightFace integration for recognition/age/gender

#### 🤖 **AI Models**

- **Custom Models** - Shape and emotion detection
- **InsightFace** - Age, gender, and identity recognition
- **dlib Models** - Face detection and landmark extraction

</div>

---

<div id="installation">

## ⚙️ Installation

### Prerequisites

- **Python 3.9+** (Python 3.11 recommended)
- **4GB RAM minimum** (8GB recommended)
- **Webcam** (for real-time mode)
- **Optional**: NVIDIA GPU with CUDA for acceleration

### Quick Install

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/BiometriQ.git
cd BiometriQ

# 2. Create virtual environment (recommended)
python -m venv biometriq_env
source biometriq_env/bin/activate  # On Windows: biometriq_env\Scripts\activate

# 3. Or manual installation
pip install -r requirements.txt
```

### Dependencies Overview

| Category      | Packages                                      |
| ------------- | --------------------------------------------- |
| **Core ML**   | `tensorflow`, `keras`, `numpy`, `dlib`        |
| **Face AI**   | `insightface`, `onnxruntime`, `opencv-python` |
| **Interface** | `PySide6`, `matplotlib`                       |
| **Data**      | `h5py`, `Pillow`, `scikit-learn`, `scipy`     |

### GPU Setup (Optional)

```bash
# For NVIDIA GPU acceleration
pip install tensorflow[and-cuda]==2.16.1
pip install onnxruntime-gpu>=1.15.0

# For Apple Silicon (Metal)
pip install tensorflow-metal
```

</div>

---

<div id="usage">

## 🎮 Usage

### Launching the Application

```bash
python main.py
```

### Operating Modes

#### 📁 **Offline Mode** - Image Analysis

1. Click **"📁 Upload Image"** from welcome screen
2. Select image file (PNG, JPG, JPEG supported)
3. View instant analysis results:
   - 🔍 Face shape prediction
   - 😊 Emotion recognition
   - 👤 Gender classification
   - 🎂 Age estimation
   - 👥 Identity recognition (if face is in database)

#### 📷 **Online Mode** - Real-time Camera

1. Click **"📷 Live Camera"** from welcome screen
2. Grant camera permissions when prompted
3. Click **"🎥 Start Camera"** to begin analysis
4. View live results updating every second
5. Use **"📸 Capture"** to add faces to recognition database

### Face Recognition Database

#### Adding New Faces

1. **From Camera**: Use "📸 Capture from Camera" in Face Recognition dialog
2. **From Image**: Use "📁 Upload Image" option
3. Enter person's name and confirm

#### Managing Database

- **Search**: Find specific people in database
- **Statistics**: View database demographics and metrics
- **Delete**: Remove individuals from recognition system

</div>

---

<div id="models">

## 🤖 AI Models & Performance

### Model Architecture

| Model           | Technology  | Input Size | Classes           | Accuracy |
| --------------- | ----------- | ---------- | ----------------- | -------- |
| **Face Shape**  | Custom CNN  | 128×128    | 5 shapes          | 88.03%   |
| **Emotion**     | Custom CNN  | 48×48      | 5 emotions        | 66.13%   |
| **Age/Gender**  | InsightFace | 112×112    | Continuous/Binary | 95%+     |
| **Recognition** | InsightFace | 112×112    | Embedding         | 99%+     |

### Supported Classifications

#### 🔍 **Face Shapes**

- **Oval** - Balanced proportions, slightly longer than wide
- **Round** - Equal width and height, soft curves
- **Square** - Strong jawline, equal width and height
- **Heart** - Wider forehead, narrower chin
- **Oblong** - Longer face, narrow features

#### 😊 **Emotions**

- **Neutral** - Relaxed, no strong emotion
- **Happy** - Smiling, positive expression
- **Angry** - Frowning, tense features
- **Surprise** - Wide eyes, raised eyebrows
- **Sad** - Downturned mouth, drooping features

#### 👤 **Demographics**

- **Age**: Precise prediction (±3 years accuracy)
- **Gender**: Binary classification (Male/Female)
- **Identity**: Face matching with similarity scores

### Training Datasets

| Model      | Dataset              | Samples | Source                                                               |
| ---------- | -------------------- | ------- | -------------------------------------------------------------------- |
| Face Shape | Celebrity Face-Shape | 5,000+  | [Research Paper](https://www.researchgate.net/publication/328775300) |
| Emotion    | FER2013              | 35,887  | [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)           |
| Age/Gender | InsightFace          | 1M+     | Various curated datasets                                             |

</div>

---

<div id="interface">

## 🖥️ User Interface

### Design Philosophy

- **Modern Dark Theme** - Easy on the eyes for extended use
- **Intuitive Navigation** - Clear visual hierarchy and flow
- **Real-time Feedback** - Instant visual confirmation of actions
- **Responsive Layout** - Adapts to different screen sizes

### Screenshots

| Interface            | Preview                                  |
| -------------------- | ---------------------------------------- |
| **Welcome Screen**   | Clean entry point with mode selection    |
| **Offline Analysis** | Side-by-side image and results display   |
| **Online Camera**    | Live feed with overlay analysis results  |
| **Face Database**    | Tabbed interface for database management |

### Key UI Features

#### 📱 **Responsive Design**

- Minimum resolution: 1200×800
- Scales appropriately on larger displays
- Scroll areas for overflow content

#### 🎨 **Visual Elements**

- **Color-coded results** - Green for success, blue for info, purple for recognition
- **Progress indicators** - Loading states and model status
- **Interactive controls** - Sliders, toggles, and buttons with hover effects

#### ⌨️ **Keyboard Shortcuts**

- `ESC` - Return to welcome screen
- `Space` - Start/stop camera (in online mode)
- `Ctrl+O` - Switch to offline mode
- `Ctrl+L` - Switch to online mode

</div>

---

<div id="performance">

## 📊 Performance Metrics

### System Requirements

| Component   | Minimum                | Recommended            |
| ----------- | ---------------------- | ---------------------- |
| **CPU**     | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| **RAM**     | 4GB                    | 8GB+                   |
| **GPU**     | Integrated             | NVIDIA GTX 1060+       |
| **Storage** | 2GB free space         | 5GB+ for datasets      |

### Benchmark Results

#### Processing Speed (per frame)

- **Face Detection**: ~15ms (CPU) / ~3ms (GPU)
- **Shape Analysis**: ~25ms (CPU) / ~8ms (GPU)
- **Emotion Recognition**: ~20ms (CPU) / ~6ms (GPU)
- **Age/Gender (InsightFace)**: ~45ms (CPU) / ~12ms (GPU)
- **Face Recognition**: ~30ms (CPU) / ~8ms (GPU)

#### Real-time Performance

- **Target FPS**: 30 FPS
- **Achieved FPS**: 25-30 FPS (GPU) / 15-20 FPS (CPU)
- **Analysis Interval**: Every 30 frames (1 second)

### Memory Usage

- **Base Application**: ~200MB
- **With Models Loaded**: ~800MB
- **Peak Usage (GPU)**: ~1.2GB
- **Face Database**: ~1MB per 100 faces

</div>

---

<div id="technical">

## 🔧 Technical Details

### Core Technologies

#### 🧠 **Machine Learning Stack**

- **TensorFlow 2.16.1** - Primary ML framework
- **Keras 3.0.2** - High-level neural network API
- **ONNX Runtime** - Optimized inference engine
- **InsightFace** - State-of-the-art face recognition

#### 🖼️ **Computer Vision**

- **OpenCV** - Image processing and camera handling
- **dlib** - Face detection and landmark extraction
- **PIL/Pillow** - Image manipulation utilities

#### 🖥️ **Interface & System**

- **PySide6** - Cross-platform GUI framework
- **Qt6** - Native look and feel on all platforms
- **HDF5** - Efficient data storage format

### Architecture Patterns

#### 🔄 **Threading Model**

- **Main Thread** - UI and event handling
- **Camera Thread** - Video capture and frame processing
- **Model Thread** - AI inference operations
- **Database Thread** - Face recognition database operations

#### 🧩 **Design Patterns**

- **Singleton** - Model manager for resource efficiency
- **Observer** - Qt signals/slots for component communication
- **Factory** - Dynamic model loading and initialization
- **State Machine** - Application mode management

### Security & Privacy

#### 🔒 **Data Protection**

- **Local Processing** - No data transmitted to external servers
- **Encrypted Storage** - Face databases can be encrypted
- **User Consent** - Clear permissions for camera access
- **Data Deletion** - Complete removal of face data on request

#### 🛡️ **Robust Error Handling**

- **Graceful Degradation** - App functions with missing models
- **Resource Cleanup** - Proper memory and thread management
- **Input Validation** - Sanitized file and camera inputs
- **Exception Recovery** - Automatic recovery from errors

</div>

---

<div id="team">

## 👥 Development Team

### Core Contributors

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/MobDev91.png" width="100px;" alt="Oussama Benyssef"/>
      <br />
      <sub><b>Oussama Benyssef</b></sub>
      <br />
      <a href="https://github.com/MobDev91">🔗 GitHub</a>
      <br />
      <small>Lead Developer & AI Engineer</small>
    </td>
    <td align="center">
      <img src="https://github.com/AbdoElfathi.png" width="100px;" alt="Abdessamad El Fathi"/>
      <br />
      <sub><b>Abdessamad El Fathi</b></sub>
      <br />
      <a href="https://github.com/AbdoElfathi">🔗 GitHub</a>
      <br />
      <small>Lead Developer & AI Engineer</small>
    </td>
  </tr>
</table>

### Contributions

- **Architecture Design** - System design and component integration
- **AI Model Development** - Custom CNN training and optimization
- **InsightFace Integration** - Advanced face recognition implementation
- **UI/UX Design** - Modern interface design and user experience
- **Performance Optimization** - Threading, GPU acceleration, memory management
- **Testing & Validation** - Comprehensive testing across platforms

</div>

---

## 🤝 Contributing

We welcome contributions! just contact us.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/BiometriQ.git

# Create a feature branch
git checkout -b feature/amazing-feature

# Make your changes and commit
git commit -m 'Add amazing feature'

# Push and create a Pull Request
git push origin feature/amazing-feature
```

## 🐛 Issues & Support

- **Bug Reports**: [Create an Issue](https://github.com/yourusername/BiometriQ/issues)
- **Feature Requests**: [Discussions](https://github.com/yourusername/BiometriQ/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/BiometriQ/wiki)

---

<div id="license">

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- **InsightFace**: [MIT License](https://github.com/deepinsight/insightface/blob/master/LICENSE)
- **dlib**: [Boost Software License](https://github.com/davisking/dlib/blob/master/LICENSE.txt)
- **TensorFlow**: [Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)

</div>

---

## 🙏 Acknowledgments

- **InsightFace Team** for the excellent face recognition framework
- **dlib contributors** for robust face detection algorithms
- **TensorFlow/Keras teams** for the ML infrastructure
- **Qt/PySide6** for the cross-platform GUI framework
- **Research Community** for the datasets and methodologies

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

**🔗 [Back to Top](#top)**

_Built with ❤️ using Python, TensorFlow, and InsightFace_

</div>
