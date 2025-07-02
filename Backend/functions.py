import cv2
import tensorflow as tf
import numpy as np
import dlib
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Functions:
    _detector = None
    _predictor = None
    _cnn_detector = None
    _thread_lock = threading.Lock()
    _cached_labels = None
    
    @classmethod
    def _initialize_detectors(cls):
        """Initialize detectors safely"""
        if cls._detector is None:
            with cls._thread_lock:
                if cls._detector is None:
                    try:
                        cls._detector = dlib.get_frontal_face_detector()
                        try:
                            cls._predictor = dlib.shape_predictor("Utilities/Face-Detection/shape_predictor_68_face_landmarks.dat")
                        except:
                            cls._predictor = None
                        try:
                            cls._cnn_detector = dlib.cnn_face_detection_model_v1("Utilities/Face-Detection/mmod_human_face_detector.dat")
                        except:
                            cls._cnn_detector = None
                        logger.info("Detectors initialized")
                    except Exception as e:
                        logger.error(f"Error initializing detectors: {e}")
                        cls._detector = dlib.get_frontal_face_detector()
    
    @staticmethod
    def preprocess(method, input_image, target_size=(128, 128)):
        """Fixed preprocessing - handles online/offline modes correctly"""
        try:
            Functions._initialize_detectors()
            
            if method == "offline":
                if not input_image or not isinstance(input_image, str):
                    return None, None
                img = cv2.imread(input_image)
                if img is None:
                    return None, None
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                # Online mode - input is numpy array
                if input_image is None or not isinstance(input_image, np.ndarray):
                    return None, None
                if len(input_image.shape) == 3:
                    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
                else:
                    img = input_image.copy()
            
            # Face detection
            faces = Functions._detector(img, 1)
            if not faces:
                return None, None
            
            # Use largest face
            face = max(faces, key=lambda f: f.width() * f.height())
            
            # Extract face with padding
            padding = 10
            left = max(0, face.left() - padding)
            top = max(0, face.top() - padding)
            right = min(img.shape[1], face.right() + padding)
            bottom = min(img.shape[0], face.bottom() + padding)
            
            extracted_face = img[top:bottom, left:right]
            if extracted_face.size == 0:
                return None, None
            
            # Resize
            resized_face = cv2.resize(extracted_face, target_size)
            path = 'test.jpg'
            cv2.imwrite(path, resized_face)
            
            # Normalize correctly for models
            if len(resized_face.shape) == 2:
                normalized_face = resized_face.astype(np.float32) / 255.0
                normalized_face = np.expand_dims(normalized_face, axis=-1)
                normalized_face = np.expand_dims(normalized_face, axis=0)
            else:
                normalized_face = resized_face.astype(np.float32) / 255.0
                normalized_face = np.expand_dims(normalized_face, axis=0)
            
            return path, normalized_face
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None, None

    @staticmethod
    def predict_shape(method, image_path, model):
        """Fixed shape prediction with proper error handling"""
        try:
            path, preprocessed_image = Functions.preprocess(method, image_path, target_size=(128, 128))
            
            if path is None or preprocessed_image is None:
                return "No face detected", None
            
            if model is None:
                return "Model not loaded", None
            
            predictions = model.predict(preprocessed_image, verbose=0)
            
            # Ensure predictions is not empty
            if predictions is None or len(predictions) == 0:
                return "Prediction failed", None
            
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            
            shape_classes = {0: 'Oblong', 1: 'Square', 2: 'Round', 3: 'Heart', 4: 'Oval'}
            predicted_class = shape_classes.get(predicted_class_index, 'Unknown')
            
            return predicted_class, predictions
            
        except Exception as e:
            logger.error(f"Shape prediction error: {e}")
            return "Prediction Error", None

    # predict_gender function removed - gender detection now handled by InsightFace in face_recognition_core.py

    @staticmethod
    def predict_emotion(method, image_path, model):
        """Fixed emotion prediction with proper index handling"""
        try:
            path, preprocessed_image = Functions.preprocess(method, image_path, target_size=(48, 48))
            
            if path is None or preprocessed_image is None:
                return "No face detected", None
            
            if model is None:
                return "Model not loaded", None
            
            predictions = model.predict(preprocessed_image, verbose=0)
            
            # Ensure predictions is not empty
            if predictions is None or len(predictions) == 0:
                return "Prediction failed", None
            
            # Define emotion classes - adjust based on your model's output
            emotion_classes = ['neutral', 'happy', 'angry', 'surprise', 'sad']
            
            # Get the prediction array
            prediction_array = predictions[0]
            
            # Check if prediction array has valid length
            if len(prediction_array) == 0:
                return "Invalid prediction", None
            
            # Handle different model output sizes
            if len(prediction_array) < len(emotion_classes):
                # If model outputs fewer classes, adjust emotion_classes
                emotion_classes = emotion_classes[:len(prediction_array)]
            elif len(prediction_array) > len(emotion_classes):
                # If model outputs more classes, extend emotion_classes
                for i in range(len(emotion_classes), len(prediction_array)):
                    emotion_classes.append(f'emotion_{i}')
            
            # Get top predictions safely
            try:
                top_indices = np.argsort(prediction_array)[-2:][::-1]
                
                if len(top_indices) >= 2:
                    top1_class = emotion_classes[top_indices[0]]
                    top1_confidence = prediction_array[top_indices[0]] * 100
                    
                    top2_class = emotion_classes[top_indices[1]]
                    top2_confidence = prediction_array[top_indices[1]] * 100
                    
                    predicted_class = f"{top1_class}: {top1_confidence:.1f}% | {top2_class}: {top2_confidence:.1f}%"
                elif len(top_indices) >= 1:
                    top1_class = emotion_classes[top_indices[0]]
                    top1_confidence = prediction_array[top_indices[0]] * 100
                    predicted_class = f"{top1_class}: {top1_confidence:.1f}%"
                else:
                    predicted_class = "Unable to determine emotion"
                    
            except IndexError as ie:
                logger.error(f"Index error in emotion prediction: {ie}")
                # Fallback to simple max prediction
                max_index = np.argmax(prediction_array)
                if max_index < len(emotion_classes):
                    predicted_class = f"{emotion_classes[max_index]}: {prediction_array[max_index] * 100:.1f}%"
                else:
                    predicted_class = f"emotion_{max_index}: {prediction_array[max_index] * 100:.1f}%"
            
            return predicted_class, predictions
            
        except Exception as e:
            logger.error(f"Emotion prediction error: {e}")
            return "Prediction Error", None

    @staticmethod
    def face_detection(image):
        """Fixed face detection for offline mode"""
        try:
            Functions._initialize_detectors()
            
            img = cv2.imread(image)
            if img is None:
                return None
                
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = Functions._detector(img_gray)
            
            if not faces:
                return None
            
            face = max(faces, key=lambda f: f.width() * f.height())
            extracted_face = img_gray[face.top():face.bottom(), face.left():face.right()]
            
            if extracted_face.size == 0:
                return None
            
            resized_face = cv2.resize(extracted_face, (128, 128))
            cv2.imwrite('test.jpg', resized_face)
            
            return resized_face
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return None

    # predict_age function removed - age prediction now handled by InsightFace in face_recognition_core.py

    # _get_age_range function removed - age categorization now handled by InsightFace

    @staticmethod
    def cleanup():
        """Cleanup resources"""
        Functions._detector = None
        Functions._predictor = None
        Functions._cnn_detector = None
        Functions._cached_labels = None