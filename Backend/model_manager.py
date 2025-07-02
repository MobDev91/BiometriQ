import os
import threading
import logging
from keras.models import load_model
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedModelManager:
    def __init__(self):
        # Removed gender_model and age_model - now both handled by InsightFace
        self.shape_model = None
        self.emotion_model = None
        # self.recognizer = None  # REMOVED - Face recognition functionality removed
        self._load_lock = threading.Lock()
        self._models_loaded = False
        
        # Optimize TensorFlow for performance
        self._configure_tensorflow()

    def _configure_tensorflow(self):
        """Configure TensorFlow for optimal performance"""
        try:
            # Enable GPU memory growth if available
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configured {len(gpus)} GPU(s) with memory growth")
            
            # Set thread configurations for CPU optimization
            tf.config.threading.set_inter_op_parallelism_threads(0)
            tf.config.threading.set_intra_op_parallelism_threads(0)
            
            # Enable XLA compilation for faster execution
            tf.config.optimizer.set_jit(True)
            
            logger.info("TensorFlow optimizations applied")
            
        except Exception as e:
            logger.warning(f"Could not fully optimize TensorFlow: {e}")

    def _validate_model_files(self):
        """Validate that model files exist and return available ones"""
        model_files = {
            # Removed 'gender': gender detection now handled by InsightFace
            'shape': 'Models/shape.h5', 
            'emotion': 'Models/emotion.h5',
        }
        
        existing_files = []
        missing_files = []
        
        for name, path in model_files.items():
            if os.path.exists(path):
                existing_files.append(name)
                logger.info(f"✅ Found {name} model: {path}")
            else:
                missing_files.append(f"{name} ({path})")
                logger.warning(f"⚠️ Missing {name} model: {path}")
        
        if len(existing_files) == 0:
            raise FileNotFoundError("No model files found!")
        
        logger.info(f"Found {len(existing_files)} out of {len(model_files)} models")
        return model_files, existing_files

    def load_models(self):
        """Load all available models with optimizations and error handling"""
        if self._models_loaded:
            logger.info("Models already loaded")
            return

        with self._load_lock:
            if self._models_loaded:  # Double-check locking
                return

            try:
                logger.info("Starting model loading...")
                
                # Validate files and get available ones
                model_files, existing_files = self._validate_model_files()
                
                # Load available models
                # Removed gender model loading - now handled by InsightFace
                
                if 'shape' in existing_files:
                    logger.info("Loading shape model...")
                    self.shape_model = self._load_single_model(
                        model_files['shape'], 
                        "Face Shape Prediction"
                    )
                
                if 'emotion' in existing_files:
                    logger.info("Loading emotion model...")
                    self.emotion_model = self._load_single_model(
                        model_files['emotion'], 
                        "Emotion Recognition"
                    )
                
                # Warm up models with dummy predictions
                self._warmup_models()
                
                self._models_loaded = True
                loaded_count = sum(1 for model in [self.shape_model, 
                                                 self.emotion_model] 
                                 if model is not None)
                logger.info(f"Successfully loaded {loaded_count} models! (Gender & Age handled by InsightFace)")
                
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                self._cleanup_models()
                raise

    def _load_single_model(self, model_path, model_name):
        """Load a single model with optimizations"""
        try:
            # Load with compile=False for faster loading, then compile manually
            model = load_model(model_path, compile=False)
            
            # Recompile with optimized settings based on model type
            # Removed gender model compilation - now handled by InsightFace
            if 'emotion' in model_path.lower():
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy', 
                    metrics=['accuracy']
                )
            elif 'shape' in model_path.lower():
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
            
            # Set model to inference mode for better performance
            model.trainable = False
            
            logger.info(f"✅ {model_name} loaded: {model.count_params():,} parameters")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load {model_name} from {model_path}: {e}")
            raise

    def _warmup_models(self):
        """Warm up models with dummy predictions for better initial performance"""
        try:
            logger.info("Warming up models...")
            import numpy as np
            
            # Create dummy inputs for different model types
            dummy_224 = np.random.random((1, 224, 224, 1)).astype(np.float32)  # For age models
            dummy_128 = np.random.random((1, 128, 128, 1)).astype(np.float32)
            dummy_48 = np.random.random((1, 48, 48, 1)).astype(np.float32)
            
            # Warmup each model
            # Removed gender and age model warmup - now handled by InsightFace
                
            if self.emotion_model:
                self.emotion_model.predict(dummy_48, verbose=0)
                logger.info("Emotion model warmed up")
                
            if self.shape_model:
                self.shape_model.predict(dummy_128, verbose=0)
                logger.info("Shape model warmed up")
                
            logger.info("Model warmup completed")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def _cleanup_models(self):
        """Clean up loaded models"""
        try:
            models = [self.shape_model, self.emotion_model]
            model_names = ['shape_model', 'emotion_model']
            
            for model, name in zip(models, model_names):
                if model is not None:
                    del model
                    setattr(self, name, None)
                    logger.info(f"Cleaned up {name}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")

    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            'models_loaded': self._models_loaded,
            'gender_model': False,  # Now handled by InsightFace
            'shape_model': self.shape_model is not None,
            'emotion_model': self.emotion_model is not None,
            'age_model': False  # Now handled by InsightFace
        }
        
        if self._models_loaded:
            try:
                # Add parameter counts
                # Removed gender_params and age_params - now handled by InsightFace
                if self.shape_model:
                    info['shape_params'] = self.shape_model.count_params()
                if self.emotion_model:
                    info['emotion_params'] = self.emotion_model.count_params()
                    
            except Exception as e:
                logger.warning(f"Could not get model parameters: {e}")
        
        return info

    def has_age_model(self):
        """Check if age model is available - now handled by InsightFace"""
        return False  # Age prediction now handled by InsightFace

    def unload_models(self):
        """Explicitly unload all models to free memory"""
        with self._load_lock:
            logger.info("Unloading models...")
            self._cleanup_models()
            self._models_loaded = False
            logger.info("Models unloaded")

    def reload_models(self):
        """Reload all models"""
        logger.info("Reloading models...")
        self.unload_models()
        self.load_models()

    def is_ready(self):
        """Check if core models are loaded and ready"""
        return (self._models_loaded and 
                self.shape_model is not None and
                self.emotion_model is not None)
                # Removed gender_model check - now handled by InsightFace

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self._cleanup_models()
        except Exception:
            pass

# Create a global instance of the optimized model manager
model_manager = OptimizedModelManager()

# Backward compatibility
class ModelManager:
    """Legacy wrapper for backward compatibility"""
    def __init__(self):
        global model_manager
        self._manager = model_manager

    def load_models(self):
        return self._manager.load_models()

    @property
    def gender_model(self):
        # Gender detection now handled by InsightFace
        return None

    @property
    def shape_model(self):
        return self._manager.shape_model

    @property
    def emotion_model(self):
        return self._manager.emotion_model

    @property
    def recognizer(self):
        # Return None since recognizer functionality was removed
        return None
    
    @property
    def age_model(self):
        # Age prediction now handled by InsightFace
        return None
