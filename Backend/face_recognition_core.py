"""
Face Recognition and Age Prediction Core Module
Integrated from face_age.ipynb notebook
"""

import cv2
import numpy as np
import h5py
import json
import os
from datetime import datetime
import warnings
import logging
from typing import Tuple, List, Dict, Optional, Any



# Suppress warnings
warnings.filterwarnings('ignore')

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available. Please install: pip install insightface")

def convert_numpy(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

class FaceRecognitionCore:
    """Core face recognition and age prediction system"""
    
    def __init__(self, db_path: str = "face_database.h5"):
        self.db_path = db_path
        self.face_embeddings = []
        self.face_metadata = []
        self.app = None
        
        # Initialize InsightFace model
        self._initialize_models()
        
        # Load existing database
        self.load_database()
    
    def _initialize_models(self):
        """Initialize face analysis models"""
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace is required. Install with: pip install insightface")
        
        try:
            print("🤖 Initializing Face Analysis Models...")
            # Initialize InsightFace with GPU support if available
            self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            
            print("✅ Face analysis model initialized successfully!")
            print("🎯 Model capabilities:")
            print("  • Face detection")
            print("  • Face recognition (embeddings)")
            print("  • Age prediction")
            print("  • Gender classification")
            print("  • Face quality assessment")
            
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            raise
    
    def load_database(self):
        """Load existing database or create new one"""
        try:
            if os.path.exists(self.db_path):
                with h5py.File(self.db_path, 'r') as f:
                    if 'embeddings' in f:
                        self.face_embeddings = list(f['embeddings'][:])
                    
                    if 'metadata' in f:
                        metadata_strings = f['metadata'][:]
                        self.face_metadata = []
                        for meta in metadata_strings:
                            try:
                                decoded_meta = meta.decode() if isinstance(meta, bytes) else meta
                                self.face_metadata.append(json.loads(decoded_meta))
                            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                                print(f"Warning: Could not decode metadata: {e}")
                                continue
                
                print(f"✅ Loaded database with {len(self.face_embeddings)} faces")
            else:
                print("📝 Creating new face database")
                self.face_embeddings = []
                self.face_metadata = []
        except Exception as e:
            print(f"⚠️ Error loading database: {e}. Creating new database.")
            self.face_embeddings = []
            self.face_metadata = []
    
    def save_database(self):
        """Save database to H5 file"""
        try:
            with h5py.File(self.db_path, 'w') as f:
                if self.face_embeddings:
                    f.create_dataset('embeddings', data=np.array(self.face_embeddings))
                    
                    # Save metadata as JSON strings
                    metadata_strings = [json.dumps(meta, default=convert_numpy) for meta in self.face_metadata]

                    dt = h5py.special_dtype(vlen=str)
                    f.create_dataset('metadata', data=metadata_strings, dtype=dt)
                    
                    f.attrs['total_faces'] = len(self.face_embeddings)
                    f.attrs['last_updated'] = str(datetime.now())
            
            print(f"✅ Database saved with {len(self.face_embeddings)} faces")
        except Exception as e:
            print(f"❌ Error saving database: {e}")
    
    def get_age_range(self, age) -> str:
        """Convert exact age to age range"""
        if age == 'Unknown' or age is None:
            return "Unknown"
        
        try:
            age = float(age)
            if age < 4:
                return "0-3 (Baby)"
            elif age < 8:
                return "4-7 (Child)"
            elif age < 12:
                return "8-11 (Kid)"
            elif age < 16:
                return "12-15 (Preteen)"
            elif age < 20:
                return "16-19 (Teen)"
            elif age < 24:
                return "20-23 (Young)"
            elif age < 28:
                return "24-27 (Adult)"
            elif age < 32:
                return "28-31 (Adult)"
            elif age < 36:
                return "32-35 (Adult)"
            elif age < 40:
                return "36-39 (Mature)"
            elif age < 44:
                return "40-43 (Mature)"
            elif age < 48:
                return "44-47 (Middle Age)"
            elif age < 52:
                return "48-51 (Middle Age)"
            elif age < 56:
                return "52-55 (Senior)"
            elif age < 60:
                return "56-59 (Senior)"
            elif age < 64:
                return "60-63 (Senior)"
            elif age < 68:
                return "64-67 (Elder)"
            else:
                return "68+ (Elder)"
        except:
            return "Unknown"
    
    def analyze_faces(self, image: np.ndarray) -> List[Dict]:
        """Analyze faces in an image with proper error handling"""
        if self.app is None:
            raise RuntimeError("Face analysis model not initialized")
        
        try:
            faces = self.app.get(image)
            results = []
            
            for i, face in enumerate(faces):
                try:
                    # Extract face information with safe attribute access
                    embedding = getattr(face, 'embedding', None)
                    if embedding is None:
                        continue
                        
                    age_exact = getattr(face, 'age', None)
                    age_range = self.get_age_range(age_exact)
                    gender_raw = getattr(face, 'gender', None)
                    confidence = getattr(face, 'det_score', 0.0)
                    bbox = getattr(face, 'bbox', None)
                    
                    # Convert gender from numeric to text with error handling
                    if gender_raw == 0:
                        gender = "Female"
                    elif gender_raw == 1:
                        gender = "Male"
                    else:
                        gender = "Unknown"
                    
                    # Try to recognize face
                    match, similarity = self.find_similar_face(embedding)
                    
                    face_data = {
                        'embedding': embedding,
                        'age_exact': age_exact,
                        'age_range': age_range,
                        'gender': gender,
                        'confidence': confidence,
                        'bbox': bbox,
                        'match': match,
                        'similarity': similarity
                    }
                    
                    results.append(face_data)
                    
                except Exception as face_error:
                    print(f"Error processing face {i}: {face_error}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"Error in analyze_faces: {e}")
            return []
    
    def add_face(self, embedding: np.ndarray, name: str, age: str, 
                 confidence: float, source: str = "upload", gender: str = "Unknown"):
        """Add a new face to the database"""
        try:
            metadata = {
                'id': len(self.face_embeddings),
                'name': name,
                'age': age,
                'gender': gender,
                'confidence': confidence,
                'source': source,
                'timestamp': str(datetime.now())
            }
            
            self.face_embeddings.append(embedding.tolist())
            self.face_metadata.append(metadata)
            print(f"✅ Added face: {name} (Age: {age})")
        except Exception as e:
            print(f"Error adding face: {e}")
    
    
    
    def find_similar_face(self, query_embedding: np.ndarray, 
                          threshold: float = 0.4) -> Tuple[Optional[Dict], float]:
        """Find similar face in database with error handling"""
        if not self.face_embeddings:
            print("[WARN] No face embeddings found in database.")
            return None, 0.0

        try:
            similarities = []
            for i, db_embedding in enumerate(self.face_embeddings):
                try:
                    db_embedding = np.array(db_embedding)

                    # Ensure embeddings have same shape
                    if query_embedding.shape != db_embedding.shape:
                        print(f"[DEBUG] Shape mismatch: query {query_embedding.shape}, db {db_embedding.shape}")
                        continue

                    # Calculate cosine similarity with error handling
                    norm_query = np.linalg.norm(query_embedding)
                    norm_db = np.linalg.norm(db_embedding)

                    if norm_query == 0 or norm_db == 0:
                        similarity = 0.0
                    else:
                        similarity = np.dot(query_embedding, db_embedding) / (norm_query * norm_db)

                    print(f"[DEBUG] Comparing with face {i}: similarity = {similarity:.4f}")
                    similarities.append((similarity, i))

                except Exception as e:
                    print(f"[ERROR] Error computing similarity for embedding {i}: {e}")
                    continue

            if not similarities:
                print("[WARN] No valid similarity comparisons made.")
                return None, 0.0

            # Get best match
            best_similarity, best_index = max(similarities, key=lambda x: x[0])

            if best_index >= len(self.face_metadata):
                print(f"[ERROR] Metadata not found for index {best_index}")
                return None, best_similarity

            best_match = self.face_metadata[best_index]

            if best_similarity >= threshold:
                print(f"[INFO] Best match at index {best_index}, similarity = {best_similarity:.4f}")
                return best_match, best_similarity
            else:
                print(f"[INFO] No match found. Best similarity = {best_similarity:.4f}")
                return None, best_similarity

        except Exception as e:
            print(f"[ERROR] Error in find_similar_face: {e}")
            return None, 0.0
    
    def search_person(self, person_name: str) -> List[Tuple[int, Dict]]:
        """Search for a specific person in the database"""
        matches = []
        
        try:
            for i, meta in enumerate(self.face_metadata):
                if isinstance(meta, dict) and 'name' in meta:
                    if person_name.lower() in meta['name'].lower():
                        matches.append((i, meta))
        except Exception as e:
            print(f"Error searching for person: {e}")
        
        return matches
    
    def delete_person(self, person_name: str) -> int:
        """Delete a person from the database"""
        try:
            matches = self.search_person(person_name)
            
            if not matches:
                return 0
            
            # Remove in reverse order to maintain indices
            for idx, _ in reversed(matches):
                if idx < len(self.face_embeddings):
                    del self.face_embeddings[idx]
                if idx < len(self.face_metadata):
                    del self.face_metadata[idx]
            
            return len(matches)
        except Exception as e:
            print(f"Error deleting person: {e}")
            return 0
    
    def get_database_stats(self) -> Dict:
        """Get database statistics with error handling"""
        try:
            if not self.face_metadata:
                return {"total_faces": 0, "age_groups": {}, "sources": {}, "gender_distribution": {}}
            
            total_faces = len(self.face_metadata)
            sources = {}
            age_groups = {}
            gender_distribution = {}
            
            for meta in self.face_metadata:
                try:
                    if not isinstance(meta, dict):
                        continue
                        
                    source = meta.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
                    
                    age = meta.get('age', 'Unknown')
                    if age != 'Unknown':
                        age_groups[age] = age_groups.get(age, 0) + 1
                    
                    gender = meta.get('gender', 'Unknown')
                    gender_distribution[gender] = gender_distribution.get(gender, 0) + 1
                    
                except Exception as e:
                    print(f"Error processing metadata: {e}")
                    continue
            
            return {
                "total_faces": total_faces,
                "age_groups": age_groups,
                "sources": sources,
                "gender_distribution": gender_distribution
            }
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {"total_faces": 0, "age_groups": {}, "sources": {}, "gender_distribution": {}}
    
    def draw_annotations(self, image: np.ndarray, face_results: List[Dict]) -> np.ndarray:
        """Draw face annotations on image with error handling"""
        try:
            annotated_image = image.copy()
            
            for face_data in face_results:
                try:
                    bbox = face_data.get('bbox')
                    if bbox is None:
                        continue
                        
                    x1, y1, x2, y2 = bbox.astype(int)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Prepare text
                    texts = []
                    match = face_data.get('match')
                    if match:
                        texts.append(f"Name: {match.get('name', 'Unknown')}")
                        texts.append(f"Similarity: {face_data.get('similarity', 0.0):.2f}")
                    else:
                        texts.append("Unknown Person")
                    
                    age_range = face_data.get('age_range', 'Unknown')
                    gender = face_data.get('gender', 'Unknown')
                    confidence = face_data.get('confidence', 0.0)
                    
                    texts.append(f"Age: {age_range}")
                    texts.append(f"Gender: {gender}")
                    texts.append(f"Conf: {confidence:.2f}")
                    
                    # Draw text
                    y_offset = y1 - 10
                    for text in texts:
                        cv2.putText(annotated_image, text, (x1, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        y_offset -= 20
                        
                except Exception as e:
                    print(f"Error drawing annotation: {e}")
                    continue
            
            return annotated_image
            
        except Exception as e:
            print(f"Error in draw_annotations: {e}")
            return image